"""数据层、增强与训练器的不变量测试。

这些测试的判据尽量用 C++ 侧的实现当“神谕”（`apply_move` / `legal_mask`），
而不是在 Python 里再算一遍：记录布局与规则语义一旦和 C++ 分叉，训练就会在错误的数据上
安静地进行，这类错误只有用独立判据才能发现。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import r2048_core
import torch

from r2048.data.dataset import Batch, BatchStream, MultiSourceReader, make_split, open_sources
from r2048.data.shards import BatchReader, record_dtype
from r2048.nn.augment import apply_boards, augment_policy, transform_policy_targets
from r2048.nn.training import (
    DatasetSpec,
    TrainConfig,
    _validate_tree_label_sources,
    policy_loss,
    run_training,
)


def _tree_zeros(count: int, nodes: int = 0) -> dict:
    """Batch 的树内节点字段（v3）。默认 K=0——这些用例测的是根候选与损失口径，
    树内节点由 `test_ranking_loss_covers_tree_nodes` 单独覆盖。"""
    return {
        "tree_afterstate": torch.zeros(count, nodes, 4, 16, dtype=torch.uint8),
        "tree_value": torch.zeros(count, nodes, 4),
        "tree_reward": torch.zeros(count, nodes, 4),
        "tree_visits": torch.zeros(count, nodes, 4),
        "tree_depth": torch.zeros(count, nodes),
        "tree_rollout_value": torch.zeros(count, nodes, 4),
        "tree_legal": torch.zeros(count, nodes, 4),
        "tree_count": torch.zeros(count, dtype=torch.int64),
    }


def training_config(data: str | Path, output: str | Path, **overrides) -> TrainConfig:
    """测试里的 TrainConfig 构造：数据来源写成 DatasetSpec，避免每个断言点都重复一遍。"""
    # 小样本回归固定随机序列，避免默认自动随机让验证集偶然为空。
    payload = {"datasets": [DatasetSpec(path=str(data))], "output": str(output), "seed": 0, "split_seed": 0}
    payload.update(overrides)
    return TrainConfig(**payload)


TINY_SAMPLES = 6000
SHARD_SAMPLES = 2000


def _r2048_binary() -> str:
    test_binary = os.environ.get("R2048_TEST_BINARY")
    if not test_binary:
        pytest.skip("未设置 R2048_TEST_BINARY：通过 make test 运行")
    binary = test_binary.replace("r2048_tests", "r2048")
    assert Path(binary).exists(), f"缺少 CLI 二进制 {binary}"
    return binary


@pytest.fixture(scope="module")
def search_batch_dir(tmp_path_factory) -> Path:
    """带搜索记录的批次：用于检验策略目标的口径（随机策略批次里 visits 全 0）。"""
    target = tmp_path_factory.mktemp("data") / "mcts"
    subprocess.run(
        [
            _r2048_binary(),
            "generate",
            "--policy",
            "mcts",
            "--simulations",
            "8",
            "--samples",
            "4000",
            "--seed",
            "20261013",
            "--shard-samples",
            "4000",
            "--out",
            str(target),
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


@pytest.fixture(scope="module")
def batch_dir(tmp_path_factory) -> Path:
    """现场生成一个小数据批：顺带验证 C++ 生成 → Python 读取这条跨语言路径。"""
    target = tmp_path_factory.mktemp("data") / "random"
    subprocess.run(
        [
            _r2048_binary(),
            "generate",
            "--policy",
            "random",
            "--samples",
            str(TINY_SAMPLES),
            "--seed",
            "424242",
            "--shard-samples",
            str(SHARD_SAMPLES),
            "--out",
            str(target),
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


def _apply(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    outcome = r2048_core.apply_move(np.ascontiguousarray(board, dtype=np.uint8), int(action))
    return np.asarray(outcome["board"], dtype=np.uint8), int(outcome["score"]), bool(outcome["valid"])


def test_record_layout_matches_cpp_rules(batch_dir: Path) -> None:
    """逐条样本用 C++ 规则复核：afterstate、合并得分、合法掩码都必须对得上。

    这条测试守的是 80 字节记录布局（`cpp/dataio/sample.h` 的偏移表）——Python 侧的
    `RECORD_DTYPE` 若与 C++ 分叉，会读出一堆“看起来像棋盘”的垃圾，训练照跑不误。
    """
    reader = BatchReader(batch_dir)
    assert reader.samples == TINY_SAMPLES
    records = reader.gather(np.arange(min(600, reader.samples)))
    checked = 0
    for record in records:
        decision = record["decision"]
        afterstate = record["afterstate"]
        action = int(record["action"])
        mask = int(record["legal_mask"])
        expected_mask = int(r2048_core.legal_mask(decision))
        assert mask == expected_mask, "样本里的合法掩码与 C++ 规则不一致"
        moved, score, valid = _apply(decision, action)
        assert valid and (mask >> action) & 1 == 1, "样本里的动作必须合法"
        assert np.array_equal(moved, afterstate), "afterstate 与 C++ 规则不一致"
        assert score == int(record["move_score"]), "合并得分与 C++ 规则不一致"
        checked += 1
    assert checked > 0


def test_episode_labels_are_consistent(batch_dir: Path) -> None:
    """回报标签口径：同一局内 `标签_t = 得分_{t+1} + 标签_{t+1}`，末步为 0。

    这是训练目标的核心契约；标签算错时损失照样下降，只有按局重算才能发现。
    """
    reader = BatchReader(batch_dir)
    records = reader.gather(np.arange(reader.samples))
    order = np.lexsort((records["step"], records["episode"]))
    episodes = records["episode"][order]
    steps = records["step"][order]
    scores = records["move_score"][order].astype(np.int64)
    labels = records["mc_return"][order].astype(np.int64)

    checked = 0
    for index in range(len(order) - 1):
        if episodes[index] != episodes[index + 1]:
            assert labels[index] == 0, "一局最后一步的标签必须是 0"
            continue
        assert steps[index + 1] == steps[index] + 1, "同一局的步号必须连续"
        assert labels[index] == scores[index + 1] + labels[index + 1], "回报标签递推关系不成立"
        checked += 1
    assert checked > 0


def test_split_keeps_episodes_whole(batch_dir: Path) -> None:
    reader = BatchReader(batch_dir)
    first = make_split(reader, val_fraction=0.2, seed=7)
    second = make_split(reader, val_fraction=0.2, seed=7)
    assert np.array_equal(first.train, second.train), "同一 seed 的划分必须可复现"
    assert np.array_equal(first.val, second.val)

    episodes = reader.episode_column()
    train_episodes = set(episodes[first.train].tolist())
    val_episodes = set(episodes[first.val].tolist())
    assert train_episodes and val_episodes
    assert not (train_episodes & val_episodes), "同一局不能同时出现在训练与验证集里"

    other = make_split(reader, val_fraction=0.2, seed=99)
    assert not np.array_equal(other.val, first.val), "换 seed 应当换划分"


def test_batch_stream_normalises_targets(batch_dir: Path) -> None:
    reader = BatchReader(batch_dir)
    split = make_split(reader, val_fraction=0.2, seed=1)
    stream = BatchStream(reader, split.train, batch_size=128, shuffle=True, seed=3)
    batch = next(iter(stream))
    assert batch.decision.shape[1] == 16
    assert batch.legal_mask.shape[1] == 4
    # 随机策略数据没有搜索记录，策略目标必须全 0（训练器据此排除这批样本）。
    assert float(batch.visits.abs().max()) == 0.0
    assert float(batch.policy_target.abs().max()) == 0.0
    # 归一化后的价值目标应当量纲合理（分数 / 1024）。
    assert float(batch.value_target.abs().max()) < 100.0
    assert policy_loss(torch.zeros(128, 4), batch, batch.policy_target) is None, (
        "无搜索记录时不应产生策略损失"
    )


def _write_v4_tree_batch(source: Path, target: Path, *, nodes: int, plays: int, seed: int) -> Path:
    """以 C++ 生成的根记录为底稿，写一个可经真实清单和 mmap 读取的 v4 小批次。"""
    reader = BatchReader(source)
    original = reader.gather(np.arange(4))
    records = np.zeros(4, dtype=record_dtype(nodes))
    for field in original.dtype.names:
        records[field] = original[field]
    if nodes:
        records["tree_count"] = nodes
        for slot in range(nodes):
            records[f"tree{slot}_legal_mask"] = 0b0101
            records[f"tree{slot}_count"] = 2
            records[f"tree{slot}_depth"] = 2 * (slot + 1)
            records[f"tree{slot}_rollout"][:, 0] = 100 + slot
            records[f"tree{slot}_rollout"][:, 1] = 999  # 非法方向必须在解码后归零。
            records[f"tree{slot}_rollout"][:, 2] = 200 + slot
    target.mkdir()
    records.tofile(target / "shard.bin")
    manifest = dict(reader.manifest.data)
    manifest.update(
        samples=4,
        episodes=1,
        seed=seed,
        tree_nodes=nodes,
        tree_label_plays=plays,
        record_bytes=records.dtype.itemsize,
        shards=[{"path": "shard.bin", "samples": 4}],
    )
    (target / "manifest.json").write_text(json.dumps(manifest))
    return target


def test_v4_tree_decode_and_source_labels(search_batch_dir: Path, tmp_path: Path) -> None:
    """K 维必须在堆叠后掩码；来源校验不能只看排列中的第一批。"""
    labeled = _write_v4_tree_batch(search_batch_dir, tmp_path / "labeled", nodes=2, plays=8, seed=101)
    unlabeled = _write_v4_tree_batch(search_batch_dir, tmp_path / "unlabeled", nodes=2, plays=0, seed=102)
    root_only = _write_v4_tree_batch(search_batch_dir, tmp_path / "root-only", nodes=0, plays=0, seed=103)

    reader = MultiSourceReader([labeled])
    batch = BatchStream(reader, np.arange(4), batch_size=4, shuffle=False).decode(np.arange(4))
    assert batch.tree_rollout_value.shape == (4, 2, 4)
    assert torch.all(batch.tree_rollout_value[:, :, 1] == 0)
    assert torch.all(batch.tree_rollout_value[:, 0, 0] == 100 / 1024)
    root_batch = BatchStream(
        MultiSourceReader([root_only]), np.arange(4), batch_size=4, shuffle=False
    ).decode(np.arange(4))
    assert root_batch.tree_rollout_value.shape == (4, 0, 4)

    config = training_config(labeled, tmp_path / "run", mode="value", tree_label_source="rollout")
    for paths in ([labeled, unlabeled], [unlabeled, labeled]):
        with pytest.raises(ValueError, match="unlabeled.*tree_label_plays=0|tree_label_plays=0.*unlabeled"):
            _validate_tree_label_sources(config, MultiSourceReader(paths))
        mixed_config = training_config(
            labeled,
            tmp_path / f"mixed-{paths[0].name}",
            mode="value",
            device="cpu",
            datasets=[DatasetSpec(path=str(path)) for path in paths],
        )
        with pytest.raises(ValueError, match="unlabeled"):
            run_training(mixed_config)
    _validate_tree_label_sources(config, MultiSourceReader([root_only]))
    _validate_tree_label_sources(
        training_config(labeled, tmp_path / "qmean", mode="value", tree_label_source="qmean"),
        MultiSourceReader([labeled, unlabeled]),
    )


def test_policy_target_is_a_distribution(search_batch_dir: Path) -> None:
    """策略目标必须是概率分布，且与 in_flight / 根求值开销无关。

    引擎里根节点自身的求值会消耗一次模拟但不落到任何边上，因此“按模拟次数归一化”
    会让目标权重小于 1（实测 56/64~63/64）。这里守住按访问和归一化的口径。
    """
    reader = BatchReader(search_batch_dir)
    split = make_split(reader, val_fraction=0.0)
    stream = BatchStream(reader, split.train, batch_size=512, shuffle=False)
    batch = next(iter(stream))
    has_search = batch.visits.sum(dim=-1) > 0
    assert bool(has_search.any()), "搜索批次里应当有带访问数的记录"
    sums = batch.policy_target[has_search].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "策略目标必须归一化到 1"
    if bool((~has_search).any()):
        assert bool(batch.policy_target[~has_search].abs().max() == 0), "无搜索记录的目标必须为 0"
    # 访问数之和不可能超过模拟次数（根求值那次不落在边上）。
    assert bool((batch.visits.sum(dim=-1) <= batch.visits.new_tensor(8.0) + 1e-6).all())
    assert policy_loss(torch.randn(int(has_search.sum()), 4), batch, batch.policy_target) is not None


def test_candidate_block_matches_rules_and_main_fields(search_batch_dir: Path) -> None:
    """候选块的每个字段都要能用规则或主记录的字段交叉核对。

    候选监督的全部风险都在“写对了没有”：槽位错位、减错立即得分、把未访问的边当成 0，
    这些都不会让训练报错，只会安静地把价值头引到错误的候选上。
    """
    reader = BatchReader(search_batch_dir)
    records = reader.gather(np.arange(reader.samples))
    checked = 0
    for record in records:
        decision = record["decision"]
        action = int(record["action"])
        mask = int(record["legal_mask"])
        assert int(record["candidate_count"]) == mask.bit_count(), "候选数量必须等于合法方向数"
        for candidate in range(4):
            if not (mask >> candidate) & 1:
                assert int(record["candidate_value"][candidate]) == 0, "非法方向的槽位应保持为空"
                assert int(record["candidate_visits"][candidate]) == 0
                continue
            moved, score, _ = _apply(decision, candidate)
            assert np.array_equal(
                np.asarray(record["candidate_afterstate"][candidate], dtype=np.uint8), moved
            ), "候选 afterstate 必须与规则一致"
            assert int(record["candidate_reward"][candidate]) == score, "候选立即得分必须与规则一致"
            checked += 1
        # 实际执行动作对应的候选槽，必须与主记录的 afterstate 一致（两处独立写入的交叉核对）。
        assert np.array_equal(
            np.asarray(record["candidate_afterstate"][action], dtype=np.uint8),
            np.asarray(record["afterstate"], dtype=np.uint8),
        ), "动作所在候选槽的 afterstate 必须等于主记录的 afterstate"
    assert checked > 0

    split = make_split(reader, val_fraction=0.0)
    batch = next(iter(BatchStream(reader, split.train, batch_size=256, shuffle=False)))
    assert batch.has_candidates, "搜索批次应当带候选监督"
    legal = batch.legal_mask.bool()
    expected_weight = torch.sqrt(batch.candidate_visits) * legal
    assert torch.allclose(batch.candidate_weight, expected_weight, atol=1e-5), (
        "候选权重必须是 sqrt(访问数) 且非法方向为 0"
    )


def test_candidate_value_loss_and_metrics() -> None:
    """候选损失与排序指标：完美预测必须给出 0 损失、100% 命中、0 后悔。"""
    from r2048.nn.training import TrainConfig, _Metrics, _value_loss

    torch.manual_seed(0)
    count = 32
    target = torch.randn(count, 4)
    weight = torch.rand(count, 4) * 3.0
    weight[torch.rand(count, 4) < 0.3] = 0.0  # 未访问/非法的候选权重为 0
    batch = Batch(
        decision=torch.zeros(count, 16, dtype=torch.uint8),
        afterstate=torch.zeros(count, 16, dtype=torch.uint8),
        action=torch.zeros(count, dtype=torch.int64),
        legal_mask=(weight > 0).float(),
        visits=weight,
        policy_target=torch.zeros(count, 4),
        value_target=torch.zeros(count),
        move_score=torch.zeros(count),
        episode=torch.zeros(count, dtype=torch.int64),
        candidate_afterstate=torch.zeros(count, 4, 16, dtype=torch.uint8),
        candidate_value=target,
        candidate_reward=torch.zeros(count, 4),
        candidate_visits=weight,
        candidate_weight=weight,
        **_tree_zeros(count),
    )
    config = TrainConfig(datasets=[DatasetSpec(path="x")], output="y", mode="value", value_loss="huber")

    perfect = target.reshape(-1)
    loss, prediction = _value_loss(perfect, batch, use_candidates=True, config=config)
    assert float(loss) == pytest.approx(0.0, abs=1e-6), "完美预测的加权 Huber 损失必须是 0"

    metrics = _Metrics(torch.device("cpu"), 1024.0)
    metrics.add_candidates(prediction, target, weight)
    assert metrics.candidate_mae() == pytest.approx(0.0, abs=1e-3)
    assert metrics.pair_agreement() == pytest.approx(1.0, abs=1e-6), "完美预测的成对排序一致率为 1"
    assert metrics.best_candidate_hit() == pytest.approx(1.0, abs=1e-6), "完美预测必然命中最优候选"
    assert metrics.candidate_regret() == pytest.approx(0.0, abs=1e-3), "完美预测的后悔值为 0"

    noisy = target.reshape(-1) + 5.0 * torch.randn(count * 4)
    noisy_loss, noisy_prediction = _value_loss(noisy, batch, use_candidates=True, config=config)
    assert float(noisy_loss) > float(loss), "带噪声的预测损失必须更大"
    noisy_metrics = _Metrics(torch.device("cpu"), 1024.0)
    noisy_metrics.add_candidates(noisy_prediction, target, weight)
    assert noisy_metrics.best_candidate_hit() < 1.0, "带噪声时不可能命中所有最优候选"


def test_candidate_ranking_with_and_without_immediate_reward() -> None:
    """候选排序的两套口径：引擎的选择值是 r + A，而价值头学的只是 A（ADR 0034）。

    两个根各是一侧的典型，方向相反——说明"用哪套口径度量"不是细节，它能直接改变判决：

    * **根 0（用户举的例子）**：up 立即合 512 分、未来价值 1000；left 不合并、未来价值 1200。
      按 A 排选 left，按 r+A 排选 up（1512 > 1200）。这里让预测的 A 排序与目标 A 相反：
      **A 口径记为未命中（白丢 200 分），r+A 口径记为命中**（引擎会下对）；
    * **根 1（反向）**：目标 A 里 up 更高（1200 > 1000），但 left 那一手合 512 分，
      所以目标的 argmax(r+A) 是 left。预测的 A 排序正确、却把 up 高估到 1600：
      **A 口径记为命中，r+A 口径记为未命中（按它走丢 312 分 Q）**。

    同时钉住一条不变量：加 r 只平移预测与目标，**误差不变**——所以 MAE 与后悔值的量级与
    口径无关，变的只有"谁排第一"。这也是为什么只需要给排序指标加一套口径。
    """
    from r2048.nn.training import _Metrics

    scale = 1024.0

    def one_root(
        target_a: list[float], reward: list[float], predicted_a: list[float]
    ) -> tuple[_Metrics, _Metrics]:
        """一个根，合法方向取前三个（第 4 槽留空）；返回同一份输入在两种口径下的指标。"""

        def as_row(values: list[float]) -> torch.Tensor:
            """补足第 4 槽（非法方向）并归一化到 score_scale。"""
            return torch.tensor([list(values) + [0.0]]) / scale

        legal = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        metrics = _Metrics(torch.device("cpu"), scale)
        metrics.add_candidates(as_row(predicted_a), as_row(target_a), legal, as_row(reward))
        return metrics

    # 根 0：up(r=512, A=1000) / left(r=0, A=1200) / down(0,0)，预测把 up 的 A 高估到 1300。
    up_merge = one_root([1000.0, 1200.0, 0.0], [512.0, 0.0, 0.0], [1300.0, 1200.0, 0.0])
    assert up_merge.best_candidate_hit() == pytest.approx(0.0), "A 口径：预测的 argmax(A) 与目标相反"
    assert up_merge.candidate_regret() == pytest.approx(200.0, abs=1e-3), "A 口径白丢 200 分"
    assert up_merge.q_best_candidate_hit() == pytest.approx(1.0), "r+A 口径：引擎的判决没被动摇"
    assert up_merge.q_candidate_regret() == pytest.approx(0.0, abs=1e-3)
    assert up_merge.pair_agreement() == pytest.approx(2.0 / 3.0), "A 口径：up/left 这一对被判为不一致"
    assert up_merge.q_pair_agreement() == pytest.approx(1.0), "r+A 口径：三对全一致"

    # 根 1：反向——目标 A 里 up 更高，但 left 那一手合 512 分，所以目标 Q 的最优是 left。
    left_merge = one_root([1200.0, 1000.0, 0.0], [0.0, 512.0, 0.0], [1600.0, 1000.0, 0.0])
    assert left_merge.best_candidate_hit() == pytest.approx(1.0), "A 口径：预测的 argmax(A) 正确"
    assert left_merge.candidate_regret() == pytest.approx(0.0, abs=1e-3)
    assert left_merge.q_best_candidate_hit() == pytest.approx(0.0), "r+A 口径：把大合并走丢了"
    assert left_merge.q_candidate_regret() == pytest.approx(312.0, abs=1e-3), "少拿 312 分 Q（1512−1200）"
    assert left_merge.pair_agreement() == pytest.approx(1.0)
    assert left_merge.q_pair_agreement() == pytest.approx(2.0 / 3.0)

    # 不给 reward 时 r+A 一整套必须是 NaN（而不是悄悄等于 A 口径）——旧调用点不能因此失真。
    plain = _Metrics(torch.device("cpu"), scale)
    plain.add_candidates(
        torch.tensor([[1.5, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.2, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    )
    assert np.isnan(plain.q_pair_agreement()) and np.isnan(plain.q_best_candidate_hit())
    assert np.isnan(plain.q_candidate_regret())

    # 误差不变量：加 r 只平移预测与目标 → 加权 MAE 必须完全一致（所以不需要第二套 MAE）。
    # 这条不是形式主义：它说明"两套口径"只影响判决、不影响误差，指标不会因为换了口径而
    # 看起来变好或变坏。
    same_pred = torch.tensor([[1.5, 1.0, 0.0, 0.0]])
    same_target = torch.tensor([[1.0, 1.2, 0.0, 0.0]])
    same_reward = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    legal = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    with_reward = _Metrics(torch.device("cpu"), scale)
    with_reward.add_candidates(same_pred, same_target, legal, same_reward)
    without = _Metrics(torch.device("cpu"), scale)
    without.add_candidates(same_pred, same_target, legal)
    assert with_reward.candidate_mae() == pytest.approx(without.candidate_mae(), abs=1e-9)


def test_ranking_loss_only_punishes_decisions() -> None:
    """Q 空间的同根成对排序项：只罚"改变引擎判决"的误差（ADR 0035）。

    手算表（一个根：up(r=512, A=1000) / left(r=0, A=1200) / down(0,0)，目标 Q = [1512, 1200]）：

    | 预测的 A | A 口径 MAE | 排序项 | 语义 |
    |---|---|---|---|
    | [1000, 1200]（与目标一致） | 0 | 0 | 什么都对 |
    | [1600, 1000]（把 up 高估 600） | 400 | **0** | A 的排序错了，但引擎仍选 up（1512 < 2112） |
    | [1000, 1600]（把 left 高估 600） | 200 | **>0** | A 的排序"对"，但引擎会改选 left（1600 > 1512） |

    第二行与第三行是这一项存在的全部理由：Huber 罚的是"像不像教师的 A"，而排序项罚的是
    "引擎会不会走出教师的判决"——两者的差别在 r 能改写 argmax 的那 12.9% 的根上（ADR 0034）。
    """
    from r2048.nn.training import _ranking_loss

    scale = 1024.0
    count = 1
    target_a = torch.tensor([[1000.0, 1200.0, 0.0, 0.0]]) / scale
    reward = torch.tensor([[512.0, 0.0, 0.0, 0.0]]) / scale
    batch = Batch(
        decision=torch.zeros(count, 16, dtype=torch.uint8),
        afterstate=torch.zeros(count, 16, dtype=torch.uint8),
        action=torch.zeros(count, dtype=torch.int64),
        legal_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        visits=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        policy_target=torch.zeros(count, 4),
        value_target=torch.zeros(count),
        move_score=torch.zeros(count),
        episode=torch.zeros(count, dtype=torch.int64),
        candidate_afterstate=torch.zeros(count, 4, 16, dtype=torch.uint8),
        candidate_value=target_a,
        candidate_reward=reward,
        candidate_visits=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        candidate_weight=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        **_tree_zeros(count),
    )

    def loss(predicted_a: list[float]) -> float:
        prediction = torch.tensor([predicted_a]) / scale
        return float(_ranking_loss(prediction, None, batch, scale, "qmean")) * scale

    assert loss([1000.0, 1200.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-3), "完美预测必须为 0"
    assert loss([1600.0, 1000.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-3), (
        "A 的排序错了但引擎的判决没变（2112 > 1512，仍选 up）⇒ 不该罚"
    )
    wrong_decision = loss([1000.0, 1600.0, 0.0, 0.0])
    assert wrong_decision > 5.0, "把 left 高估到让引擎改选它 ⇒ 必须罚（1512 vs 1600）"
    assert loss([800.0, 1800.0, 0.0, 0.0]) > wrong_decision, "错得越离谱，罚得越重"
    # 合法失败的极端情形仍有界：上界是"这一对的 Q 间隔"（成对后悔的定义）。
    assert wrong_decision < 312.0, "软 hinge 的有界性：不会超过该对的目标 Q 间隔"


def test_ranking_loss_covers_tree_nodes() -> None:
    """排序项必须覆盖**树内节点**，不只是轨迹根（ADR 0041）。

    价值头在搜索里被直接使用的位置是叶节点与未被访问的边（那里的 Q 就是 r + A），而根那一点
    有几百次模拟、搜索自己就把排序算出来了（ADR 0035）。这条用例钉住三件事：

    1. 树内节点的候选**参与**这一项：把它们的排序预测搞错，损失必须上升；
    2. 未访问的边（`tree_visits == 0`）**不参与**——它的"标签"是教师自己的一次性估计，
       训它等于让学生复制自己（与根候选用 candidate_weight>0 同一口径）；
    3. K=0 的批（没有树内块）行为与从前完全一致：传 None 不改变任何数。
    """
    from r2048.nn.training import _ranking_loss

    scale = 1024.0
    count = 1
    legal = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # 树内节点一个：up(r=512, A=1000) / left(r=0, A=1200) ⇒ 目标 Q = [1512, 1200]，最优 up。
    tree_value = torch.tensor([[[1000.0, 1200.0, 0.0, 0.0]]]) / scale
    tree_reward = torch.tensor([[[512.0, 0.0, 0.0, 0.0]]]) / scale
    tree_visits = torch.tensor([[[5.0, 3.0, 0.0, 0.0]]])
    batch = Batch(
        decision=torch.zeros(count, 16, dtype=torch.uint8),
        afterstate=torch.zeros(count, 16, dtype=torch.uint8),
        action=torch.zeros(count, dtype=torch.int64),
        legal_mask=legal,
        visits=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        policy_target=torch.zeros(count, 4),
        value_target=torch.zeros(count),
        move_score=torch.zeros(count),
        episode=torch.zeros(count, dtype=torch.int64),
        candidate_afterstate=torch.zeros(count, 4, 16, dtype=torch.uint8),
        candidate_value=torch.zeros(count, 4),  # 根候选没有监督（权重 0）⇒ 这一项只由树内节点决定
        candidate_reward=torch.zeros(count, 4),
        candidate_visits=torch.zeros(count, 4),
        candidate_weight=torch.zeros(count, 4),
        tree_afterstate=torch.zeros(count, 1, 4, 16, dtype=torch.uint8),
        tree_value=tree_value,
        tree_reward=tree_reward,
        tree_visits=tree_visits,
        tree_depth=torch.ones(count, 1),
        # 独立标签：与 tree_value 同值，好让"换标签来源"这件事不影响这条用例的算术（它测的是
        # 排序项的形状与语义，不是标签来源）。
        tree_rollout_value=tree_value,
        tree_legal=torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
        tree_count=torch.ones(count, dtype=torch.int64),
    )

    def loss(tree_predicted_a: list[float]) -> float:
        prediction = torch.tensor([tree_predicted_a]) / scale  # (1, 4)？—— 见下
        return (
            float(_ranking_loss(torch.zeros(count, 4), prediction.reshape(1, 1, 4), batch, scale, "rollout"))
            * scale
        )

    assert loss([1000.0, 1200.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-3), "树内节点的完美预测必须为 0"
    assert loss([1600.0, 1000.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-3), (
        "A 排序错但引擎判决没变（2112 > 1512）⇒ 树内节点同样不该罚"
    )
    assert loss([1000.0, 1600.0, 0.0, 0.0]) > 5.0, "让引擎在树内节点上改选 left ⇒ 必须罚"

    # 两种标签来源对"未访问的边"的处理必须**不同**——这正是 v4 存在的理由（ADR 0043）：
    #   qmean：访问数为 0 的边其 `tree_value` 就是 pure one_ply（头自己的输出）⇒ 必须排除；
    #   rollout：独立标签与访问数无关 ⇒ 未访问的边**参与**（那正是头说了算的区域）。
    batch_extra = Batch(**{**batch.__dict__, "tree_visits": torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])})
    swapped = torch.tensor([[1000.0, 1600.0, 0.0, 0.0]]) / scale
    assert float(
        _ranking_loss(torch.zeros(count, 4), swapped.reshape(1, 1, 4), batch_extra, scale, "qmean")
    ) == pytest.approx(0.0, abs=1e-6), "qmean 口径：left 的访问数为 0 ⇒ 它的排序不参与"
    assert (
        float(_ranking_loss(torch.zeros(count, 4), swapped.reshape(1, 1, 4), batch_extra, scale, "rollout"))
        * scale
        > 5.0
    ), "rollout 口径：未访问的边有独立标签 ⇒ 它参与，且把 up 搞错必须被罚"

    # 没有树内块时（tree_prediction=None）这一项只看根候选：这里根候选权重全 0 ⇒ 恰好为 0。
    assert float(_ranking_loss(torch.zeros(count, 4), None, batch, scale, "rollout")) == pytest.approx(
        0.0, abs=1e-6
    )


def test_improved_policy_target_arithmetic_and_limits() -> None:
    """一步策略改进的目标：手算值、β→∞ 的极限、无候选的回退。

    这里的两个极限是本算子的**唯一**可离线验证的性质（"改进后是不是更好"必须靠对局评测，
    因为"用同一个 Q 度量改进"是恒真的，见 ADR 0032）：
    * β→∞ 必须逐样本退化成原访问分布——这条一旦不成立，A/B 的差异就分不清是算子还是别的；
    * 没有候选监督的记录必须原样保留 π̂，否则混入随机策略批次时会把垃圾写进目标。
    """
    from r2048.nn.training import improved_policy_target, model_id_for

    count = 2
    legal = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    visits = torch.tensor([[8.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    reward = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[100.0, 90.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    # 第二条记录没有候选监督（权重为 0），它的目标必须保持原样。
    weight = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    fallback = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    batch = Batch(
        decision=torch.zeros(count, 16, dtype=torch.uint8),
        afterstate=torch.zeros(count, 16, dtype=torch.uint8),
        action=torch.zeros(count, dtype=torch.int64),
        legal_mask=legal,
        visits=visits,
        policy_target=fallback,
        value_target=torch.zeros(count),
        move_score=torch.zeros(count),
        episode=torch.zeros(count, dtype=torch.int64),
        candidate_afterstate=torch.zeros(count, 4, 16, dtype=torch.uint8),
        candidate_value=value,
        candidate_reward=reward,
        candidate_visits=visits,
        candidate_weight=weight,
        **_tree_zeros(count),
    )
    # Q = r + A = [110, 90]；π̂ = [0.8, 0.2]；V = 106；Â = [4, −16]；同根 std = 10。
    # β=1 ⇒ 权重 = [1, e^{−2}] ⇒ π̃ ∝ [0.8, 0.2·e^{−2}, 0, 0]。
    raw = torch.tensor([0.8, 0.2 * float(torch.exp(torch.tensor(-2.0))), 0.0, 0.0])
    expected = raw / raw.sum()
    target = improved_policy_target(batch, beta=1.0, floor=0.0)
    assert torch.allclose(target[0], expected, atol=1e-5), f"π̃ 与手算不符: {target[0]}"
    assert torch.allclose(target[1], fallback[1], atol=1e-9), "无候选监督的记录必须保持 π̂"

    # β→∞：指数项恒为 0，权重全 1，π̃ 必须逐样本等于 π̂ = visits/Σvisits（不是"接近"）。
    limit = improved_policy_target(batch, beta=float("inf"), floor=0.0)
    assert torch.allclose(limit[0], torch.tensor([0.8, 0.2, 0.0, 0.0]), atol=1e-7), "β→∞ 必须退化回访问分布"

    # 目标必须是概率分布，且带候选的那条记录在非法方向上恒为 0
    # （第 2 条走的是回退路径、原样保留传入的 π̂，真实数据里它本来就只落在合法方向上）。
    for candidate in (target, limit):
        assert torch.allclose(candidate.sum(dim=-1), torch.ones(count), atol=1e-5)
        assert float(candidate[0][~legal[0].bool()].abs().max()) == 0.0

    # model_id 必须能区分同一个 mode + 同一批数据下的两种策略目标（ADR 0030 的同名事故）。
    plain = TrainConfig(datasets=[DatasetSpec(path="/data/批")], output="x", mode="policy")
    improved = TrainConfig(
        datasets=[DatasetSpec(path="/data/批")], output="x", mode="policy", policy_target="improved"
    )
    assert model_id_for(plain, "candidate") != model_id_for(improved, "candidate")
    assert model_id_for(improved, "candidate") == "policy-批+improved-b1"
    # 挑选口径**总是**进名字：默认值换过（best_hit → q_best_hit），只给非默认值加后缀会让
    # 新默认训出的 `value-<数据>` 与历史上按 A 口径挑出的同名（ADR 0030 §5.7 的同名事故）。
    default_value = TrainConfig(datasets=[DatasetSpec(path="/data/批")], output="x", mode="value")
    legacy = TrainConfig(
        datasets=[DatasetSpec(path="/data/批")],
        output="x",
        mode="value",
        value_selection="best_hit",
        value_ranking_weight=0.0,  # 这一段只测挑选口径，把排序项的默认后缀摘掉
    )
    # 默认还带排序项的权重（ADR 0035 默认 λ=1）：三个旋钮都要能从名字里读出来。
    assert model_id_for(default_value, "candidate") == "value-批+q_best_hit+label-rollout+rank1"
    no_rank = TrainConfig(
        datasets=[DatasetSpec(path="/data/批")], output="x", mode="value", value_ranking_weight=0.0
    )
    assert model_id_for(no_rank, "candidate") == "value-批+q_best_hit"
    assert model_id_for(legacy, "candidate") == "value-批+best_hit"
    assert model_id_for(default_value, "candidate") != model_id_for(legacy, "candidate")
    # 策略模式不训价值头，挑选口径与它无关，不该进名字。
    policy_only = TrainConfig(
        datasets=[DatasetSpec(path="/data/批")], output="x", mode="policy", value_selection="best_hit"
    )
    assert model_id_for(policy_only, "candidate") == "policy-批"


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("reflection", range(2))
def test_augmentation_commutes_with_rules(rotation: int, reflection: int) -> None:
    """D4 增强的核心不变量：变换棋盘与变换动作方向必须等价。

    对任意棋盘、任意方向：
        transform(apply_move(board, action)) == apply_move(transform(board), transform_action(action))
    只有这条成立，策略标签才没有指错方向。这里直接驱动变换原语（而不是随机包装），
    用 C++ 规则当判据逐个变换核对。
    """
    boards = torch.tensor(
        [
            [1, 2, 0, 0, 3, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0],
            [2, 2, 3, 0, 0, 1, 1, 0, 4, 0, 0, 2, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
    )
    legal_mask = torch.zeros(len(boards), 4)
    for index in range(len(boards)):
        mask = int(r2048_core.legal_mask(boards[index].numpy()))
        for action in range(4):
            legal_mask[index, action] = float((mask >> action) & 1)
    visits = torch.zeros(len(boards), 4)
    visits[:, 0] = 1.0  # 全部押在 UP 上，便于检查方向标签是否被同步变换

    rotations = torch.full((len(boards),), rotation, dtype=torch.long)
    reflections = torch.full((len(boards),), reflection, dtype=torch.long)
    transformed = apply_boards(boards, rotations, reflections)
    _, transformed_visits = transform_policy_targets(legal_mask, visits, rotations, reflections)

    def manual_transform(board: torch.Tensor) -> np.ndarray:
        shaped = board.reshape(4, 4).numpy()
        rotated = np.rot90(shaped, rotation)  # 与实现一致：torch.rot90 语义为逆时针
        if reflection:
            rotated = np.fliplr(rotated)
        return np.ascontiguousarray(rotated.reshape(16).astype(np.uint8))

    # 方向标签：逆时针旋转一次把方向编号减一，水平镜像再交换左右。
    rotated_direction = (0 - rotation) % 4
    expected_direction = (-rotated_direction) % 4 if reflection else rotated_direction

    for index in range(len(boards)):
        board = boards[index].numpy()
        assert np.array_equal(transformed[index].numpy(), manual_transform(boards[index]))
        assert int(transformed_visits[index].argmax()) == expected_direction

        # 用规则核对：原局面在 UP 上的 afterstate 变换后，必须等于变换后的局面在
        # “变换后方向”上的 afterstate。这一条把棋盘变换与方向标签绑在一起。
        expected_afterstate = manual_transform(torch.from_numpy(_apply(board, 0)[0]))
        actual_afterstate, _, _ = _apply(
            np.asarray(transformed[index].numpy(), dtype=np.uint8), expected_direction
        )
        assert np.array_equal(actual_afterstate, expected_afterstate), (
            f"rotation={rotation} reflection={reflection} 时增强与规则不交换"
        )


def test_random_augment_keeps_board_and_mask_consistent() -> None:
    """随机增强（训练里实际使用的那条路径）也必须保持棋盘与合法掩码一致。

    判据用 C++ 的 `legal_mask`：把增强后的棋盘交给 C++ 重新算一次掩码，
    必须与增强后的掩码逐位相同——否则棋盘被某个变换、掩码被另一个变换。
    """
    torch.manual_seed(0)
    boards = torch.tensor(
        [
            [1, 2, 0, 0, 3, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0],
            [0, 0, 4, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 3, 0, 0],
        ],
        dtype=torch.uint8,
    )
    legal_mask = torch.zeros(len(boards), 4)
    for index in range(len(boards)):
        mask = int(r2048_core.legal_mask(boards[index].numpy()))
        for action in range(4):
            legal_mask[index, action] = float((mask >> action) & 1)
    visits = torch.zeros(len(boards), 4)
    visits[:, 1] = 1.0

    rng = torch.Generator(device="cpu").manual_seed(20261006)
    for _ in range(16):
        transformed, mask, target = augment_policy(boards, legal_mask, visits, rng)
        for index in range(len(boards)):
            expected_mask = int(r2048_core.legal_mask(transformed[index].numpy()))
            actual_mask = 0
            for action in range(4):
                actual_mask |= (int(mask[index, action].item()) & 1) << action
            assert actual_mask == expected_mask, "增强后的棋盘与合法掩码不一致"
            assert float(target[index].sum()) == pytest.approx(1.0), "策略目标必须保持归一化"
            # 棋盘上的块值集合（指数）不应因变换而改变。
            assert sorted(transformed[index].tolist()) == sorted(boards[index].tolist())


@pytest.mark.parametrize("mode", ["value", "joint", "dual"])
@pytest.mark.parametrize("value_target", ["candidate", "trajectory"])
@pytest.mark.parametrize("nodes", [0, 2])
@pytest.mark.parametrize("training,augment", [(True, True), (True, False), (False, True)])
def test_epoch_augments_all_value_inputs(monkeypatch, mode, value_target, nodes, training, augment) -> None:
    """检查真正送入前向的每一行，防止候选/树内监督再次绕过增强或错配标签。"""
    from r2048.nn import training as trainer

    count = 2

    def boards(rows: int) -> torch.Tensor:
        result = torch.arange(16, dtype=torch.uint8).repeat(rows, 1)
        result[:, 0] = torch.arange(16, 16 + rows, dtype=torch.uint8)
        return result

    tree = _tree_zeros(count, nodes)
    tree["tree_afterstate"] = boards(count * nodes * 4).reshape(count, nodes, 4, 16)
    tree["tree_count"].fill_(nodes)
    tree["tree_legal"].fill_(1)
    tree["tree_visits"].fill_(1)
    tree["tree_value"] = torch.arange(count * nodes * 4).float().reshape(count, nodes, 4)
    tree["tree_rollout_value"] = tree["tree_value"].clone()
    batch = Batch(
        decision=boards(count),
        afterstate=boards(count),
        action=torch.zeros(count, dtype=torch.long),
        legal_mask=torch.ones(count, 4),
        visits=torch.ones(count, 4),
        policy_target=torch.full((count, 4), 0.25),
        value_target=torch.arange(count).float(),
        move_score=torch.zeros(count),
        episode=torch.arange(count),
        candidate_afterstate=boards(count * 4).reshape(count, 4, 16),
        candidate_value=torch.arange(count * 4).float().reshape(count, 4),
        candidate_reward=torch.zeros(count, 4),
        candidate_visits=torch.ones(count, 4),
        candidate_weight=torch.ones(count, 4),
        **tree,
    )
    original = {name: tensor.clone() for name, tensor in batch.__dict__.items()}
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if training else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1) if training else None
    inputs = []

    def forward(model, rows, amp_dtype):
        inputs.append(rows.clone())
        # 保留真实损失与反向路径，只替换网络以直接观察未经编码的前向棋盘。
        values = model(rows.float().mean(dim=1, keepdim=True)).squeeze(1)
        return values, values[:, None].expand(-1, 4)

    class Stream(list):
        def set_epoch(self, epoch):
            pass

    monkeypatch.setattr(trainer, "_forward", forward)
    config = TrainConfig(
        mode=mode, value_target=value_target, augment=augment, value_ranking_weight=1, log_every=0
    )
    loss, _, _ = trainer._run_epoch(
        model, Stream([batch]), optimizer, scheduler, config, torch.device("cpu"), None,
        1024.0, torch.Generator().manual_seed(17), 0, "test",
    )
    assert np.isfinite(loss)
    assert len(inputs) == 1, "所有监督行仍应合并为一次前向"
    value_inputs = inputs[0][count:] if mode in ("joint", "dual") else inputs[0]
    base = batch.candidate_afterstate.reshape(-1, 16) if value_target == "candidate" else batch.afterstate
    expected = torch.cat([base, batch.tree_afterstate.reshape(-1, 16)])
    if training and augment:
        # 每行独立检查 D4 轨道，并保留行下标：仅检查块值集合会漏掉标签槽位错配。
        for before, after in zip(expected, value_inputs, strict=True):
            shaped = before.reshape(4, 4)
            orbit = [
                torch.rot90(shaped, turns).flip(-1) if reflection else torch.rot90(shaped, turns)
                for turns in range(4) for reflection in range(2)
            ]
            assert any(torch.equal(after, transformed.reshape(16)) for transformed in orbit)
        assert not torch.equal(value_inputs[:len(base)], base), "基础价值输入必须实际增强"
        if nodes:
            assert not torch.equal(value_inputs[len(base):], expected[len(base):]), "树内输入必须实际增强"
    else:
        assert torch.equal(value_inputs, expected), "验证集或关闭增强时必须保留原棋盘"
    for name, tensor in original.items():
        assert torch.equal(getattr(batch, name), tensor), f"增强不能修改原批次的 {name}"


def test_training_smoke_writes_checkpoint(batch_dir: Path, tmp_path: Path) -> None:
    """价值模式跑一轮：损失有限、指标改善方向正确、checkpoint 元数据齐全。"""
    run_dir = tmp_path / "run"
    config = training_config(
        batch_dir,
        run_dir,
        mode="value",
        epochs=2,
        batch_size=512,
        lr=1e-3,
        val_fraction=0.2,
        device="cpu",
        seed=3,
        max_samples=1024,
        log_every=0,
    )
    result = run_training(config)
    assert result.metrics, "至少要有一轮记录"
    final = result.metrics[-1]
    assert np.isfinite(final.train_loss), "训练损失必须有限（NaN 会让整轮白跑）"
    assert np.isfinite(final.val_value_mae)
    assert Path(result.best_checkpoint).exists()
    assert Path(result.train_json).exists()

    payload = torch.load(result.best_checkpoint, weights_only=False)
    metadata = payload["metadata"]
    assert metadata["training_target"] == "value"
    assert metadata["data_lineage"]["complete"]
    assert metadata["inherited_dataset_samples"] == 0
    assert metadata["dataset_samples"] == metadata["current_dataset_samples"]
    assert metadata["model_id"].startswith("run-")
    last_meta = torch.load(result.last_checkpoint, weights_only=False)["metadata"]
    assert last_meta["data_lineage"] == metadata["data_lineage"]
    assert metadata["rules_version"] == int(r2048_core.RULES_VERSION)
    assert metadata["batch_id"]
    assert metadata["input_encoding"]
    assert abs(float(metadata["score_scale"]) - 1024.0) < 1e-9
    record = json.loads(Path(result.train_json).read_text())
    assert record["config_digest"] and record["data"]["samples"] == TINY_SAMPLES


def test_policy_training_requires_trained_head_source(batch_dir: Path, tmp_path: Path) -> None:
    """冻结价值训练必须给出源 checkpoint：在随机价值上只训策略头没有意义。"""
    config = training_config(
        batch_dir,
        tmp_path / "nope",
        mode="policy",
        freeze_value=True,
        epochs=1,
        device="cpu",
        log_every=0,
    )
    with pytest.raises(ValueError, match="init-checkpoint"):
        run_training(config)


def test_improved_target_requires_candidates(batch_dir: Path, tmp_path: Path) -> None:
    """要求改进目标却没有候选块时必须报错，而不是安静地退化回访问分布。

    `batch_dir` 是随机策略批次（没有候选监督，也就无法重建 Q）。这类批次上"改进"没有
    任何材料可用，静默退回原口径等于"改了参数却没生效"——那是这个仓库明确拒绝的一类错误。
    """
    value_run = run_training(
        training_config(batch_dir, tmp_path / "value", mode="value", epochs=1, device="cpu", log_every=0)
    )
    config = training_config(
        batch_dir,
        tmp_path / "policy",
        mode="policy",
        freeze_value=True,
        init_checkpoint=value_run.best_checkpoint,
        policy_target="improved",
        epochs=1,
        device="cpu",
        log_every=0,
    )
    with pytest.raises(ValueError, match="候选监督"):
        run_training(config)
    # β ≤ 0 是配置错误（会把目标退化成 argmax 或反过来奖励差的候选），也必须在开跑前失败。
    with pytest.raises(ValueError, match="improve_beta"):
        run_training(
            training_config(
                batch_dir,
                tmp_path / "policy-bad-beta",
                mode="policy",
                freeze_value=True,
                init_checkpoint=value_run.best_checkpoint,
                policy_target="improved",
                improve_beta=0.0,
                epochs=1,
                device="cpu",
                log_every=0,
            )
        )


# --- P3a：多来源的身份、划分与采样 -------------------------------------------------


@pytest.fixture(scope="module")
def sibling_batches(tmp_path_factory) -> tuple[Path, Path]:
    """同 seed、只差 simulations 的两批：这正是"局号是批内的"这件事最容易咬人的形态
    （teacher-s256 与 teacher-s1024 就是这样的一对）。"""
    root = tmp_path_factory.mktemp("siblings")
    dirs = []
    for simulations in (8, 16):
        target = root / f"mcts{simulations}"
        subprocess.run(
            [
                _r2048_binary(),
                "generate",
                "--policy",
                "mcts",
                "--simulations",
                str(simulations),
                "--samples",
                "8000",
                "--seed",
                "424242",
                "--shard-samples",
                "1500",
                "--out",
                str(target),
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        dirs.append(target)
    return dirs[0], dirs[1]


def test_merging_sources_does_not_change_either_split(sibling_batches, tmp_path: Path) -> None:
    """合并多个来源不得改变任何来源自己的划分，且同一局号在两个来源里不再被判成同一局。

    两条都是"来源折进身份"的直接推论：划分是对 `(source_key, episode)` 判定的，所以
    ① 加一个来源不会扰动已有来源的训练/验证归属（否则每加一批数据都得重训才能对比）；
    ② 同一个局号落在哪一侧由来源决定——如果身份里没有来源，两条来源的同一局号必然同侧，
    这个断言就永远找不到反例。
    """
    first, second = sibling_batches
    specs = [(str(first), 1.0, 0), (str(second), 1.0, 0)]
    reader, split = open_sources(specs, val_fraction=0.4, seed=5)
    assert len({source.key for source in reader.sources}) == 2, "同 seed 不同 simulations 必须是两个来源"

    def local_of(source_index: int, indices: np.ndarray) -> np.ndarray:
        source = reader.sources[source_index]
        start, stop = source.offset, source.offset + source.samples
        return indices[(indices >= start) & (indices < stop)] - start

    for index, path in enumerate((first, second)):
        single = make_split(BatchReader(path), val_fraction=0.4, seed=5)
        assert np.array_equal(local_of(index, split.train), single.train), "合并不得改变已有来源的划分"
        assert np.array_equal(local_of(index, split.val), single.val)

    episodes = reader.episode_column()
    per_source = []
    for index, source in enumerate(reader.sources):
        start, stop = source.offset, source.offset + source.samples
        per_source.append(
            (
                set(episodes[start:stop].tolist()),  # 这个来源用到的局号
                set(episodes[local_of(index, split.train)].tolist()),
                set(episodes[local_of(index, split.val)].tolist()),
            )
        )
    shared = per_source[0][0] & per_source[1][0]
    assert shared, "两个来源必须共用同一套局号（否则这条测试没测到它要测的东西）"
    # 判据：存在某个局号，在一个来源里判为验证、在另一个来源里判为训练。
    assert any((number in per_source[0][2]) != (number in per_source[1][2]) for number in shared), (
        "同一局号在两个来源必须能被判到不同一侧"
    )


def test_multi_source_split_has_no_leakage(sibling_batches) -> None:
    """① 不泄漏：以 `(来源, 局号)` 为身份，训练集与验证集不能有同一个身份。

    只用裸局号去查会"通过"——那正是串线：两个来源的第 0 局本来就不是同一局，
    而按身份查出来的交集必须为空。逐来源的局数也要报出来，否则"某个来源整批进了训练集"
    这件事在日志里看不出来。
    """
    first, second = sibling_batches
    # 验证比例取 0.5：这个小批次每个来源只有十几局，0.2 时"某来源一局都没进验证集"的概率
    # 约 1.4%，而来源身份里含 build_id——重建 C++ 就换哈希，这种小概率事件真的会翻出来。
    # 这条测的是"逐来源都非空"的不变式，不该靠运气。
    reader, split = open_sources([(str(first), 1.0, 0), (str(second), 1.0, 0)], val_fraction=0.5, seed=11)
    episodes = reader.episode_column()
    codes = np.repeat(
        np.asarray([source.code for source in reader.sources], dtype=np.uint64),
        [source.samples for source in reader.sources],
    )
    train_identity = {(int(codes[i]), int(episodes[i])) for i in split.train}
    val_identity = {(int(codes[i]), int(episodes[i])) for i in split.val}
    assert train_identity and val_identity
    assert not (train_identity & val_identity), "同一局不能同时出现在训练与验证集里"

    assert len(split.sources) == 2
    for report in split.sources:
        assert report.val_episodes > 0, "每个来源都应当在验证集里有局（否则验证指标只覆盖一部分池子）"
        assert report.train_episodes > 0
    assert sum(report.val_episodes for report in split.sources) == split.val_episodes
    assert sum(report.samples for report in split.sources) == reader.samples


def test_weight_zero_source_leaves_the_curve_identical(
    batch_dir: Path, search_batch_dir: Path, tmp_path: Path
) -> None:
    """② 权重 0 等于没列这个来源：训练曲线必须与只用第一个来源**完全一致**。

    判据用逐张量相等而不是"大致相同"：权重 0 的来源既不进每轮长度、也不消耗随机数，
    所以数据顺序、增强的随机数序列、梯度都一样，最终权重没有理由差一个比特。
    """
    common = {
        "mode": "value",
        "epochs": 2,
        "batch_size": 256,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "device": "cpu",
        "seed": 3,
        "split_seed": 0,
        "max_samples": 512,
        "log_every": 0,
    }

    def run(datasets, name: str):
        config = TrainConfig(datasets=datasets, output=str(tmp_path / name), **common)
        return run_training(config)

    single = run([DatasetSpec(path=str(batch_dir))], "single")
    with_zero = run(
        [DatasetSpec(path=str(batch_dir), weight=1.0), DatasetSpec(path=str(search_batch_dir), weight=0.0)],
        "with-zero",
    )

    left = torch.load(single.last_checkpoint, weights_only=False)["model"]
    right = torch.load(with_zero.last_checkpoint, weights_only=False)["model"]
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), f"{key} 在权重 0 的来源存在时发生了变化"
    assert [m.train_loss for m in single.metrics] == [m.train_loss for m in with_zero.metrics]


# --- P3b：断点续跑 -----------------------------------------------------------------


class _Interrupt(BaseException):
    """在指定位置打断训练（模拟进程被杀，但不涉及信号与子进程）。"""

    def __init__(self, epoch: int, step: int) -> None:
        super().__init__(f"在第 {epoch + 1} 轮第 {step} 步打断")
        self.epoch = epoch
        self.step = step


def _interrupting_progress(epoch: int, done: int):
    """进度回调：走到（第 epoch+1 轮，第 done 批完成）时抛异常。

    回调是在落盘之后被调用的（`_run_epoch` 里先 `on_step` 再报进度），所以打断之后
    `last.pt` 恰好停在这一批的边界上——这就是"被杀"最真实的形状。
    """

    def callback(payload: dict) -> None:
        if int(payload["extra"]["epoch"]) == epoch + 1 and int(payload["done"]) == done:
            raise _Interrupt(epoch, done)

    return callback


def test_resume_matches_one_shot_bitwise(batch_dir: Path, tmp_path: Path) -> None:
    """A 一次跑完 4 轮 = B "跑到第 2 轮中途被杀 → --resume 跑完"，**逐张量相同**。

    这条成立的前提是三样东西都被保存了：模型/优化器/调度器状态、**随机数状态**、
    以及"这一轮已经训到第几批"。少任何一样，续跑都会在某个位置走上另一条轨迹。
    判据取逐张量 `torch.equal`，不接受"接近"。
    """
    common = {
        "mode": "value",
        "epochs": 4,
        "batch_size": 256,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "device": "cpu",
        "seed": 5,
        "split_seed": 0,
        "max_samples": 2048,
        "log_every": 1,
    }

    def line(name: str, **overrides) -> TrainConfig:
        return TrainConfig(
            datasets=[DatasetSpec(path=str(batch_dir))],
            output=str(tmp_path / name),
            **{**common, **overrides},
        )

    once = run_training(line("once"))

    # 第一步：跑到第 2 轮第 3 批就被打断（每步都落盘，所以 checkpoint 正好在那个批边界上）。
    # max_samples 2048 / batch 256 = 每轮 8 步，第 3 批**在轮内**，这样才测到"轮内续跑"。
    with pytest.raises(_Interrupt):
        run_training(line("resumed", save_every_steps=1, on_progress=_interrupting_progress(epoch=1, done=3)))

    state = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False, map_location="cpu")
    assert state["checkpoint_version"] == 2
    assert (state["epoch"], state["step_in_epoch"]) == (1, 3), "checkpoint 应停在打断的那一批边界上"
    assert "rng" in state and "augment" in state["rng"], "随机数状态必须在 checkpoint 里"

    # 第二步：同样的配置续跑（只多了 resume）。
    result = run_training(line("resumed", resume=True))

    left = torch.load(once.last_checkpoint, weights_only=False)["model"]
    right = torch.load(result.last_checkpoint, weights_only=False)["model"]
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), f"{key} 在续跑后与一次跑完不一致"

    # 被打断的那一轮只统计到"续跑的那一截"，其余各轮的指标必须逐项相同。
    assert len(result.metrics) == len(once.metrics) == 4
    for index in (0, 2, 3):
        assert result.metrics[index].train_loss == once.metrics[index].train_loss
        assert result.metrics[index].val_value_mae == once.metrics[index].val_value_mae


@pytest.mark.parametrize("validation", [False, True])
def test_cooperative_stop_preserves_weights_rng_and_lineage(batch_dir, tmp_path, monkeypatch, validation):
    from r2048.nn import training
    from r2048.task_control import TaskStopped

    torch.set_num_threads(1)
    control = tmp_path / "control"
    monkeypatch.setenv("R2048_TASK_CONTROL", str(control))
    common = {
        "mode": "value",
        "architecture": "sh-s6c64",
        "architecture_params": {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2},
        "epochs": 2,
        "batch_size": 128,
        "max_samples": 1024,
        "seed": 61,
        "split_seed": 0,
        "device": "cpu",
        "log_every": 1,
        "val_fraction": 0.25,
        "datasets": [DatasetSpec(str(batch_dir))],
    }
    once = run_training(TrainConfig(output=str(tmp_path / "once"), **common))
    original = training._run_epoch

    def interrupted(*args, **kwargs):
        if validation and kwargs.get("label") == "验证":
            control.write_text("stop")
        return original(*args, **kwargs)

    def progress(payload):
        if not validation and payload["done"] == 1:
            control.write_text("stop")

    monkeypatch.setattr(training, "_run_epoch", interrupted)
    with pytest.raises(TaskStopped):
        run_training(TrainConfig(output=str(tmp_path / "continued"), on_progress=progress, **common))
    last = torch.load(tmp_path / "continued/last.pt", weights_only=False)
    assert "rng" in last and last["metadata"]["data_lineage"]["complete"]
    assert bool(last["pending_epoch"]) == validation
    control.unlink()
    monkeypatch.setattr(training, "_run_epoch", original)
    restored = run_training(TrainConfig(output=str(tmp_path / "continued"), resume=True, **common))
    expected = torch.load(once.last_checkpoint, weights_only=False)
    actual = torch.load(restored.last_checkpoint, weights_only=False)
    assert expected["metadata"]["dataset_samples"] == actual["metadata"]["dataset_samples"]
    for name in expected["model"]:
        assert torch.equal(expected["model"][name], actual["model"][name]), name


def test_automatic_training_seeds_survive_resume(batch_dir: Path, tmp_path: Path) -> None:
    from r2048.fileio import read_json

    torch.set_num_threads(1)
    common = {
        "architecture": "sh-s6c64",
        "architecture_params": {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2},
        "device": "cpu",
        "batch_size": 128,
        "max_samples": 512,
        "val_fraction": 0,
        "seed": -1,
        "split_seed": -1,
        "log_every": 0,
    }
    first = run_training(training_config(batch_dir, tmp_path / "auto", epochs=1, **common))
    seeds = read_json(tmp_path / "auto/seeds.json")
    resumed = run_training(training_config(batch_dir, tmp_path / "auto", epochs=2, resume=True, **common))
    assert read_json(tmp_path / "auto/seeds.json") == seeds
    saved = torch.load(resumed.last_checkpoint, weights_only=False)["config"]
    assert saved["seed"] == seeds["seed"] >= 0
    assert saved["split_seed"] == seeds["split_seed"] >= 0
    assert len(first.metrics) == 1 and len(resumed.metrics) == 2


def test_resume_refuses_changed_data(batch_dir: Path, search_batch_dir: Path, tmp_path: Path) -> None:
    """数据身份变了就必须拒绝续跑：换数据接着训是静默错误（划分与标签都变了，损失照样降）。"""
    common = {
        "mode": "value",
        "epochs": 1,
        "batch_size": 256,
        "lr": 1e-3,
        "device": "cpu",
        "seed": 7,
        "split_seed": 0,
        "max_samples": 512,
        # 两个批次都取 0.5：mcts 那批只有 12 局，0.2 有约 7% 的概率一局都进不了验证集
        # （那会被"验证集为空"的闸门拦下，让这条测试变成在测概率）。
        "val_fraction": 0.5,
        "log_every": 0,
    }

    def line(data: Path, name: str, **overrides) -> TrainConfig:
        return TrainConfig(
            datasets=[DatasetSpec(path=str(data))], output=str(tmp_path / name), **{**common, **overrides}
        )

    run_training(line(batch_dir, "line"))

    # 同一个 --output 里换成另一批数据：必须指名拒绝（说清是哪一项不同）。
    with pytest.raises(ValueError, match="数据身份"):
        run_training(line(search_batch_dir, "line", resume=True))

    # 换数据 + 换目录 = 一条新线：resume 指向空目录时不会去读 last.pt，正常跑。
    assert run_training(line(search_batch_dir, "line-2", resume=True)).metrics


def test_resume_warns_about_config_change(batch_dir: Path, tmp_path: Path, capsys) -> None:
    """配置变了要**说出来**（但仍然续跑）："把轮数调长"是正常操作，安静地改变才是问题。"""
    common = {
        "mode": "value",
        "epochs": 2,
        "batch_size": 256,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "device": "cpu",
        "seed": 9,
        "split_seed": 0,
        "max_samples": 512,
        "log_every": 0,
    }

    def line(**overrides) -> TrainConfig:
        return TrainConfig(
            datasets=[DatasetSpec(path=str(batch_dir))],
            output=str(tmp_path / "line"),
            **{**common, **overrides},
        )

    run_training(line())
    result = run_training(line(resume=True, lr=2e-3))
    printed = capsys.readouterr().out
    assert "配置与上一段不同" in printed and "lr" in printed
    assert result.metrics, "配置变化不是拒绝续跑的理由"


def test_resume_keeps_best_selection(batch_dir: Path, tmp_path: Path) -> None:
    """最佳值跨续跑保留：续跑不能把已经选出的最佳忘掉（否则一个更差的轮会覆盖 model.pt）。

    判据的设定：续跑那一轮用 lr=0，于是它不改变任何参数——模型里没有 BatchNorm/Dropout
    这类会自己漂移的状态（全是 GroupNorm），所以新一轮的验证误差与上一轮**完全相同**，
    不构成"改善"。此时"沿用旧最佳"与"重启挑选"（把 best_metric 清成 None，于是第一轮
    必定算作改善）会给出不同结果：只有前者不会重写最佳 checkpoint。
    """
    common = {
        "mode": "value",
        "epochs": 2,
        "batch_size": 256,
        "lr": 1e-3,
        "val_fraction": 0.2,
        "device": "cpu",
        "seed": 11,
        "split_seed": 0,
        "max_samples": 2048,
        "log_every": 0,
    }

    def line(**overrides) -> TrainConfig:
        return TrainConfig(
            datasets=[DatasetSpec(path=str(batch_dir))],
            output=str(tmp_path / "line"),
            **{**common, **overrides},
        )

    first = run_training(line())
    state = torch.load(tmp_path / "line" / "last.pt", weights_only=False, map_location="cpu")
    assert state["best_metric_kind"] == "val_value_mae" and state["best_metric"] is not None
    # 前提：第一段的最佳**不差于**最后一轮（这样 lr=0 的那一轮就不构成改善）。
    # 之前这里写的是"最佳 = 最后一轮"，它在另一批数据上不成立并直接失败了——那是对的
    # （前提不成立就该失败），但这个更强的前提并不是判据需要的：判据只需要"新旧一轮不更好"。
    assert state["best_metric"] <= state["history"][-1]["val_value_mae"]
    selected_digest = torch.load(first.best_checkpoint, weights_only=False, map_location="cpu")["metadata"][
        "config_digest"
    ]

    second = run_training(line(resume=True, epochs=3, lr=0.0))
    assert second.metrics[-1].epoch == 2
    assert second.metrics[-1].val_value_mae == state["history"][-1]["val_value_mae"], (
        "lr=0 的一轮应当给出与上一轮完全相同的验证指标"
    )
    resumed = torch.load(tmp_path / "line" / "last.pt", weights_only=False, map_location="cpu")
    assert resumed["best_metric"] == state["best_metric"], "旧的最佳值必须被沿用，不能重启挑选"
    assert (
        torch.load(first.best_checkpoint, weights_only=False, map_location="cpu")["metadata"]["config_digest"]
        == selected_digest
    ), "没有改善时不应覆盖最佳 checkpoint（被覆盖说明最佳的选择被重启了）"
