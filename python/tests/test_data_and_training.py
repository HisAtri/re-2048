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

from r2048.data.dataset import Batch, BatchStream, make_split, open_sources
from r2048.data.shards import BatchReader
from r2048.nn.augment import apply_boards, augment_policy, transform_policy_targets
from r2048.nn.training import DatasetSpec, TrainConfig, policy_loss, run_training


def training_config(data: str | Path, output: str | Path, **overrides) -> TrainConfig:
    """测试里的 TrainConfig 构造：数据来源写成 DatasetSpec，避免每个断言点都重复一遍。"""
    payload = {"datasets": [DatasetSpec(path=str(data))], "output": str(output)}
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
    assert policy_loss(torch.zeros(128, 4), batch) is None, "无搜索记录时不应产生策略损失"


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
    assert policy_loss(torch.randn(int(has_search.sum()), 4), batch) is not None


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
        candidate_visits=weight,
        candidate_weight=weight,
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
    reader, split = open_sources([(str(first), 1.0, 0), (str(second), 1.0, 0)], val_fraction=0.2, seed=11)
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
