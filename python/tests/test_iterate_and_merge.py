"""迭代编排与半模型合并的测试。

迭代本身（四步跑一遍）很慢，所以这里只测两件不需要真训练的东西：
**合并的语义与校验**、**迭代的步骤标记与解析**。四步的真实链路在跑谱/训练/评测各自的
测试与 ADR 0023 记录的手工验收里覆盖——把一次完整迭代塞进单测只会让测试变成几分钟。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import r2048_core
import torch

from r2048.iterate import (
    DEFAULT_BASELINES,
    STEPS,
    IterateConfig,
    baseline_arms,
    run_iteration,
)
from r2048.workflows.merge import merge_halves


def fake_checkpoint(
    path,
    *,
    model_id: str,
    policy_scale: float = 1.0,
    rules_version: int | None = None,
    training_target: str = "value",
):
    """造一个"半模型"：只需要参数名与元数据，合并逻辑不关心网络结构本身。"""
    model = {
        "stem.weight": torch.full((2, 2), 0.5),
        "value_head.weight": torch.full((3,), 0.25),
        "policy_head.weight": torch.full((4,), policy_scale),
    }
    metadata = {
        "model_id": model_id,
        "rules_version": int(r2048_core.RULES_VERSION) if rules_version is None else rules_version,
        "input_encoding": "v1",
        "architecture": "dual_head_resnet_v1",
        "training_target": training_target,
    }
    torch.save({"checkpoint_version": 1, "model": model, "metadata": metadata}, path)
    return path, model


def test_merge_takes_each_half_by_prefix(tmp_path: Path) -> None:
    """价值侧提供非策略张量、策略侧提供策略张量，两边的取值都不串。"""
    value_path, value_model = fake_checkpoint(tmp_path / "value.pt", model_id="v1")
    policy_path, policy_model = fake_checkpoint(
        tmp_path / "policy.pt", model_id="p1", policy_scale=0.9, training_target="policy"
    )

    result = merge_halves(value_path, policy_path, tmp_path / "merged.pt", note="测试")
    payload = torch.load(result.path, weights_only=False)
    merged = payload["model"]
    assert torch.equal(merged["stem.weight"], value_model["stem.weight"]), "非策略张量必须来自价值侧"
    assert torch.equal(merged["value_head.weight"], value_model["value_head.weight"])
    assert torch.equal(merged["policy_head.weight"], policy_model["policy_head.weight"]), (
        "策略张量必须来自策略侧"
    )
    assert result.policy_tensors == 1 and result.value_tensors == 2

    metadata = payload["metadata"]
    # 两个头都训过：训练模式标成 dual，因为评测侧用它判断"能不能跑 nn_policy"
    # （沿用价值那一半的 "value" 会让 nn_policy 被闸门拦下，实测踩到过）。
    assert metadata["training_target"] == "dual"
    assert metadata["merged_from"]["value"]["model_id"] == "v1"
    assert metadata["merged_from"]["policy"]["model_id"] == "p1"
    # 两半**各自**的训练目标要如实记录：合并会把合并件自己标成 dual，
    # 但"价值半原来是 value、策略半原来是 policy"这条来源信息不能被它冲掉
    # （`metadata` 与 `value_meta` 是同一个对象，写顺序错了就会记成两个 dual）。
    assert metadata["merged_from"]["value"]["training_target"] == "value"
    assert metadata["merged_from"]["policy"]["training_target"] == "policy"
    assert metadata["merged_from"]["note"] == "测试"


def test_merge_refuses_incompatible_halves(tmp_path: Path) -> None:
    """两半必须来自同一条线：规则版本不同、结构形状不同都要拒绝。"""
    value_path, _ = fake_checkpoint(tmp_path / "value.pt", model_id="v1")
    other_rules = r2048_core.RULES_VERSION + 1
    policy_path, _ = fake_checkpoint(tmp_path / "policy.pt", model_id="p1", rules_version=other_rules)
    with pytest.raises(ValueError, match="不兼容"):
        merge_halves(value_path, policy_path, tmp_path / "merged.pt")

    # 形状不同：结构其实不一样，靠元数据看不出来，只能逐张量比。
    broken = {
        "stem.weight": torch.full((3, 3), 0.5),
        "value_head.weight": torch.full((3,), 0.25),
        "policy_head.weight": torch.full((4,), 1.0),
    }
    metadata = {
        "model_id": "p2",
        "rules_version": int(r2048_core.RULES_VERSION),
        "input_encoding": "v1",
        "architecture": "dual_head_resnet_v1",
    }
    torch.save({"checkpoint_version": 1, "model": broken, "metadata": metadata}, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="形状"):
        merge_halves(value_path, tmp_path / "broken.pt", tmp_path / "merged.pt")


def test_merge_halves_cli(tmp_path: Path) -> None:
    """CLI 一层：参数名接线对、stdout 末行是产物路径（ADR 0008 的约定）。

    用合成的两半（只有参数名与元数据）：这里要验的是命令行接线与输出约定，
    真实权重的合并语义由上面两条覆盖。
    """
    value_path, _ = fake_checkpoint(tmp_path / "value.pt", model_id="v1")
    policy_path, _ = fake_checkpoint(tmp_path / "policy.pt", model_id="p1")
    merged = tmp_path / "merged" / "model.pt"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "r2048",
            "merge-halves",
            "--value",
            str(value_path),
            "--policy",
            str(policy_path),
            "--output",
            str(merged),
            "--note",
            "CLI 测试",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert output.stdout.strip().splitlines()[-1] == str(merged), "stdout 末行必须是产物路径"
    assert merged.exists()
    assert torch.load(merged, weights_only=False)["metadata"]["merged_from"]["note"] == "CLI 测试"


def test_iterate_step_markers_are_idempotent(tmp_path: Path) -> None:
    """完成的步骤靠标记文件跳过：重跑不再动数据，且把上一次的产物路径交回来。

    这一条是"迭代可以中断重来"的全部机制，所以它值得一个不依赖训练的测试：
    预置标记文件 → 跑一次 → 立刻返回，产物与标记一致。
    """
    config = IterateConfig(
        name="unit",
        config_path=str(tmp_path / "train.yaml"),
        output=str(tmp_path / "iter"),
        steps=("selfplay",),
        samples=1000,
    )
    recorded = {
        "batch_id": "unit-s8-x",
        "data_dir": str(tmp_path / "data"),
        "manifest": str(tmp_path / "data" / "manifest.json"),
        "teacher": "cpp-mcts8",
        "samples": 1000,
        "episodes": 10,
        "state": "complete",
        "seed": 1,
    }
    (tmp_path / "iter" / "steps").mkdir(parents=True)
    (tmp_path / "iter" / "steps" / "selfplay.json").write_text(json.dumps(recorded))

    result = run_iteration(config)  # 没有数据目录、没有 YAML：跳过步骤就不该碰它们
    assert result.steps["selfplay"] == recorded, "跳过的步骤必须把上一次的产物交回来"
    assert result.seconds < 1.0


def test_iterate_rejects_bad_plans(tmp_path: Path) -> None:
    base = {"name": "x", "config_path": str(tmp_path / "train.yaml"), "output": str(tmp_path / "iter")}
    # 步骤名只能是四个之一（顺序也由代码定，不接受调用方重排）。
    with pytest.raises(ValueError, match="未知步骤"):
        run_iteration(IterateConfig(steps=("train", "finetune"), samples=10, **base))
    # 迭代里的跑谱必须有目标：不限量是给人守着终端的形态，不是流水线的形态。
    with pytest.raises(ValueError, match="--samples"):
        run_iteration(IterateConfig(steps=("selfplay",), samples=0, **base))
    assert STEPS == ("selfplay", "train", "evaluate", "rate")


def test_iterate_baseline_specs(tmp_path: Path) -> None:
    """基线臂的名字就是评分池里的标识：`random` / `mcts<模拟数>`。"""
    config = IterateConfig(
        name="x",
        config_path="y.yaml",
        output=str(tmp_path),
        baselines=("random", "mcts64", "mcts256"),
    )
    assert baseline_arms(config) == [
        ("random", {"policy": "random"}),
        ("mcts64", {"policy": "mcts", "simulations": 64}),
        ("mcts256", {"policy": "mcts", "simulations": 256}),
    ]
    broken = IterateConfig(name="x", config_path="y.yaml", output=str(tmp_path), baselines=("mcts",))
    with pytest.raises(ValueError, match="未知基线"):
        baseline_arms(broken)
    assert DEFAULT_BASELINES == ("random", "mcts64")
