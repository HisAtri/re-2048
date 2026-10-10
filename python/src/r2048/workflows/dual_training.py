"""双头训练的共同工作流：迭代与 WebUI 使用同一套断点、合并和来源规则。"""

from pathlib import Path
from typing import Any

from r2048.config import TRAIN_SECTION, load_config, train_section_payload
from r2048.fileio import read_json, write_json_atomic
from r2048.identity import allocate_id
from r2048.nn.lineage import file_sha256, resolve_lineage
from r2048.nn.training import TrainConfig, run_training
from r2048.task_control import check_stop
from r2048.workflows.merge import merge_halves


def train_dual(
    directory: Path, name: str, config_path: str, datasets: list, init_checkpoint: str = "", on_progress=None
) -> dict:
    check_stop()
    if init_checkpoint:
        # 默认迭代必须最终可发布；在昂贵训练前确认父模型的累计来源可追溯。
        resolve_lineage(init_checkpoint, strict=True)
    request_path = directory / "requests" / "train.json"
    if request_path.exists():
        attempt = read_json(request_path)["attempt"]
    else:
        attempt = f"run-{allocate_id('run')}"
        write_json_atomic(request_path, {"attempt": attempt})
    run_root = directory / "runs" / attempt
    complete = run_root / "complete.json"
    if complete.exists():
        result = read_json(complete)
        if file_sha256(result["model"]) != result["model_sha256"]:
            raise ValueError("已完成训练的合并产物摘要不一致")
        return result
    file_config = load_config(config_path)
    train_params: dict[str, Any] = train_section_payload(config_path, file_config.get(TRAIN_SECTION, {}))
    train_params["datasets"] = datasets
    if on_progress is not None:
        train_params["on_progress"] = on_progress
    seed = int(train_params.get("seed", 0))
    if "batch_size" not in train_params:
        from r2048.performance import resolve

        effective, performance = resolve(
            "train",
            {**train_params, "mode": "value", "batch_size": None},
            init_checkpoint,
            recipe=train_params,
        )
        train_params.update(batch_size=effective["batch_size"], performance=performance)

    value_run = run_root / "value"
    value_config = TrainConfig(
        output=str(value_run),
        init_checkpoint=init_checkpoint,
        resume=(value_run / "last.pt").exists(),
        mode="value",
        seed=seed,
        **{
            key: value
            for key, value in train_params.items()
            if key not in {"output", "mode", "seed", "init_checkpoint", "freeze_value", "resume"}
        },
    )
    check_stop()
    value = run_training(value_config)
    if not value.best_checkpoint:
        raise ValueError("价值训练未选出最佳 checkpoint：检查验证集是否有可用的有限指标")

    # 策略 run 的起点：有上一轮模型就从它的策略路径继续，否则用**刚训好的价值模型**
    # （首轮没有别的东西可用；它的策略头是随机初始化的，正好就是"要训的那部分"）。
    policy_head_run = run_root / "policy-head"
    policy_config = TrainConfig(
        output=str(policy_head_run),
        init_checkpoint=init_checkpoint or value.best_checkpoint,
        mode="policy",
        freeze_value=True,
        resume=(policy_head_run / "last.pt").exists(),
        # 两个 run 用相邻的种子：它们必须是不同的采样/增强序列，而把两个种子都塞进 CLI
        # 只会让 YAML 里的 seed 失去意义。
        seed=seed + 1,
        **{
            key: value
            for key, value in train_params.items()
            if key not in {"output", "mode", "seed", "freeze_value", "init_checkpoint", "resume"}
        },
    )
    check_stop()
    policy = run_training(policy_config)
    if not policy.best_checkpoint:
        raise ValueError("策略训练未选出最佳 checkpoint：检查验证集策略监督")

    check_stop()
    merged = merge_halves(
        value.best_checkpoint,
        policy.best_checkpoint,
        run_root / "policy" / "model.pt",
        note=f"{name}：价值来自 value run（{seed}），策略来自 policy-head run（{seed + 1}）",
    )
    result = {
        "value_checkpoint": value.best_checkpoint,
        "policy_head_checkpoint": policy.best_checkpoint,
        "model": merged.path,
        "model_id": merged.model_id,
        "seed": seed,
        "data": [spec.path for spec in datasets],
        "model_sha256": file_sha256(merged.path),
    }
    write_json_atomic(complete, result)
    return result
