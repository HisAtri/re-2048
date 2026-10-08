"""把两个半模型拼成一个完整模型：价值路径取一边，策略路径取另一边。

为什么需要它：双塔结构下两个头可以分开训练——价值头在候选级目标上训练、策略头在
**冻结价值**上训练（ADR 0010）。两个 run 各自产出一个"完整" checkpoint，但每个都只在
一个头上是新的：要评测"这一轮的搜索强度"，必须把这一轮的**价值路径**与这一轮的
**策略路径**拼在一起，否则会拿旧策略头配新价值头，线间对照失去意义。

这件事以前是 `candidate_iter.sh` 里的一段内联 heredoc：`PYTHONPATH=... python - <<EOF`
加手写的张量前缀过滤。它没有任何"人的判断"成分，只是一条约定（策略路径的参数名以
`policy` 开头），所以搬进仓库、带上校验与来源记录。

顺带把"接错了骨头"变成加载期就报错：两半必须来自同一条线（规则/编码/结构/分数尺度一致），
否则合并出来的模型会在评测时给出一个看着正常、其实语义错位的结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from r2048.fileio import save_torch_atomic

# 策略路径的参数名前缀。与 `nn/model.py` 的命名一致：合并就是把"属于价值路径的"
# 与"属于策略路径的"按名字分开，不做任何形状推断。
POLICY_PREFIX = "policy"
# 两半必须一致的元数据：缺任何一项的比对都可能让两个语义不同的模型拼在一起。
COMPATIBLE_METADATA = ("rules_version", "input_encoding", "architecture")


@dataclass
class MergeResult:
    path: str
    model_id: str
    value_checkpoint: str
    policy_checkpoint: str
    policy_tensors: int
    value_tensors: int


def merge_halves(
    value_checkpoint: str | Path,
    policy_checkpoint: str | Path,
    output: str | Path,
    note: str = "",
) -> MergeResult:
    """`value_checkpoint` 提供非策略张量，`policy_checkpoint` 提供策略张量。"""
    value_path, policy_path = Path(value_checkpoint), Path(policy_checkpoint)
    value_payload = torch.load(value_path, map_location="cpu", weights_only=False)
    policy_payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    value_meta = dict(value_payload.get("metadata", {}))
    policy_meta = dict(policy_payload.get("metadata", {}))

    for field in COMPATIBLE_METADATA:
        before, after = value_meta.get(field), policy_meta.get(field)
        if before != after:
            raise ValueError(
                f"两半模型不兼容：{field} 分别是 {before!r} 与 {after!r}。"
                "合并的前提是它们来自同一条线（同规则、同编码、同结构）。"
            )

    value_model = value_payload["model"]
    policy_model = policy_payload["model"]
    value_names = {name for name in value_model if not name.startswith(POLICY_PREFIX)}
    policy_names = {name for name in policy_model if name.startswith(POLICY_PREFIX)}
    # 名字集合必须互补且完整。漏一个张量不会在这里出错，而会在评测加载时变成
    # "权重没更新"——那种错误不会报警，只会让对照结果安静地失真。
    missing_in_value = {name for name in value_model if name.startswith(POLICY_PREFIX)} - set(policy_model)
    missing_in_policy = {name for name in policy_model if not name.startswith(POLICY_PREFIX)} - set(
        value_model
    )
    if missing_in_value or missing_in_policy:
        raise ValueError(
            f"两半模型的结构不一致：价值侧缺 {sorted(missing_in_value)[:3]}，"
            f"策略侧缺 {sorted(missing_in_policy)[:3]}"
        )
    if not value_names or not policy_names:
        raise ValueError(
            f"没有可合并的张量（价值侧 {len(value_names)} 个、策略侧 {len(policy_names)} 个）："
            f"检查参数名前缀是否为 {POLICY_PREFIX!r}"
        )
    # 两半的非策略形状也要逐一对上：形状不同说明结构不同（上面的 field 比对只能看声明）。
    for name in value_names:
        if value_model[name].shape != policy_model[name].shape:
            raise ValueError(
                f"{name} 在两半里的形状不同：{tuple(value_model[name].shape)} vs "
                f"{tuple(policy_model[name].shape)}"
            )

    merged = {name: value_model[name] for name in value_names}
    merged.update({name: policy_model[name] for name in policy_names})

    value_id = str(value_meta.get("model_id", "")) or value_path.parent.name
    policy_id = str(policy_meta.get("model_id", "")) or policy_path.parent.name
    # 两半各自的训练目标要在改动合并件之前读出来：`metadata` 就是 value_meta 这个对象，
    # 先写 training_target="dual" 会把价值那一半的原始目标冲掉，于是来源记录里会说
    # "价值半是 dual 训出来的"——正是这条记录要回答的问题（首次按新规范实跑时踩到）。
    value_target = str(value_meta.get("training_target", ""))
    policy_target = str(policy_meta.get("training_target", ""))
    metadata = value_meta
    metadata["model_id"] = f"{value_id}+{policy_id}"
    # 合并后的模型**两个头都训过**，所以 `training_target` 是 `dual`（评测侧的闸门用它判断
    # "这份权重能不能拿来跑 nn_policy"，见 workflows/evaluate.py）。沿用价值那一半的 "value"
    # 会让 nn_policy 被自己的闸门拦下——首次跑迭代时撞到的就是这个。
    metadata["training_target"] = "dual"
    metadata["merged_from"] = {
        # 逐半记来源：这是"这份权重是哪两半拼的"唯一凭据（比一个新编的名字有用得多）。
        "value": {
            "path": str(value_path),
            "model_id": value_id,
            "training_target": value_target,
            "config_digest": value_meta.get("config_digest", ""),
        },
        "policy": {
            "path": str(policy_path),
            "model_id": policy_id,
            "training_target": policy_target,
            "config_digest": policy_meta.get("config_digest", ""),
            # 策略头学的是哪个目标（visits / improved）以及它的温度：合并件是评测与跑谱真正
            # 加载的东西，而"这份策略头是纯蒸馏还是被一步改进过"只从 model_id 里看不出来
            # （见 ADR 0032）。既然合并件的 metadata 以价值半为底，就必须显式带过来。
            "policy_target": policy_meta.get("policy_target", "none"),
            "improve_beta": policy_meta.get("improve_beta"),
        },
        "note": note,
    }
    target = Path(output)
    # 合并也不该有半截文件：它是评测直接加载的东西（原子替换与训练 checkpoint 同一口径）。
    save_torch_atomic(
        target,
        {"checkpoint_version": 1, "model": merged, "metadata": json.loads(json.dumps(metadata, default=str))},
    )
    return MergeResult(
        path=str(target),
        model_id=str(metadata["model_id"]),
        value_checkpoint=str(value_path),
        policy_checkpoint=str(policy_path),
        policy_tensors=len(policy_names),
        value_tensors=len(value_names),
    )
