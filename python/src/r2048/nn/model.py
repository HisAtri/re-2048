"""模型的加载、构造与冻结：按版本 ID 选定义，其余全看 metadata。

结构本身在 `nn/models/`（一版一个模块，见 ADR 0024）；这里只做四件事：

1. **按 ID 选定义并构造**（`create_model`）：ID 是索引，超参从 `architecture_params` 读；
2. **加载闸门**（`load_checkpoint`）：`rules_version` / `input_encoding` / `score_scale` /
   **版本 ID** 任何一项不匹配或缺失都必须直接失败——用错语义的模型产出的数据比没有数据更糟，
   因为它看起来是有效的；
3. **落盘**（`save_checkpoint`，原子替换）；
4. **冻结策略路径之外的一切**（`freeze_value_path`）：ADR 0010 的结论，策略交叉熵会改写
   价值主干，把搜索依赖的叶值尺度一起改坏（value MAE 从 352 恶化到 13,406）。

两个头各自的语义（价值看 afterstate 输出 A(u)、策略看决策局面输出四方向 logits）写在
`nn/models/` 的定义里；搜索在一次前向里同时要这两者，这正是跨语言批处理能拿到 5 倍以上
批大小的原因（见 ADR 0003）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from r2048.fileio import save_torch_atomic
from r2048.nn.encode import INPUT_ENCODING, encode_boards
from r2048.nn.models import (  # noqa: F401 - 转出注册表 API 与常量，保持既有 import 路径可用
    SCORE_SCALE,
    WEIGHTS_DTYPE,
    UnknownModelDefinition,
    build,
    canonical_id,
    default_params,
    known_names,
    resolve,
    supported_ids,
)

# 结构名从定义模块转出：既有代码（training.py、CLI 帮助、测试）从这里 import，
# 所以这里的名字对外是一个契约，不能因为"本文件用不到"就删掉。
from r2048.nn.models.dual_tower import VERSION_ID as DUAL_TOWER_ARCHITECTURE  # noqa: F401
from r2048.nn.models.shared_resnet import VERSION_ID as SHARED_ARCHITECTURE

# 默认定义：没有 metadata（随机初始化、链路自检）时用它。
ARCHITECTURE = SHARED_ARCHITECTURE


def create_model(metadata: dict[str, Any] | None = None) -> nn.Module:
    """按 metadata 里的版本 ID 与超参构造模型。

    没有 ID 时用默认定义（随机初始化的自检路径）。**加载已有权重不要走这条默认**：
    `load_checkpoint` 会要求 ID 必须存在且受支持。
    """
    metadata = metadata or {}
    version_id = str(metadata.get("architecture") or ARCHITECTURE)
    params = dict(metadata.get("architecture_params") or {})
    return build(version_id, params)


def architecture_conflict(version_id: str, params: dict | None, metadata: dict) -> str | None:
    """请求的架构/超参与这份 metadata 是否**真的冲突**（没有则返回 None）。

    用于"从 checkpoint 初始化"这条路径：那时结构由权重决定，请求里写了不同结构就会被静默
    忽略——而"我明明改了宽度"是最难查的一类困惑，所以要拦住。判据是**解析后**的比较：
    两份训练 YAML 里都写着同样的架构与超参是正常用法（同一条线的配置会被反复带上），
    只有真的不同才算冲突；旧 checkpoint 缺 `head_width`/`groups` 这类字段时补默认值再比。
    """
    stored_version = str(metadata.get("architecture") or "")
    # 按 canonical 比较：历史产物记的是旧名字（`dual_tower_v1`），而配置里可能写正式别名
    # （`smnet-b9c64dt`）——同一份定义的两个名字，不是"改了结构"。只有指向不同定义才算冲突。
    if version_id and canonical_id(version_id) != canonical_id(stored_version):
        return (
            f"architecture={version_id}（这份权重用的是 {stored_version}，即 {canonical_id(stored_version)}）"
        )
    if not params:
        return None
    definition = resolve(stored_version)
    defaults = default_params(definition)
    stored = {**defaults, **(metadata.get("architecture_params") or {})}
    requested = {**defaults, **params}
    differing = {
        name: (stored[name], value) for name, value in requested.items() if stored.get(name) != value
    }
    if differing:
        return "architecture_params 与原模型不同：" + "，".join(
            f"{name} 原为 {before}，请求为 {after}" for name, (before, after) in differing.items()
        )
    return None


def freeze_value_path(model: nn.Module) -> list[str]:
    """冻结除策略路径之外的全部参数，返回被冻结参数的名单（供日志与产物记录）。

    策略路径是**按参数名前缀 `policy`** 认出来的：定义必须把策略专属参数放在 `policy*`
    这个名字下（现有的两版都是 `policy_tower` / `policy_head`）。这是当前唯一一处
    "外部要知道定义内部命名"的耦合，代价与替代方案记在 ADR 0024 的待办里——
    换非对称塔或把策略塔嵌进父模块时，第一件要做的事就是把它换成"由定义声明策略路径"。
    """
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("policy"):
            parameter.requires_grad_(True)
            continue
        parameter.requires_grad_(False)
        frozen.append(name)
    if not any(name.startswith("policy") for name, _ in model.named_parameters()):
        raise ValueError("没有找到策略路径参数，冻结逻辑与结构不匹配")
    return frozen


def save_checkpoint(
    path: str, model: nn.Module, metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None
) -> None:
    payload: dict[str, Any] = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    # 原子替换：checkpoint 被写一半时（进程被杀、磁盘满）读回必然崩，
    # 而它往往是续跑唯一依赖的文件（见 fileio.save_torch_atomic）。
    save_torch_atomic(path, payload)


def read_checkpoint_payload(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """训练文件保留 PyTorch 容器；发布文件用 JSON 保存嵌套元数据，避免 pickle 依赖。"""
    if Path(path).suffix == ".safetensor":
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = json.loads((handle.metadata() or {})["metadata"])
        return {"model": load_file(str(path), device=str(device)), "metadata": metadata}
    return torch.load(path, map_location=device, weights_only=False)


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """加载 checkpoint 并校验元数据。

    rules_version / input_encoding / score_scale / 版本 ID 任何一项不匹配都必须直接失败：
    用错语义的模型产出的数据比没有数据更糟，因为它看起来是有效的。版本 ID 缺失也算失败——
    没有 ID 就无法确认该用哪个定义解释这份权重。
    """
    import r2048_core

    payload = read_checkpoint_payload(path, device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint 结构不符合预期: {path}")
    metadata = payload.get("metadata", {})
    version_id = str(metadata.get("architecture", ""))
    if not version_id:
        raise ValueError(f"checkpoint 没有 architecture（版本 ID），无法确认它属于哪个网络定义: {path}")
    resolve(version_id)  # 不支持的 ID 在这里退出（消息里列出本版本支持的定义）

    expected_rules = payload.get("rules_version", metadata.get("rules_version", 1))
    if int(expected_rules) != int(r2048_core.RULES_VERSION):
        raise ValueError(
            f"checkpoint 的规则版本 {expected_rules} 与当前内核 {r2048_core.RULES_VERSION} 不一致"
        )
    if metadata.get("input_encoding", INPUT_ENCODING) != INPUT_ENCODING:
        raise ValueError(f"checkpoint 的输入编码与当前实现不一致: {metadata.get('input_encoding')}")
    scale = float(metadata.get("score_scale", SCORE_SCALE))
    if abs(scale - float(r2048_core.SCORE_SCALE)) > 1e-6:
        raise ValueError(f"checkpoint 的分数尺度 {scale} 与内核 {r2048_core.SCORE_SCALE} 不一致")
    # 权重精度：缺失当 float32（历史文件都没这个键），认得出别的值就拒绝。
    # 不拒的话 `load_state_dict` 会把量化过的权重**静默**转成 float32，于是"这份权重被降过精度"
    # 这件事在评测里看不出任何痕迹（ADR 0026）。
    weights_dtype = str(metadata.get("weights_dtype", WEIGHTS_DTYPE))
    if weights_dtype != WEIGHTS_DTYPE:
        raise ValueError(
            f"checkpoint 的权重精度是 {weights_dtype!r}，本版本只会按 {WEIGHTS_DTYPE} 加载："
            "量化对齐过的权重需要它自己的加载路径（以及一条说明「这份权重与原件不等价」的记录），"
            "直接转精度加载会让评测结论归到原件名下。"
        )

    model = create_model(metadata)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, metadata


def encode_and_forward(
    model: nn.Module, boards, device: torch.device, policy_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (N,16) 棋盘一次前向，返回 (value, logits)。

    policy_mask 非空时把非法方向的 logits 置为 -inf，避免它们进入 softmax。
    """
    encoded = encode_boards(torch.as_tensor(boards, dtype=torch.long).to(device))
    with torch.no_grad():
        value, logits = model(encoded)
        if policy_mask is not None:
            logits = logits.masked_fill(~policy_mask.to(device), float("-inf"))
    return value, logits
