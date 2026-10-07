"""双头网络：决策态输出策略、afterstate 输出价值。

两个头的输入与语义不同（这是本项目与 KataGo 最本质的差异之一）：
  * 价值头看的是 afterstate u = T(s, a)，即“移动后、随机块生成前”的局面，
    输出 A(u)——从 u 出发的期望新增得分（不含产生 u 的那次合并得分）。
  * 策略头看的是决策局面 s，输出四个方向的 logits，作为搜索的先验。
搜索在展开一个决策节点时同时需要这两者，因此一次前向就能覆盖 1 个决策态 + 4 个
afterstate，这正是跨语言批处理能拿到 5 倍以上批大小的原因（见 ADR 0003）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from r2048.nn.encode import INPUT_CHANNELS, INPUT_ENCODING, encode_boards

# 两种结构都保留：
#   * shared：stem + 6 个全共享残差块 + 两个头（实现最简，作为对照臂）
#   * dual_tower：stem + 3 个共享块 + 每个头各 3 个块（旧版同数据同配方的对照里胜出，
#     配对差 +5,367 分，95% CI [+3,560, +7,186]，到达 4096 的局数 103/400 vs 41/400。
#     代价在 batch 128 的运行点上只有 1.07–1.14×，见 ADR 0012）
SHARED_ARCHITECTURE = "dual_head_resnet_v1"
DUAL_TOWER_ARCHITECTURE = "dual_tower_v1"
ARCHITECTURE = SHARED_ARCHITECTURE
SCORE_SCALE = 1024.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return F.relu(inputs + residual)


class DualHeadResNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6, head_width: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        flat = channels * 4 * 4
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 4)
        )

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (归一化价值, 四方向 logits)。

        两个头在同一批数据上一起算：搜索里同一个批次的请求既有价值请求也有策略请求，
        分开算等于把前向次数翻倍。
        """
        features = self.trunk(self.stem(encoded))
        value = self.value_head(features).squeeze(-1)
        logits = self.policy_head(features)
        return value, logits

    def metadata(self, model_id: str, parent: str = "") -> dict[str, Any]:
        return {
            "architecture": SHARED_ARCHITECTURE,
            "architecture_params": {"channels": self.channels, "blocks": self.blocks},
            "input_encoding": INPUT_ENCODING,
            "score_scale": SCORE_SCALE,
            "model_id": model_id,
            "parent": parent,
        }


class DualTowerResNet(nn.Module):
    """浅层共享、深层分塔的双头网络。

    Value 与 Policy 在 2048 上的输入状态与优化目标都不同（afterstate vs 决策局面、
    回归 vs 分类），只共享浅层棋盘特征既保留参数效率，又避免策略交叉熵直接改写
    已经验收过的价值表示。旧版的对照实验证实这不是锦上添花：见模块开头的数字。
    """

    def __init__(
        self, channels: int = 64, shared_blocks: int = 3, tower_blocks: int = 3, head_width: int = 256
    ) -> None:
        super().__init__()
        self.channels = channels
        self.shared_blocks = shared_blocks
        self.tower_blocks = tower_blocks
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(*(ResidualBlock(channels) for _ in range(shared_blocks)))
        self.value_tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(tower_blocks)))
        self.policy_tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(tower_blocks)))
        flat = channels * 4 * 4
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 4)
        )

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(self.stem(encoded))
        value = self.value_head(self.value_tower(shared)).squeeze(-1)
        logits = self.policy_head(self.policy_tower(shared))
        return value, logits

    def metadata(self, model_id: str, parent: str = "") -> dict[str, Any]:
        return {
            "architecture": DUAL_TOWER_ARCHITECTURE,
            "architecture_params": {
                "channels": self.channels,
                "shared_blocks": self.shared_blocks,
                "tower_blocks": self.tower_blocks,
            },
            "input_encoding": INPUT_ENCODING,
            "score_scale": SCORE_SCALE,
            "model_id": model_id,
            "parent": parent,
        }


def create_model(metadata: dict[str, Any] | None = None) -> nn.Module:
    """按 metadata 里的 architecture 构造模型。

    结构标识写进 checkpoint 并被加载路径校验：换结构就必须换标识，
    否则会拿一套权重去喂另一种拓扑，而 `load_state_dict` 未必报错。
    """
    metadata = metadata or {}
    architecture = metadata.get("architecture", SHARED_ARCHITECTURE)
    params = metadata.get("architecture_params", {})
    if architecture == SHARED_ARCHITECTURE:
        return DualHeadResNet(channels=int(params.get("channels", 64)), blocks=int(params.get("blocks", 6)))
    if architecture == DUAL_TOWER_ARCHITECTURE:
        return DualTowerResNet(
            channels=int(params.get("channels", 64)),
            shared_blocks=int(params.get("shared_blocks", 3)),
            tower_blocks=int(params.get("tower_blocks", 3)),
        )
    raise ValueError(f"不支持的模型结构: {architecture}")


def freeze_value_path(model: nn.Module) -> list[str]:
    """冻结除策略路径之外的全部参数，返回被冻结参数的名单（供日志与产物记录）。

    ADR 0008 的结论：策略交叉熵会改写价值主干，把搜索依赖的叶值尺度一起改坏
    （value MAE 从 352 恶化到 13,406）。因此策略头的默认训练方式是在已验收的
    价值模型上只训练策略路径。
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
    path: str, model: DualHeadResNet, metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None
) -> None:
    payload: dict[str, Any] = {
        "checkpoint_version": 1,
        "model": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[DualHeadResNet, dict[str, Any]]:
    """加载 checkpoint 并校验元数据。

    rules_version / input_encoding / score_scale 任何一项不匹配都必须直接失败：
    用错语义的模型产出的数据比没有数据更糟，因为它看起来是有效的。
    """
    import r2048_core

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint 结构不符合预期: {path}")
    metadata = payload.get("metadata", {})
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

    model = create_model(metadata)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, metadata


def encode_and_forward(
    model: DualHeadResNet, boards, device: torch.device, policy_mask: torch.Tensor | None = None
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
