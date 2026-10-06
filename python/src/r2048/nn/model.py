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

ARCHITECTURE = "dual_head_resnet_v1"
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
        self.value_head = nn.Sequential(nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(),
                                        nn.Linear(head_width, 1))
        self.policy_head = nn.Sequential(nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(),
                                         nn.Linear(head_width, 4))

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
            "architecture": ARCHITECTURE,
            "architecture_params": {"channels": self.channels, "blocks": self.blocks},
            "input_encoding": INPUT_ENCODING,
            "score_scale": SCORE_SCALE,
            "model_id": model_id,
            "parent": parent,
        }


def create_model(metadata: dict[str, Any] | None = None) -> DualHeadResNet:
    metadata = metadata or {}
    architecture = metadata.get("architecture", ARCHITECTURE)
    if architecture != ARCHITECTURE:
        raise ValueError(f"不支持的模型结构: {architecture}")
    params = metadata.get("architecture_params", {})
    return DualHeadResNet(channels=int(params.get("channels", 64)), blocks=int(params.get("blocks", 6)))


def save_checkpoint(path: str, model: DualHeadResNet, metadata: dict[str, Any],
                    optimizer: torch.optim.Optimizer | None = None) -> None:
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
        raise ValueError(f"checkpoint 的规则版本 {expected_rules} 与当前内核 "
                         f"{r2048_core.RULES_VERSION} 不一致")
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


def encode_and_forward(model: DualHeadResNet, boards, device: torch.device,
                       policy_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (N,16) 棋盘一次前向，返回 (value, logits)。

    policy_mask 非空时把非法方向的 logits 置为 -inf，避免它们进入 softmax。
    """
    encoded = encode_boards(torch.as_tensor(boards, dtype=torch.long).to(device))
    with torch.no_grad():
        value, logits = model(encoded)
        if policy_mask is not None:
            logits = logits.masked_fill(~policy_mask.to(device), float("-inf"))
    return value, logits
