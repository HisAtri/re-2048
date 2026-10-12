"""现有两版卷积模型显式共用的指数编码，不作为后续模型的默认编码。

调用端只传原始棋盘，各定义自行选择是否复用本工具；保持公式不变，让历史权重继续等价。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

INPUT_CHANNELS = 25


def encode_boards(boards: torch.Tensor) -> torch.Tensor:
    """(N, 16) 指数棋盘 → (N, 25, 4, 4) 浮点特征。

    前 24 个通道是指数 0..23 的 one-hot（指数做截断，常见块级互相独立）；
    第 25 个通道是指数 / 23 的连续值，使大于 23 的指数仍然可区分，
    因此“截断”只是编码选择，不会把 K 当成规则的实际上限。
    """
    values = boards.reshape(-1, 4, 4).to(dtype=torch.long)
    clipped = values.clamp(min=0, max=23)
    one_hot = F.one_hot(clipped, num_classes=24).permute(0, 3, 1, 2).to(dtype=torch.float32)
    scaled = values.to(dtype=torch.float32).unsqueeze(1) / 23.0
    return torch.cat((one_hot, scaled), dim=1)
