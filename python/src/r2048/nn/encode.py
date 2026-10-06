"""棋盘编码：跨语言线上只传原始棋盘，编码只存在于这一处。

为什么不把编码好的特征跨语言传递：编码方式是最容易在两侧悄悄分叉的东西，
一旦 C++ 与 Python 对同一局面的特征理解不同，训练与自对弈就会用两套语义，
而且错得很隐蔽。让 C++ 只送 16 个指数、由这里唯一地编码，是从结构上排除这类分叉。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

BOARD_CELLS = 16
INPUT_CHANNELS = 25
# 写入 checkpoint metadata 的编码标识：推理侧据此拒绝与当前编码不匹配的模型。
INPUT_ENCODING = "24 clipped exponent one-hot + exponent/23"


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


def encode_numpy(boards: np.ndarray, device: torch.device) -> torch.Tensor:
    """从 numpy (N, 16) uint8 直接编码，避免中间的 Python 循环。"""
    tensor = torch.from_numpy(np.ascontiguousarray(boards)).to(device=device, dtype=torch.uint8)
    return encode_boards(tensor.to(dtype=torch.long))
