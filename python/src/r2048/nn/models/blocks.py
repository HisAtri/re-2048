"""网络定义的公共积木。

放这里而不是放进某个版本的定义模块里，是因为多个版本会共用它——而"共用"必须是有意的：
某个版本一旦需要不同形状的块（下采样、不同的归一化），就在自己的模块里定义自己的块，
**不要改这里**。改公共积木会静默改变已经发布过的模型的语义：参数形状可能一点没变
（例如把 GroupNorm 的分组数从 8 改成 16），于是旧权重照样加载成功，只是算出来的
东西不再是当初训练的那个网络。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def check_groups(channels: int, groups: int) -> None:
    """分组数必须整除通道数。

    `nn.GroupNorm` 自己会报错，但它的报错在几十层之外、说的是张量形状；
    宽度是后面一定要改的东西，所以让它在定义构造时就被指名。
    """
    if groups <= 0 or channels % groups:
        raise ValueError(f"groups={groups} 必须整除 channels={channels}（每组的通道数得是整数）")


class ResidualBlock(nn.Module):
    """等宽残差块：卷积 → 归一化 → ReLU → 卷积 → 归一化 → 残差相加。

    刻意**没有** stride / 下采样：4×4 的棋盘上空间维度只有 16 格，逐级下采样会把
    位置信息压没，加深应当靠堆同形状的块（KataGo 的小盘网也是这个形状）。
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        check_groups(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return F.relu(inputs + residual)
