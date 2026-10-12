"""定义 `smnet-b9c64dt`（共享 3 块 + 两塔各 3 块，共 9 块，64 通道）：每个头各一座塔。

旧版同数据同配方的对照里它以配对差 +5,367 分（95% CI [+3,560, +7,186]）胜出，
到达 4096 的局数 103/400 vs 41/400，代价在 batch 128 上只有 1.07–1.14×（ADR 0012）。
它是"共享编码器 + 任务专属塔"这个方向的第一版：`stem + shared` 是编码器，
`value_tower` / `policy_tower` 是两座塔。

**位置编码、注意力这类"定义内部的实现"就加在这里**（放进编码器或塔内部），
外部只认 `forward(boards) -> (value, logits)`，编码也由本定义负责（ADR 0054）。

两座塔目前只有**深度**可以各自不同（`tower_blocks` 是一个数、宽度共享）。要非对称到
宽度或层内结构时，就在这个模块里加参数——这正是"一版一模块"要保护的自由度；
同时要记得：策略路径必须能被识别出来（今天靠参数名前缀 `policy`，见 ADR 0024 的待办）。
"""

from __future__ import annotations

import torch
from torch import nn

from r2048.nn.components.blocks import ResidualBlock, check_groups
from r2048.nn.components.exponent_one_hot import INPUT_CHANNELS, encode_boards
from r2048.nn.metadata import definition_metadata

# 正式版别名：b9 表示总共 9 个残差块，c64 表示通道数，dt 表示双塔（ADR 0047）。
# **它只是名字**：代码不解析它、也不据此推导结构（结构只在 architecture_params 里，ADR 0026）。
VERSION_ID = "smnet-b9c64dt"
# 同一份定义的历史名字：20 个既有 checkpoint 记着 `dual_tower_v1`，测试版时期用过日期流水号。
# 别名只影响"按名字找定义"，故列出即可；**新产物只写 VERSION_ID**。
ALIASES = ("dt-s3p3v3c64", "dual_tower_v1", "20261006001")


class DualTowerResNet(nn.Module):
    VERSION_ID = VERSION_ID
    ALIASES = ALIASES

    def __init__(
        self,
        channels: int = 64,
        shared_blocks: int = 3,
        tower_blocks: int = 3,
        head_width: int = 256,
        groups: int = 8,
    ) -> None:
        super().__init__()
        check_groups(channels, groups)
        # 每个超参都存成同名属性：metadata 与重建靠它自动完成（见 contract.py）。
        self.channels = channels
        self.shared_blocks = shared_blocks
        self.tower_blocks = tower_blocks
        self.head_width = head_width
        self.groups = groups

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(*(ResidualBlock(channels, groups) for _ in range(shared_blocks)))
        self.value_tower = nn.Sequential(*(ResidualBlock(channels, groups) for _ in range(tower_blocks)))
        self.policy_tower = nn.Sequential(*(ResidualBlock(channels, groups) for _ in range(tower_blocks)))
        flat = channels * 4 * 4
        self.value_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, head_width), nn.ReLU(), nn.Linear(head_width, 4)
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encode_boards(boards)
        # 编码特征的内存布局属于卷积定义内部，外部不必知道特征的维度和通道数。
        if self.stem[0].weight.is_contiguous(memory_format=torch.channels_last):
            encoded = encoded.contiguous(memory_format=torch.channels_last)
        shared = self.shared(self.stem(encoded))
        value = self.value_head(self.value_tower(shared)).squeeze(-1)
        logits = self.policy_head(self.policy_tower(shared))
        return value, logits

    def metadata(self, model_id: str, parent: str = "") -> dict:
        return definition_metadata(self, VERSION_ID, model_id, parent)
