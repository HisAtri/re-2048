"""版本 `dual_tower_v1`：浅层共享 + 每个头各一座塔。

旧版同数据同配方的对照里它以配对差 +5,367 分（95% CI [+3,560, +7,186]）胜出，
到达 4096 的局数 103/400 vs 41/400，代价在 batch 128 上只有 1.07–1.14×（ADR 0012）。
它是"共享编码器 + 任务专属塔"这个方向的第一版：`stem + shared` 是编码器，
`value_tower` / `policy_tower` 是两座塔。

**位置编码、注意力这类"定义内部的实现"就加在这里**（放进编码器或塔内部），
外部只认 `forward(encoded) -> (value, logits)`，所以传参不变（ADR 0024）。

两座塔目前只有**深度**可以各自不同（`tower_blocks` 是一个数、宽度共享）。要非对称到
宽度或层内结构时，就在这个模块里加参数——这正是"一版一模块"要保护的自由度；
同时要记得：策略路径必须能被识别出来（今天靠参数名前缀 `policy`，见 ADR 0024 的待办）。
"""

from __future__ import annotations

import torch
from torch import nn

from r2048.nn.encode import INPUT_CHANNELS
from r2048.nn.models.blocks import ResidualBlock, check_groups
from r2048.nn.models.contract import definition_metadata

VERSION_ID = "dual_tower_v1"


class DualTowerResNet(nn.Module):
    VERSION_ID = VERSION_ID

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

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(self.stem(encoded))
        value = self.value_head(self.value_tower(shared)).squeeze(-1)
        logits = self.policy_head(self.policy_tower(shared))
        return value, logits

    def metadata(self, model_id: str, parent: str = "") -> dict:
        return definition_metadata(self, VERSION_ID, model_id, parent)
