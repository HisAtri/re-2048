"""定义 `sh-s6c64`（全共享 6 块共享主干，宽度 c64）：主干 + 两个头。

为什么保留这一版（它已经不是默认的推荐结构，但删不得）：ADR 0012 里它是与双塔对照的
那一臂，历史产物（`runs/c1-*`、`c2-*` 等 20 个 checkpoint）要用它才能重新加载。
**已发布的模型定义不允许删除或改写**——参数形状一变，旧权重就算能加载也已经不是原网络。
"""

from __future__ import annotations

import torch
from torch import nn

from r2048.nn.encode import INPUT_CHANNELS
from r2048.nn.models.blocks import ResidualBlock, check_groups
from r2048.nn.models.contract import definition_metadata

# 正式版别名：形状代号（sh = 全共享主干，s6 = 6 个共享块，c64 = 64 通道）。
# **它只是名字**：代码不解析它、也不据此推导结构（ADR 0026）。
VERSION_ID = "sh-s6c64"
# 历史名字：全部 20 个既有 checkpoint 记的是 `dual_head_resnet_v1`（当时还没有分层命名）。
ALIASES = ("dual_head_resnet_v1",)


class DualHeadResNet(nn.Module):
    VERSION_ID = VERSION_ID
    ALIASES = ALIASES

    def __init__(self, channels: int = 64, blocks: int = 6, head_width: int = 256, groups: int = 8) -> None:
        super().__init__()
        check_groups(channels, groups)
        # 每个超参都存成同名属性：metadata 与重建靠它自动完成（见 models/__init__ 的契约）。
        self.channels = channels
        self.blocks = blocks
        self.head_width = head_width
        self.groups = groups

        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels, groups) for _ in range(blocks)))
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

    def metadata(self, model_id: str, parent: str = "") -> dict:
        return definition_metadata(self, VERSION_ID, model_id, parent)
