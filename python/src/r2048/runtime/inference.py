"""批推理服务：把 C++ 送来的一批请求合并成一次前向。

要点：一个批次里同时有价值请求（afterstate）与策略请求（决策局面），两者都在同一次
前向里算完，再按请求种类各取所需。这样一次调用覆盖的“局面数”是双头网络的两倍以上，
而且不需要在 C++ 侧区分两套批处理路径。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import r2048_core
import torch

from r2048.nn.encode import encode_boards


class ForwardModel(Protocol):
    def __call__(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class BatchedEvaluator:
    """一次前向覆盖一批请求。

    **没有精度参数**：这里曾经有一个 `dtype` 形参，但它从未参与计算——谁传都会静默拿到
    fp32。按 ADR 0025 的决策先不引入精度开关，所以参数一并删掉，而不是留一个空转的旋钮；
    真要引入时，接线是一处 autocast（约 15 行）并把 dtype 名写进协议里的
    `inference_precision`。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.batches = 0
        self.rows = 0

    def evaluate(self, kinds: np.ndarray, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """kinds: (N,) uint8；boards: (N,16) uint8。

        返回 (values (N,) float32 归一化价值, logits (N,4) float32)。
        value 与 logits 对每个请求都有效——两个头都在同一批上算完了。
        """
        if boards.size == 0:
            return (np.zeros(0, dtype=np.float32), np.zeros((0, r2048_core.NUM_ACTIONS), dtype=np.float32))

        rows = boards.shape[0]
        encoded = encode_boards(torch.from_numpy(np.ascontiguousarray(boards)).to(torch.long).to(self.device))
        with torch.no_grad():
            value, logits = self.model(encoded)
        self.batches += 1
        self.rows += rows
        return (
            value.reshape(-1).to(dtype=torch.float32).cpu().numpy(),
            logits.to(dtype=torch.float32).cpu().numpy(),
        )
