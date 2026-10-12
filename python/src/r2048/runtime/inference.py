"""批推理服务：把 C++ 送来的一批请求合并成一次前向。

要点：一个批次里同时有价值请求（afterstate）与策略请求（决策局面），两者都在同一次
前向里算完，再按请求种类各取所需。这样一次调用覆盖的“局面数”是双头网络的两倍以上，
而且不需要在 C++ 侧区分两套批处理路径。
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import r2048_core
import torch


class ForwardModel(Protocol):
    def __call__(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


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
        # 分相位计时。**GPU 执行时间单独用 CUDA event 量**：Python 侧的 forward 调用是
        # 异步的（只是把核函数排进流），墙钟量到的是"排队"而不是"算"，唯一能说明
        # "GPU 忙了多久"的是 event。这两个数放在一起才能回答"到底谁在等谁"。
        self.seconds = {"transfer": 0.0, "forward": 0.0, "collect": 0.0}
        self.gpu_seconds = 0.0

    def evaluate(self, kinds: np.ndarray, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """kinds: (N,) uint8；boards: (N,16) uint8。

        返回 (values (N,) float32 归一化价值, logits (N,4) float32)。
        value 与 logits 对每个请求都有效——两个头都在同一批上算完了。
        """
        if boards.size == 0:
            return (np.zeros(0, dtype=np.float32), np.zeros((0, r2048_core.NUM_ACTIONS), dtype=np.float32))

        rows = boards.shape[0]
        started = time.monotonic()
        inputs = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
        self.seconds["transfer"] += time.monotonic() - started

        started = time.monotonic()
        # event 只在 CUDA 上能用；CPU 后端就没有"GPU 执行时间"这回事，记 0。
        use_event = self.device.type == "cuda"
        begin = end = None
        if use_event:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        with torch.no_grad():
            value, logits = self.model(inputs)
        if use_event:
            end.record()
        self.seconds["forward"] += time.monotonic() - started

        started = time.monotonic()
        out = (
            value.reshape(-1).to(dtype=torch.float32).cpu().numpy(),
            logits.to(dtype=torch.float32).cpu().numpy(),
        )
        self.seconds["collect"] += time.monotonic() - started
        if use_event:
            # 取回结果已经把流同步了，所以这里的 elapsed_time 是可靠的。
            self.gpu_seconds += begin.elapsed_time(end) / 1e3

        self.batches += 1
        self.rows += rows
        return out
