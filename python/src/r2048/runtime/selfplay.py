"""NN 自对弈驱动：Python 只做“跑前向”，树推进与数据落盘都在 C++。

循环形态（与 ADR 0003 的两阶段协议一致）：
    advance(max_batch) → 一批请求 → 一次前向 → submit 回填 → 继续

两侧的并行性在这里同时成立：C++ 在 advance/submit 里按 lane 并行推进搜索树
（调用期间 GIL 已释放），Python 侧的前向则在 GPU 上把整批请求一次算完。
任何一侧都不会因为另一侧的工作而空转：CPU 搜索与 GPU 推理是流水重叠的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import r2048_core

from r2048.runtime.inference import BatchedEvaluator


@dataclass
class SelfplayConfig:
    data_dir: str
    batch_id: str
    samples: int
    seed: int
    model_id: str = ""
    simulations: int = 64
    lanes: int = 64
    # 每个 lane 在途模拟数：决定一次前向能凑出多大的批次。太小会让流水线退化成
    # “算一个叶子等一次 GPU 往返”，太大则同一条搜索里并行推进的模拟互相看不见
    # 对方的访问数（探索变浅）。它是吞吐与搜索质量之间的旋钮。
    in_flight_per_lane: int = 8
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_moves: int = 30
    cache_entries: int = 1 << 17
    shard_samples: int = 100000
    max_batch: int = 16384
    progress_every: float = 0.5


@dataclass
class SelfplayResult:
    manifest_path: str
    batch_id: str
    samples: int
    episodes: int
    seconds: float
    forward_batches: int
    forward_rows: int
    stats: dict = field(default_factory=dict)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0

    @property
    def rows_per_forward(self) -> float:
        return self.forward_rows / self.forward_batches if self.forward_batches > 0 else 0.0


def run_selfplay(config: SelfplayConfig, model: torch.nn.Module, device: torch.device) -> SelfplayResult:
    engine_config = {
        "lanes": config.lanes,
        "simulations": config.simulations,
        "in_flight_per_lane": config.in_flight_per_lane,
        "c_puct": config.c_puct,
        "temperature": config.temperature,
        "temperature_moves": config.temperature_moves,
        "cache_entries": config.cache_entries,
        "score_scale": float(r2048_core.SCORE_SCALE),
        "seed": config.seed,
    }
    engine = r2048_core.BatchedEngine(
        engine_config,
        str(config.data_dir),
        config.batch_id,
        config.model_id,
        int(config.samples),
        int(config.shard_samples),
    )
    evaluator = BatchedEvaluator(model, device)

    started = time.monotonic()
    last_report = started
    idle_rounds = 0
    while True:
        advance = engine.advance(config.max_batch)
        request_ids = advance["request_ids"]
        if request_ids.size:
            idle_rounds = 0
            values, logits = evaluator.evaluate(advance["request_kinds"], advance["boards"])
            engine.submit(request_ids, values, logits)
        else:
            idle_rounds += 1
            if advance["done"]:
                break
            if idle_rounds > 1000:
                # 既没有请求也没有结束：说明所有 lane 都停了但写入没走到目标，
                # 属于引擎内部的记账错误，直接失败而不是空转到天荒地老。
                raise RuntimeError("自对弈卡住：既无待求值请求，也未达到目标样本数")

        now = time.monotonic()
        if now - last_report >= config.progress_every:
            last_report = now
            written = advance["samples_written"]
            rate = written / max(1e-9, now - started)
            print(
                f"\r自对弈 {written}/{config.samples} 条（{100.0 * written / config.samples:.1f}%），"
                f"{rate:.0f} 样本/秒，批均值 {evaluator.rows / max(1, evaluator.batches):.0f} 行",
                end="",
                flush=True,
            )

    manifest_path = engine.finish()
    seconds = time.monotonic() - started
    if config.progress_every > 0:
        print()

    stats = engine.stats()
    return SelfplayResult(
        manifest_path=manifest_path,
        batch_id=config.batch_id,
        samples=int(stats["samples"]),
        episodes=int(stats["episodes"]),
        seconds=seconds,
        forward_batches=evaluator.batches,
        forward_rows=evaluator.rows,
        stats=dict(stats),
    )
