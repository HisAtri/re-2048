"""NN 自对弈驱动：Python 只做“跑前向”，树推进与数据落盘都在 C++。

循环形态（与 ADR 0003 的两阶段协议一致）：
    advance(max_batch) → 一批请求 → 一次前向 → submit 回填 → 继续

两侧的并行性在这里同时成立：C++ 在 advance/submit 里按 lane 并行推进搜索树
（调用期间 GIL 已释放），Python 侧的前向则在 GPU 上把整批请求一次算完。
任何一侧都不会因为另一侧的工作而空转：CPU 搜索与 GPU 推理是流水重叠的。

生命周期（暂停/继续/终止）由 C++ 侧的 Control 通道负责（见 cpp/core/control.h）：
写 `stop`/`pause` 到控制文件，或按一次 Ctrl-C。停车点只有“局与局之间”，
正在进行的局一定走完，已产出的数据一定被提交——所以停下来这件事本身不会丢数据。

**控制文件与信号的两种入口在这里汇合**：纯 C++ 的 `generate` 自己装 SIGINT/SIGTERM
处理器（它没有 Python 运行时），而这条 NN 路径跑在 Python 进程里，Ctrl-C 是 Python 的
KeyboardInterrupt，由 run_selfplay 捕获后调 `request_stop()`——两条路都落到同一个
"局边界停车 + 提交已产出数据"的语义上。SIGTERM/SIGKILL 不在优雅处理范围内，
但它们同样不会破坏已经提交的数据：清单是增量提交日志，未落盘的那一屏在续跑时重算。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import r2048_core
import torch

from r2048.jobs import finish_console, progress_payload
from r2048.runtime.inference import BatchedEvaluator


@dataclass
class SelfplayConfig:
    data_dir: str
    batch_id: str
    # 目标样本条数。0 表示不限量：跑到被暂停/终止为止，随时可以 --continue 接着跑。
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
    # 控制文件路径；空则用 <数据批目录>/control。内容是 pause 或 stop，一次性消费。
    control_path: str = ""
    # 续跑：在既有批次上接着编号写下去（参数逐项比对，不一致直接拒绝）。
    resume: bool = False
    # 结构化进度回调（见 r2048.jobs）。给了回调就不再打印，由调用方决定怎么呈现——
    # CLI 与将来的界面因此只在这一点上分叉。回调收到的 dict 每 0.5 秒一次，
    # 一次自对弈几千次，所以回调本身要廉价（写文件要走原子替换，不要在这里做重活）。
    on_progress: Callable[[dict], None] | None = None


@dataclass
class SelfplayResult:
    manifest_path: str
    batch_id: str
    samples: int
    episodes: int
    seconds: float
    # complete 表示跑满目标；paused/stopped 表示被叫停，可以 --continue 接着跑。
    state: str
    forward_batches: int
    forward_rows: int
    stats: dict = field(default_factory=dict)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0

    @property
    def rows_per_forward(self) -> float:
        return self.forward_rows / self.forward_batches if self.forward_batches > 0 else 0.0


def _progress_line(written: int, total: int | None, rate: float, rows_per_forward: float) -> str:
    head = (
        f"自对弈 {written} 条"
        if total is None
        else f"自对弈 {written}/{total} 条（{100.0 * written / total:.1f}%）"
    )
    return f"{head}，{rate:.0f} 样本/秒，批均值 {rows_per_forward:.0f} 行"


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
        "control_path": config.control_path or str(Path(config.data_dir) / "control"),
        "resume": int(config.resume),
        # 只有 Python 侧知道这两项，但它们会改变逐局结果（批组成决定在途模拟的调度），
        # 所以必须进数据来源记录，也让续跑能发现"换了设备/改了批大小"。
        "provenance": {"device": device.type, "max_batch": str(config.max_batch)},
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
    graceful = False
    result: SelfplayResult | None = None

    def finish() -> SelfplayResult:
        """收尾：写完最后一个分片、落清单、回报终态。异常路径也必须走到这里——
        否则一次中断会把已经产出、只差一个提交动作的数据全丢掉。"""
        finished = engine.finish()
        return SelfplayResult(
            manifest_path=finished["manifest_path"],
            batch_id=config.batch_id,
            samples=int(finished["samples"]),
            episodes=int(finished["episodes"]),
            seconds=time.monotonic() - started,
            state=str(finished["state"]),
            forward_batches=evaluator.batches,
            forward_rows=evaluator.rows,
            stats=dict(engine.stats()),
        )

    try:
        while True:
            # 整个循环体都在这一层里：Ctrl-C 会落在任意一个 Python 帧上，而一次前向
            # （evaluate/submit）也是 Python 代码，所以只包住 advance() 是不够的
            # ——实测中断落在 torch 的卷积里，直接穿到顶层，虽然 finally 保住了数据，
            # 但用户看到的是一段回溯，而不是"已停车"。
            try:
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
                    written = int(advance["samples_written"])
                    rate = written / max(1e-9, now - started)
                    rows_per_forward = evaluator.rows / max(1, evaluator.batches)
                    payload = progress_payload(
                        "selfplay",
                        written,
                        config.samples or None,  # 0 = 不限量：不编一个假的分母
                        "samples",
                        started,
                        line=_progress_line(written, config.samples or None, rate, rows_per_forward),
                        extra={
                            "batch_id": config.batch_id,
                            "forward_batches": evaluator.batches,
                            "forward_rows": evaluator.rows,
                            "episodes": int(advance["episodes_written"]),
                        },
                    )
                    if config.on_progress is None:
                        print("\r" + payload["line"], end="", flush=True)
                    else:
                        config.on_progress(payload)
            except KeyboardInterrupt:
                # 一次 Ctrl-C = 请求停车：进行中的局走完、已产出的数据提交，然后正常退出。
                # 未回填的前向请求不必抢救：引擎会在下一次 advance 里补发它们（见
                # batched_engine.cpp 的 "还没被用掉的需求会被补发"），所以这里继续循环、
                # 让这几局走完，比立刻收尾多留下几局数据。
                # 再按一次才是硬退出（此时 finally 里仍然会把已产出的数据提交掉）。
                if graceful:
                    raise
                graceful = True
                engine.request_stop()
                print(
                    "收到 Ctrl-C：正在停车（进行中的局会走完，已产出的数据会被提交）；再按一次强制退出",
                    file=sys.stderr,
                    flush=True,
                )
                continue

        result = finish()
    finally:
        # 正常路径已经在上面收尾过；异常路径（含第二次 Ctrl-C）在这里兜底提交。
        if result is None:
            result = finish()
        if config.progress_every > 0 and config.on_progress is None:
            finish_console()

    return result
