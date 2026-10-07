"""多线程搜索：K 个搜索引擎 + 一个合并推理线程。

**为什么是 K 个引擎，而不是"一个引擎内并行推进 lane"**：一个引擎里所有 lane 共用一份叶子
缓存，每次叶子查找都要拿那把全局锁，而一步搜索只有几微秒——实测每加一倍线程吞吐就掉一截
（197k → 172k → 154k → 137k 模拟/秒，ADR 0027）。K 个引擎各持一份缓存，锁争用从根上消失。

**共享的两样东西**（少了任何一样都不行）：

* **写入侧**（`SampleSink`）：K 个引擎共用同一个批次号、同一个局号计数器、同一份清单，
  所以产出的仍是**一批**数据（用户明确否掉的"多个跑谱进程各建一批"不会发生）。
* **一次前向**：推理线程把各组此刻的请求**合并**成一次前向再分发结果，所以批量不会因为
  分组而缩水（分组让每次前向变小是"分组交错"那条路失败的原因，ADR 0027）。

**重叠来自哪里**：搜索线程调用 `advance()` 时 C++ 会释放 GIL，所以 K 个线程的搜索在不同核上
真并行；推理线程在 GPU 上算这一轮时，各搜索线程的下一轮已经在推进。单引擎做不到这件事——
它的 lane 会一轮把在途预算用满，`advance()` 只能返回空批（ADR 0027 那条边界）。
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from r2048.jobs import finish_console, progress_payload
from r2048.runtime.inference import BatchedEvaluator
from r2048.runtime.selfplay import SelfplayConfig, SelfplayResult, _progress_line, phase_summary


class _Handoff:
    """搜索线程交给推理线程的一批请求。"""

    __slots__ = ("advance", "index")

    def __init__(self, index: int, advance: dict):
        self.index = index
        self.advance = advance


def run_parallel_selfplay(config: SelfplayConfig, model, device: torch.device) -> SelfplayResult:
    threads = max(1, int(config.search_threads))
    if threads == 1:
        raise ValueError("run_parallel_selfplay 需要 search_threads > 1（单线程走顺序驱动）")

    per_engine_lanes = max(1, config.lanes // threads)
    total_lanes = per_engine_lanes * threads
    engine_base = {
        "simulations": config.simulations,
        "in_flight_per_lane": config.in_flight_per_lane,
        "c_puct": config.c_puct,
        "temperature": config.temperature,
        "temperature_moves": config.temperature_moves,
        "cache_entries": config.cache_entries,
        "score_scale": float(__import__("r2048_core").SCORE_SCALE),
        "seed": config.seed,
        "control_path": config.control_path or str(Path(config.data_dir) / "control"),
        "resume": int(config.resume),
        "provenance": {"device": device.type, "max_batch": str(config.max_batch)},
        # 整批的并发数与搜索线程数记进来源记录：它们改变调度与批组成（ADR 0017），
        # 所以既进协议参数、也进数据来源身份（ADR 0021）。
        "lanes_total": total_lanes,
        "search_threads": threads,
    }
    import r2048_core

    engines = [
        r2048_core.BatchedEngine(
            {**engine_base, "lanes": per_engine_lanes},
            str(config.data_dir),
            config.batch_id,
            config.model_id,
            int(config.samples),
            int(config.shard_samples),
        )
    ]
    sink = engines[0].sink()
    for _ in range(threads - 1):
        engines.append(
            r2048_core.BatchedEngine(
                {**engine_base, "lanes": per_engine_lanes},
                str(config.data_dir),
                config.batch_id,
                config.model_id,
                int(config.samples),
                int(config.shard_samples),
                sink,
            )
        )

    evaluator = BatchedEvaluator(model, device)
    handoff: queue.Queue = queue.Queue()
    ready = [threading.Event() for _ in range(threads)]
    for event in ready:
        event.set()  # 初始就有在途预算可用
    abort = threading.Event()
    errors: list[BaseException] = []
    phases = {"advance": 0.0, "evaluate": 0.0, "submit": 0.0, "search_busy": 0.0}
    started = time.monotonic()

    def search_worker(index: int) -> None:
        engine = engines[index]
        idle = 0
        try:
            while not abort.is_set():
                ready[index].wait()
                ready[index].clear()
                if abort.is_set():
                    break
                tick = time.monotonic()
                advance = engine.advance(config.max_batch)
                phases["advance"] += time.monotonic() - tick
                phases["search_busy"] += time.monotonic() - tick
                if advance["request_ids"].size:
                    idle = 0
                    handoff.put(_Handoff(index, advance))
                    continue
                if advance["done"]:
                    break
                # 空批：要么在途预算满了（正常，等 ready），要么引擎真的卡住了。
                idle += 1
                if idle > 1000:
                    raise RuntimeError("自对弈卡住：既无待求值请求，也未达到目标样本数")
                time.sleep(0.001)
        except BaseException as error:  # noqa: BLE001 - 任何异常都要让推理线程知道
            errors.append(error)
            abort.set()
        finally:
            handoff.put(None)  # 哨兵：告诉推理线程这个搜索线程退出了

    def inference_worker() -> None:
        active = threads
        try:
            while active > 0:
                item = handoff.get()
                if item is None:
                    active -= 1
                    continue
                batch = [item]
                while True:  # 把此刻已经就绪的请求一起收上来：合并成一次前向，批量不缩水
                    try:
                        extra = handoff.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        active -= 1
                        continue
                    batch.append(extra)
                kinds = np.concatenate([entry.advance["request_kinds"] for entry in batch])
                boards = np.concatenate([entry.advance["boards"] for entry in batch])
                tick = time.monotonic()
                values, logits = evaluator.evaluate(kinds, boards)
                phases["evaluate"] += time.monotonic() - tick
                tick = time.monotonic()
                offset = 0
                for entry in batch:
                    count = int(entry.advance["request_ids"].size)
                    engines[entry.index].submit(
                        entry.advance["request_ids"],
                        values[offset : offset + count],
                        logits[offset : offset + count],
                    )
                    offset += count
                    ready[entry.index].set()  # 这一组可以继续推进了
                phases["submit"] += time.monotonic() - tick
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
            abort.set()
        finally:
            for event in ready:  # 别让搜索线程永远等在 ready 上
                event.set()

    workers = [threading.Thread(target=search_worker, args=(i,), name=f"search{i}") for i in range(threads)]
    inference = threading.Thread(target=inference_worker, name="inference")
    for worker in workers:
        worker.start()
    inference.start()

    last_report = started
    try:
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.05)
            now = time.monotonic()
            if now - last_report >= config.progress_every:
                last_report = now
                written = int(sink.written_samples())
                rate = written / max(1e-9, now - started)
                payload = progress_payload(
                    "selfplay",
                    written,
                    config.samples or None,
                    "samples",
                    started,
                    line=_progress_line(
                        written, config.samples or None, rate, evaluator.rows / max(1, evaluator.batches)
                    ),
                    extra={
                        "batch_id": config.batch_id,
                        "forward_batches": evaluator.batches,
                        "forward_rows": evaluator.rows,
                        "search_threads": threads,
                        "phases": phase_summary(phases, evaluator, now - started, threads),
                    },
                )
                if config.on_progress is None:
                    print("\r" + payload["line"], end="", flush=True)
                else:
                    config.on_progress(payload)
    except KeyboardInterrupt:
        # 一次 Ctrl-C = 请求停车：进行中的局走完、已产出的数据提交（与顺序驱动同一语义）。
        engines[0].request_stop()  # 广播给共享写入侧：K 个引擎一起在下一个局面边界上停
        print(
            "\n收到 Ctrl-C：正在停车（进行中的局会走完，已产出的数据会被提交）；再按一次强制退出",
            flush=True,
        )
        try:
            while any(worker.is_alive() for worker in workers):
                time.sleep(0.05)
        except KeyboardInterrupt:
            abort.set()
    finally:
        abort.set()
        for event in ready:
            event.set()
        for worker in workers:
            worker.join()
        inference.join()
        if config.on_progress is None:
            finish_console()

    if errors:
        raise errors[0]
    finished = engines[0].finish()
    # 各引擎的计数按引擎求和：它们各自持有叶子缓存与统计，合起来才是整批的量。
    summed = {}
    for engine in engines:
        for key, value in engine.stats().items():
            summed[key] = summed.get(key, 0) + int(value)
    return SelfplayResult(
        manifest_path=finished["manifest_path"],
        batch_id=config.batch_id,
        samples=int(finished["samples"]),
        episodes=int(finished["episodes"]),
        seconds=time.monotonic() - started,
        state=str(finished["state"]),
        forward_batches=evaluator.batches,
        forward_rows=evaluator.rows,
        stats={
            **summed,
            "search_threads": threads,
            "per_engine_lanes": per_engine_lanes,
            "total_lanes": total_lanes,
            "phases": phase_summary(phases, evaluator, time.monotonic() - started, threads),
        },
    )
