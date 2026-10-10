"""在当前引擎上测量真实工作负载，保存本机画像（ADR 0051）。"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import tempfile
import time
from functools import wraps
from pathlib import Path

from r2048 import paths
from r2048.identity import file_lock
from r2048.performance import save_measurement


def exclusive_probe(function):
    @wraps(function)
    def run(*args, **kwargs):
        with file_lock(paths.output_root() / "benchmarks/.probe.lock", blocking=False):
            return function(*args, **kwargs)

    return run


def require_idle():
    from r2048.fileio import read_json
    from r2048.webui.tasks import worker_alive

    for status in (paths.output_root() / "jobs").glob("job-*/status.json"):
        record = read_json(status)
        if record["state"] in ("running", "stopping") and worker_alive(status.parent, record):
            raise ValueError(f"任务 {record['task_id']} 正在计算，请待其结束后运行 benchmark")


def _selfplay_worker(checkpoint, settings, directory, seconds, ready, start, results):
    import r2048_core
    import torch

    from r2048.nn.model import load_checkpoint
    from r2048.runtime.inference import BatchedEvaluator
    from r2048.runtime.selfplay import pin_cpu_threads

    pin_cpu_threads()
    model, _ = load_checkpoint(checkpoint, torch.device(settings["device"]))
    evaluator = BatchedEvaluator(model, torch.device(settings["device"]))
    engine = r2048_core.BatchedEngine(
        {
            "simulations": settings["simulations"],
            "lanes": settings["lanes"],
            "in_flight_per_lane": settings["in_flight"],
            "tree_nodes": settings["tree_nodes"],
            "tree_label_plays": settings["tree_label_plays"],
            "tree_label_policy": 1 if settings["tree_label_policy"] == "top_visits" else 0,
            "tree_label_greedy": int(settings["tree_label_rollout"] == "greedy"),
            "seed": 20261010,
            "cache_entries": 131072,
            "temperature": 1.0,
            "temperature_moves": 30,
            "join": 1,
        },
        directory,
        Path(directory).name,
        "benchmark",
        0,
        100000,
    )
    ready.put(True)
    start.wait()
    began = time.monotonic()
    steps = 0
    search_seconds = 0.0
    try:
        while time.monotonic() - began < seconds:
            tick = time.monotonic()
            advance = engine.advance(settings["max_batch"])
            search_seconds += time.monotonic() - tick
            # 样本按整局提交，短基准可能尚未落一局；使用已完成模拟数计量真实搜索吞吐。
            steps = int(engine.stats()["simulations"])
            if advance["request_ids"].size:
                value, policy = evaluator.evaluate(advance["request_kinds"], advance["boards"])
                engine.submit(advance["request_ids"], value, policy)
        elapsed = time.monotonic() - began
        results.put(
            {
                "simulations": steps,
                "seconds": elapsed,
                "gpu_seconds": evaluator.gpu_seconds,
                "search_seconds": search_seconds,
                "forward_rows": evaluator.rows,
                "forward_batches": evaluator.batches,
            }
        )
    finally:
        engine.finish()


def _benchmark_process(*arguments):
    import torch

    try:
        _selfplay_worker(*arguments)
    except (RuntimeError, OSError, ValueError) as error:
        message = {"error": str(error), "oom": isinstance(error, torch.cuda.OutOfMemoryError)}
        # 初始化与计算期都可能 OOM；两个通道分别唤醒等待就绪和等待结果的父进程。
        arguments[-3].put(message)
        arguments[-1].put(message)


@exclusive_probe
def probe_selfplay(checkpoint, settings, candidates, seconds=20):
    if seconds <= 0 or not candidates or any(value <= 0 for value in candidates):
        raise ValueError("测量秒数和 worker 候选必须大于零")
    # 正在运行的正式任务会污染测量；不自动暂停用户任务，也不把争用结果保存成画像。
    require_idle()
    # 与正式 worker 的启动环境一致，避免基准只在导入之后限制 torch 线程。
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    context = mp.get_context("spawn")
    measurements = []
    with tempfile.TemporaryDirectory(prefix="r2048-benchmark-") as temporary:
        for workers in candidates:
            ready, results, start = context.Queue(), context.Queue(), context.Event()
            processes = [
                context.Process(
                    target=_benchmark_process,
                    args=(
                        checkpoint,
                        settings,
                        str(Path(temporary) / f"w{workers}"),
                        seconds,
                        ready,
                        start,
                        results,
                    ),
                )
                for index in range(workers)
            ]
            try:
                for process in processes:
                    process.start()
                count = 0
                while count < workers:
                    try:
                        reply = ready.get(timeout=1)
                        if isinstance(reply, dict):
                            if reply["oom"] and measurements:
                                print("显存不足，停止扩大 worker 数，保留已完成测量。", flush=True)
                                break
                            raise RuntimeError(reply["error"])
                        count += 1
                    except queue.Empty:
                        if any(process.exitcode is not None for process in processes):
                            raise RuntimeError("benchmark worker 初始化失败，未保存画像")
                if count < workers:
                    break
                start.set()
                for process in processes:
                    process.join()
                if any(process.exitcode for process in processes):
                    raise RuntimeError("benchmark worker 执行失败，未保存画像")
                rows = [results.get(timeout=5) for _ in processes]
                errors = [row for row in rows if "error" in row]
                if errors:
                    if all(row["oom"] for row in errors) and measurements:
                        print("显存不足，停止扩大 worker 数，保留已完成测量。", flush=True)
                        break
                    raise RuntimeError(errors[0]["error"])
                elapsed = max(row["seconds"] for row in rows)
                measured = {
                    "workers": workers,
                    "simulations_per_second": sum(row["simulations"] for row in rows) / elapsed,
                    "worker_measurements": rows,
                }
                measurements.append(measured)
                print(f"workers={workers}: {measured['simulations_per_second']:.1f} 模拟/秒", flush=True)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                ready.close()
                results.close()
    fastest = max(row["simulations_per_second"] for row in measurements)
    if fastest <= 0:
        raise ValueError("未生成可计量样本，请增加测量时间；未保存画像")
    # 近似同速选较少 worker，避免为利用率数字浪费资源。
    best = min(
        (row for row in measurements if row["simulations_per_second"] >= fastest * 0.97),
        key=lambda row: row["workers"],
    )
    recommended = {key: settings[key] for key in ("lanes", "in_flight", "max_batch")}
    recommended["workers"] = best["workers"]
    require_idle()
    return save_measurement("selfplay", settings, checkpoint, measurements, recommended)


@exclusive_probe
def probe_generate(settings, candidates, samples=5000):
    from r2048.jobs import run_cpp

    require_idle()
    if samples <= 0 or not candidates or any(value <= 0 for value in candidates):
        raise ValueError("样本目标和 worker 候选必须大于零")
    measured = []
    with tempfile.TemporaryDirectory(prefix="r2048-generate-benchmark-") as temporary:
        for workers in candidates:
            began = time.monotonic()
            run_cpp(
                "generate",
                {
                    "policy": "mcts",
                    "samples": samples,
                    "workers": workers,
                    "simulations": settings["simulations"],
                    "seed": 20261010,
                    "tree_nodes": settings["tree_nodes"],
                    "rollout_policy": settings["rollout_policy"],
                    "out": str(Path(temporary) / str(workers)),
                    "batch_id": f"benchmark-{workers}",
                    "quiet": True,
                },
            )
            measured.append({"workers": workers, "samples_per_second": samples / (time.monotonic() - began)})
    best = max(measured, key=lambda row: row["samples_per_second"])
    require_idle()
    return save_measurement("selfplay", settings, "", measured, {"workers": best["workers"]})


@exclusive_probe
def probe_evaluate(checkpoint, settings, candidates, seconds=20):
    import numpy as np
    import torch

    from r2048.nn.model import load_checkpoint
    from r2048.runtime.inference import BatchedEvaluator
    from r2048.runtime.selfplay import pin_cpu_threads

    require_idle()
    pin_cpu_threads()
    model, _ = load_checkpoint(checkpoint, torch.device(settings["device"]))
    measured = []
    for batch in candidates:
        if batch <= 0 or seconds <= 0:
            raise ValueError("批大小和测量秒数必须大于零")
        evaluator = BatchedEvaluator(model, torch.device(settings["device"]))
        boards = np.random.default_rng(20261010).integers(0, 12, (batch, 16), dtype=np.uint8)
        kinds = np.zeros(batch, dtype=np.uint8)
        # 同形状预热，避免把首轮卷积初始化计入吞吐（ADR 0025）。
        for _ in range(10):
            evaluator.evaluate(kinds, boards)
        began, calls = time.monotonic(), 0
        while time.monotonic() - began < seconds:
            evaluator.evaluate(kinds, boards)
            calls += 1
        measured.append({"max_batch": batch, "rows_per_second": batch * calls / (time.monotonic() - began)})
    best = max(measured, key=lambda row: row["rows_per_second"])
    recommendation = {"max_batch": best["max_batch"]}
    require_idle()
    return save_measurement("evaluate", settings, checkpoint, measured, recommendation)


@exclusive_probe
def probe_train(checkpoint, settings, candidates, data, recipe):
    from r2048.nn.training import DatasetSpec, TrainConfig, run_training

    require_idle()
    measurements = []
    with tempfile.TemporaryDirectory(prefix="r2048-training-benchmark-") as directory:
        for batch in candidates:
            if batch <= 0:
                raise ValueError("训练批大小必须大于零")
            parameters = {
                **recipe,
                **settings,
                "batch_size": batch,
                "epochs": 1,
                "output": str(Path(directory) / str(batch)),
                "init_checkpoint": checkpoint,
                # 划分按整局进行；过短前缀可能只包含一局且全被分到验证集。
                "datasets": [DatasetSpec(path=data, limit=max(5000, batch * 16))],
                "performance": {},
                "log_every": 0,
            }
            result = run_training(TrainConfig(**parameters))
            elapsed = sum(epoch.seconds for epoch in result.metrics)
            measurements.append(
                {
                    "batch_size": batch,
                    "seconds": elapsed,
                    "samples_per_second": sum(epoch.samples for epoch in result.metrics) / elapsed,
                }
            )
    best = max(measurements, key=lambda row: row["samples_per_second"])
    require_idle()
    return save_measurement(
        "train", settings, checkpoint, measurements, {"batch_size": best["batch_size"]}, recipe
    )
