"""多进程跑谱的启动器：父进程只做监督，N 个 worker 进程写同一批数据。

为什么要这一层：跑谱的并行路径已经定了是"多进程 join"（可复现，见 ADR 0031），而样本目标
已经改成**按整批判定**（ADR 0029 的订正）——所以进程数随便改，不需要给每个进程分摊配额，
也没有任务分配平衡问题。缺的只是"一条命令起 N 个进程"的便利：手工起 N 条命令行
（`--join` 指向同一个 `--out`）能用，但编排里没法自动做。

父进程**不参与搜索**：它只负责起进程、把批级进度报出来、把 Ctrl-C 变成一次 `stop` 广播、
等所有 worker 收尾、读清单给出汇总。把监督与干活分开，是为了让"跑到哪了"只有一个来源
（清单里的批级统计），而不是每个进程各报一份自己的进度——那正是批级统计替掉的东西。
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from r2048.jobs import console_progress, finish_console, progress_payload
from r2048.runtime.selfplay import SelfplayConfig

# 监督循环的轮询间隔。清单是批级统计的唯一来源，读它比解析各 worker 的输出可靠。
POLL_SECONDS = 2.0


@dataclass
class WorkersResult:
    manifest_path: str
    batch_id: str
    data_dir: str
    samples: int
    episodes: int
    seconds: float
    state: str
    workers: int = 0
    worker_exits: list[int] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0


def _worker_argv(config: SelfplayConfig, checkpoint: str, device: str) -> list[str]:
    """一个 worker 的命令行：`--join` 写进同一批，其余照抄配置。

    `search_threads` 硬写成 1：worker 必须是单线程，否则产出的数据不可复现、也无法重放
    校验（ADR 0031）。并行度由进程数提供。
    """
    argv = [
        sys.executable,
        "-m",
        "r2048",
        "selfplay",
        "--out",
        config.data_dir,
        "--batch-id",
        config.batch_id,
        "--samples",
        str(config.samples),
        "--seed",
        str(config.seed),
        "--simulations",
        str(config.simulations),
        "--lanes",
        str(config.lanes),
        "--in-flight",
        str(config.in_flight_per_lane),
        "--c-puct",
        str(config.c_puct),
        "--temperature",
        str(config.temperature),
        "--temperature-moves",
        str(config.temperature_moves),
        "--tree-nodes",
        str(config.tree_nodes),
        "--tree-label-plays",
        str(config.tree_label_plays),
        "--tree-label-policy",
        config.tree_label_policy,
        "--tree-label-rollout",
        config.tree_label_rollout,
        "--cache-entries",
        str(config.cache_entries),
        "--shard-samples",
        str(config.shard_samples),
        "--max-batch",
        str(config.max_batch),
        "--search-threads",
        "1",
        "--device",
        device,
        "--join",
    ]
    if config.model_id:
        argv += ["--model-id", config.model_id]
    argv += ["--checkpoint", checkpoint] if checkpoint else ["--random-init"]
    return argv


def _tail(path: Path, lines: int = 4) -> str:
    """失败 worker 的日志尾巴：报告失败必须带上现场，否则只能靠复现去猜。"""
    try:
        text = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    except OSError as error:
        return f"(读不到日志：{error})"
    return "\n".join(text[-lines:])


def run_selfplay_workers(
    config: SelfplayConfig,
    checkpoint: str,
    device: str,
    workers: int,
    *,
    log_dir: Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> WorkersResult:
    """起 `workers` 个进程写同一批数据，等它们全部收尾并汇总。

    目标是**整批**的：每个 worker 拿到同一个 `config.samples`，谁先观察到批级总量达标谁就
    让自己那份停下来；实际总量会多出"在途的局"（达标后进行中的局必须走完，否则局号会出现
    空洞）。这与"数据集不需要精确规模"是一致的。
    """
    if workers < 2:
        raise ValueError("run_selfplay_workers 至少需要 2 个 worker（单进程请用 run_selfplay）")
    # 这里不预先校验 argv 的每个字段，而是在收尾后核对**产物**的声明：worker 的参数漏传
    # 过一次（--tree-nodes 忘了加，于是整批数据的 tree_nodes=0 而父进程毫不知情），
    # 表现出来只是"记录短了 216 字节"，不报任何错。这类静默失效必须在收尾处变成显式失败。
    # 覆盖**全部**会改变记录内容的字段，而不只是当时想到的那一个：漏传过两次（--tree-nodes、
    # 以及三个标注参数），每次都是"启用了却没生效"，而产物清单看起来正常。闸门按清单声明逐项
    # 核对，把这类"参数加了但没接上"变成显式失败。
    expected_format = {
        "tree_nodes": int(config.tree_nodes),
        "tree_label_plays": int(config.tree_label_plays),
        "tree_label_policy": 0 if config.tree_label_policy == "depth_stratified" else 1,
        "tree_label_rollout": 1 if config.tree_label_rollout == "greedy" else 0,
    }
    if config.search_threads != 1:
        raise ValueError("worker 必须是单线程（ADR 0031）：多线程产出的数据不可复现")
    if config.join:
        raise ValueError(
            "workers>1 与 --join 互斥：--join 的意思是「加入别人正在写的批次」，"
            "而 workers>1 是「我自己起 N 个进程写一批」，两者只能选一个"
        )

    data_dir = Path(config.data_dir)
    log_dir = Path(log_dir) if log_dir else data_dir.parent / "worker-logs" / config.batch_id
    log_dir.mkdir(parents=True, exist_ok=True)
    # 控制文件：worker 读、父进程写。Ctrl-C 在这里变成一次 `stop` 广播——join 模式下
    # 控制文件是粘住的，一次 stop 停所有进程（ADR 0029）。
    control_path = Path(config.control_path or (data_dir / "control"))

    argv = _worker_argv(config, checkpoint, device)
    started = time.monotonic()
    logs: list[str] = []
    procs: list[subprocess.Popen] = []
    handles: list[object] = []
    for index in range(workers):
        log_path = log_dir / f"worker-{index}.log"
        handle = log_path.open("wb")
        logs.append(str(log_path))
        handles.append(handle)
        procs.append(
            subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        )

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        try:
            control_path.parent.mkdir(parents=True, exist_ok=True)
            control_path.write_text("stop")
            print(
                f"收到中断：已写 {control_path}，等 {workers} 个 worker 走到局边界停车",
                file=sys.stderr,
                flush=True,
            )
        except OSError as error:
            print(f"写控制文件失败（{control_path}）：{error}", file=sys.stderr, flush=True)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    manifest_path = data_dir / "manifest.json"
    last_seen = (-1, -1)
    try:
        while True:
            alive = [proc for proc in procs if proc.poll() is None]
            if not alive:
                break
            samples = episodes = 0
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text())
                    samples = int(data.get("samples", 0))
                    episodes = int(data.get("episodes", 0))
                except (OSError, ValueError):
                    # 清单正被某个 worker 原子重写：读到的半截内容不该让监督失败。
                    pass
            if (samples, episodes) != last_seen:
                last_seen = (samples, episodes)
                total = config.samples if config.samples > 0 else None
                line = (
                    f"多进程自对弈 {samples}/{total if total is not None else '不限量'} 条，"
                    f"{episodes} 局，{len(alive)}/{workers} 个 worker 在跑"
                )
                # 进度 payload 的组装只有 jobs.progress_payload 一处（速率/ETA 的算术、
                # 以及"控制台要打印的那一行"都在它里面）——自己拼 dict 会漏掉 line，
                # 调用方的 console_progress 直接 KeyError（实测踩到）。
                payload = progress_payload(
                    "selfplay-workers",
                    samples,
                    total,
                    "samples",
                    started,
                    line,
                    extra={"episodes": episodes, "workers": workers, "alive": len(alive)},
                )
                if on_progress is not None:
                    on_progress(payload)
                else:
                    console_progress(payload)
            time.sleep(POLL_SECONDS)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # 覆盖式进度收尾：两种路径都要换行（回调路径同样打印了进度行）。
        finish_console()
        for handle in handles:
            handle.close()  # type: ignore[attr-defined]

    exits = [proc.returncode for proc in procs]
    seconds = time.monotonic() - started
    data = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # 产物声明的格式参数必须与请求一致：不一致说明 worker 的参数漏传了（实测踩过：漏了
    # --tree-nodes，整批数据悄悄少一块、训练照跑）。这里失败比事后发现便宜得多。
    if not data:
        raise RuntimeError(
            f"worker 没有产出清单（{manifest_path}）：多半是某个 worker 启动就失败了"
            f"（参数不认识、模型加载失败等）。逐个看日志：{logs}"
        )
    for key, expected in expected_format.items():
        actual = int(data.get(key, -1))
        if actual != expected:
            raise RuntimeError(
                f"跑谱产物的 {key}={actual} 与请求的 {expected} 不一致：worker 的参数没有传全"
                f"（见 _worker_argv）。已产出的数据批不可用，请删掉重来。"
            )
    result = WorkersResult(
        manifest_path=str(manifest_path),
        batch_id=config.batch_id,
        data_dir=str(data_dir),
        samples=int(data.get("samples", 0)),
        episodes=int(data.get("episodes", 0)),
        seconds=seconds,
        state=str(data.get("state", "")),
        workers=workers,
        worker_exits=[int(code) for code in exits],
        logs=logs,
    )
    failed = [index for index, code in enumerate(exits) if code != 0]
    if failed:
        tails = "\n".join(
            f"  worker-{index}（日志 {logs[index]}）：\n{_tail(Path(logs[index]))}" for index in failed
        )
        raise RuntimeError(f"{len(failed)}/{workers} 个 worker 退出码非 0：\n{tails}")
    return result
