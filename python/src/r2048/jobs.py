"""作业目录、结构化进度、C++ 子进程封装：给长任务一个能被外部观察的形状。

为什么需要这一层：跑谱、训练、评测都是几分钟到几小时的任务，而**它们的进度此前
只存在于 stdout 的一行 `\r` 里**。于是任何外部观察者（脚本、将来的界面、跨机器监控）
都只能去解析人类可读的打印文本；C++ 子命令的进度更是只在终端里闪一下，什么也没留下。
三件事被固定在这里：

* **作业目录** `<输出根>/runs/<job_id>/`：
  `job.json`（这次跑的是什么）、`status.json`（现在到哪了，随时可读）、
  `log.jsonl`（发生过什么，只追加）。
* **结构化进度**：所有长任务通过 `on_progress(payload)` 汇报，字段含义固定；
  `payload["line"]` 是给人看的那一行，其余字段是给程序看的。
  没设回调时各阶段仍然打印那一行，因此 CLI 的行为不变——CLI 与界面的分叉只在这一个点。
* **子进程封装**：C++ 子命令在这里统一调用，路径解析与环境注入收在一处
  （此前这套知识只散在 Makefile 与临时脚本里），顺带把每一行输出记进作业日志。

关于 `status.json` 的诚实性：进程被 `kill -9` 时没有任何代码有机会改写它，
所以它保留的是**最后一次已知进度**、状态仍是 `running`。判断"这个作业是不是还活着"
是读者的事（比对 `pid`），这里不假装知道。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from r2048 import paths
from r2048.fileio import read_json, write_json_atomic

# 进度 payload 的字段含义（全部固定，脚本与界面按这些键读）：
#   stage  阶段名（selfplay / train / eval / generate / ...）
#   done   已完成量      total 目标量（None = 不限量，例如"跑到被停下为止"）
#   unit   done/total 的单位（samples / games / steps / epochs）
#   elapsed 已用秒数     rate  单位/秒（按 elapsed 平均）     eta   预计剩余秒数（无法估计则 None）
#   line   给人看的一行（阶段自己格式化，因为它才知道该带哪些细节）
#   extra  阶段特有的读数（批均值、显存、命中率……）
PROGRESS_KEYS = ("stage", "done", "total", "unit", "elapsed", "rate", "eta", "line", "extra")


def progress_payload(
    stage: str,
    done: int,
    total: int | None,
    unit: str,
    started: float,
    line: str,
    extra: dict | None = None,
) -> dict:
    """组装一个进度 payload。速率与 ETA 的算术只在这里算一次。"""
    elapsed = max(0.0, time.monotonic() - started)
    rate = done / elapsed if elapsed > 0 else 0.0
    payload = {
        "stage": stage,
        "done": int(done),
        "total": None if total is None else int(total),
        "unit": unit,
        "elapsed": round(elapsed, 3),
        "rate": round(rate, 3),
        "eta": round((total - done) / rate, 1) if total is not None and rate > 0 else None,
        "line": line,
    }
    if extra:
        payload["extra"] = extra
    return payload


def console_progress(payload: dict) -> None:
    """默认的进度呈现：覆盖式一行。终端里与改动前完全一样。"""
    print("\r" + payload["line"], end="", flush=True)


def finish_console() -> None:
    """覆盖式打印收尾：换行，否则后续输出会接在同一行上。"""
    print()


@dataclass
class CppResult:
    """C++ 子命令的结果。`last_line` 是既有约定（ADR 0008）：产物路径从 stdout 末行取。"""

    command: str
    returncode: int
    lines: list[str] = field(default_factory=list)

    @property
    def last_line(self) -> str:
        return self.lines[-1] if self.lines else ""

    def __str__(self) -> str:  # pragma: no cover - 只为报错可读
        return f"r2048 {self.command} 退出码 {self.returncode}"


def cli_path() -> Path:
    """C++ CLI 的位置：优先 `R2048_CLI`（Makefile 导出），否则从扩展模块反推。

    pyext 与 bin 是 build/ 下的兄弟目录，所以从 `r2048_core` 的位置能反推出 CLI，
    不需要任何人先设环境变量——这条路径解析此前只存在于 Makefile 与临时脚本里。
    """
    override = os.environ.get("R2048_CLI")
    if override:
        return Path(override)
    import r2048_core

    candidate = Path(r2048_core.__file__).resolve().parent.parent / "bin" / "r2048"
    if not candidate.exists():
        raise FileNotFoundError(f"找不到 C++ CLI（{candidate}）：先 make build，或设 R2048_CLI 指向它")
    return candidate


def _argv(command: str, args: dict | list | None) -> list[str]:
    if args is None:
        return []
    if isinstance(args, list):
        return [str(item) for item in args]
    argv: list[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def parse_cpp_progress(line: str) -> dict | None:
    """解析 C++ `--progress-json` 的一行；不是进度行则返回 None。

    单位换算只在这里做一次：C++ 的 Json 没有浮点，所以它报毫秒与整数速率，
    而作业 status.json 的契约是秒（见 PROGRESS_KEYS）。两个单位混在同一字段名里，
    迟早在某个 ETA 上算错——所以交界处就换掉。
    `total == 0` 转成 None：C++ 用 0 表示"不限量"，Python 侧的不限量是 None。
    """
    if not line.startswith('{"progress"'):
        return None
    try:
        payload = json.loads(line)["progress"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    eta = int(payload.get("eta_seconds", -1))
    return {
        "stage": str(payload.get("stage", "")),
        "done": int(payload.get("done", 0)),
        "total": int(payload.get("total", 0)) or None,
        "unit": str(payload.get("unit", "")),
        "elapsed": round(int(payload.get("elapsed_ms", 0)) / 1000.0, 3),
        "rate": float(payload.get("rate", 0)),
        "eta": None if eta < 0 else float(eta),
        "line": str(payload.get("line", "")),
    }


def run_cpp(
    command: str,
    args: dict | list | None = None,
    *,
    job: JobDir | None = None,
    check: bool = True,
    echo: bool = True,
    env: dict | None = None,
    progress: bool = False,
) -> CppResult:
    """跑一个 C++ 子命令，逐行转发输出。

    `echo` 控制是否把输出打到终端；无论是否回显，只要给了 `job` 就都会进作业日志——
    子命令的输出是这次运行唯一的现场记录，它不该只留在终端回滚缓冲里。

    输出根显式传给子进程：C++ 侧读同一个环境变量（`R2048_OUTPUT_DIR`），
    但库调用方（脚本、界面）不该依赖"启动前先 export 过"——一旦两边算出不同的根，
    数据会写到两个地方而没有任何报错。

    **两条流分开读**：stdout 是结果（ADR 0008：末行是产物路径 / 汇总 JSON），
    stderr 是进度与日志。合并成一条会让"末行"这个约定失效——进度行会在结果之后
    继续出现，于是 `last_line` 变成随机取到哪一行。`progress=True` 时给子命令加
    `--progress-json`，并把 stderr 上的进度行解析成结构化 payload 交给作业状态文件。
    """
    argv = [str(cli_path()), command, *_argv(command, args)]
    if progress:
        argv.append("--progress-json")
    child_env = {**os.environ, paths.ENV_OUTPUT_DIR: str(paths.output_root()), **(env or {})}
    if job is not None:
        job.log("exec", {"argv": argv[1:]})
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=child_env,
    )
    lines: list[str] = []
    rendered = []  # 非空 = 已经在终端上覆盖式渲染过进度，收尾需要补一个换行

    def drain_stderr() -> None:
        assert process.stderr is not None
        for raw in process.stderr:
            text = raw.rstrip("\n")
            payload = parse_cpp_progress(text) if progress else None
            if payload is not None:
                if job is not None:
                    job.progress(payload)
                if echo:
                    print("\r" + payload["line"], end="", file=sys.stderr, flush=True)
                    rendered.append(True)
                continue
            if job is not None:
                job.log("stderr", {"line": text})
            if echo:
                print(text, file=sys.stderr, flush=True)

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if job is not None:
                job.log("stdout", {"line": line})
            if echo:
                print(line, flush=True)
    finally:
        returncode = process.wait()
        reader.join()
    if rendered and echo:
        print(file=sys.stderr, flush=True)
    result = CppResult(command=command, returncode=returncode, lines=lines)
    if check and returncode != 0:
        raise RuntimeError(f"{result} —— 见作业日志")
    return result


class JobDir:
    """一次长任务的工作目录。用 `with` 保证结束状态一定被写上（含异常路径）。

    `job_id` 标识**一次尝试**（例如 `selfplay-nn-s64-20261007-120000`），
    而不是一类任务：同一个目录被再次进入时日志是追加的，`job.json` 会被覆盖。

    目录位置有两种意图，所以有两个参数而不是一个含糊的 `root`：

    * `root`：作业目录**们**所在的目录，作业目录 = `<root>/<job_id>`
      （默认是输出根的 `runs/`）。`selfplay` / `eval` 走这条。
    * `path`：直接用这个目录当作业目录。`train` 走这条——它的运行目录本来就是
      `--output`（checkpoint 与 train.json 都写在那里），作业记录应当落在同一处，
      而不是在它下面再套一层。
    """

    def __init__(
        self,
        job_id: str,
        kind: str,
        params: dict | None = None,
        root: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        if root is not None and path is not None:
            raise ValueError("JobDir 只接受 root 或 path 之一：前者是作业目录们的所在目录，后者就是作业目录本身")
        self.job_id = job_id
        self.kind = kind
        self.params = dict(params or {})
        if path is not None:
            self.path = Path(path)
        elif root is not None:
            self.path = Path(root) / job_id
        else:
            self.path = paths.run_dir(job_id)
        self._started = time.monotonic()
        self._log = None
        self._finished = False

    @property
    def job_path(self) -> Path:
        return self.path / "job.json"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def log_path(self) -> Path:
        return self.path / "log.jsonl"

    def __enter__(self) -> Self:
        self.path.mkdir(parents=True, exist_ok=True)
        # 构建身份一起记下来：作业目录是排查问题的第一现场，而"数据/结果对不对"
        # 与产出它们的构建是绑定的。
        import r2048_core

        write_json_atomic(
            self.job_path,
            {
                "job_id": self.job_id,
                "kind": self.kind,
                "params": self.params,
                "pid": os.getpid(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "build_id": str(r2048_core.BUILD_ID),
                "rules_version": int(r2048_core.RULES_VERSION),
                "protocol_version": int(r2048_core.PROTOCOL_VERSION),
            },
        )
        self._log = self.log_path.open("a", encoding="utf-8")
        self._write_status("running", None)
        return self

    def __exit__(self, error_type, error, traceback) -> bool:
        if self._finished:
            return False
        if error is None:
            self.finish()
        else:
            self.fail(f"{error_type.__name__}: {error}" if error_type else "未知错误")
        return False

    def log(self, event: str, payload: dict | None = None) -> None:
        if self._log is None:
            return
        record = {"t": round(time.monotonic() - self._started, 3), "event": event}
        if payload:
            record.update(payload)
        self._log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log.flush()

    def progress(self, payload: dict) -> None:
        """长任务的进度回调：状态文件原子重写 + 日志追加一行。"""
        self._write_status("running", payload)
        self.log(
            "progress",
            {key: payload[key] for key in PROGRESS_KEYS if key in payload and key != "line"},
        )

    def callback(self) -> Callable[[dict], None]:
        """给阶段用的 `on_progress`：写状态与日志，同时在终端打印那一行。"""

        def sink(payload: dict) -> None:
            self.progress(payload)
            console_progress(payload)

        return sink

    def finish(self, **result) -> None:
        if self._finished:
            return
        if result:
            self.log("result", result)
            write_json_atomic(self.path / "result.json", result)
        self.log("finished", {"seconds": round(time.monotonic() - self._started, 3)})
        self._write_status("finished", None)
        self._finished = True
        self._close()

    def fail(self, error: str) -> None:
        if self._finished:
            return
        self.log("failed", {"error": error})
        self._write_status("failed", None, error=error)
        self._finished = True
        self._close()

    def status(self) -> dict:
        """读回状态。作业目录不存在时返回空字典，而不是抛异常：观察者不该因为
        "作业还没开始"就崩掉。"""
        if not self.status_path.exists():
            return {}
        return read_json(self.status_path)

    def _write_status(self, state: str, payload: dict | None, error: str = "") -> None:
        status = {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": state,
            "pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.monotonic() - self._started, 3),
        }
        if payload:
            status["progress"] = payload
        if error:
            status["error"] = error
        write_json_atomic(self.status_path, status)

    def _close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
