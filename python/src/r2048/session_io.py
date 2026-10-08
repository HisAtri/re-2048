"""会话协议：一行命令 → 若干行应答，GTP 风格。

形状（与旧版 `.reference/2048ai` 的引擎协议一致，这里只保留真正用到的那部分）：

    > set 2 a1
    = {"state": {...}}

    > analyze simulations=64
    =
    info {"is_during_search": true, "directions": [...]}
    info {"is_during_search": false, "directions": [...]}
    <空行>

* **一行命令，空行结束一次应答**。命令前可以带一个数字 id，应答会原样回显——批量驱动时
  用来把应答对上请求（旧版 GTP 扩展的 `12 genmove` → `12 = left`）。
* **`is_during_search`** 区分"搜索进行中的帧"与终帧：调用方只需要等 `false` 的那一帧。
* **状态变更命令一律回整份状态**，于是界面一次往返就拿到新盘面，不必再问一次。

放在独立进程里、用文本协议，而不是让界面直接调库：引擎会崩、模型要换、而界面不该跟着一起
倒下（旧版三层结构的理由）。文本而非 JSON 帧的好处是它可以被人手敲着调试：
`python -m r2048 session --checkpoint ... ` 之后直接敲 `set 2 a1` 就能用。

**命令词汇与棋谱是同一套**（`clear` / `set <值> <坐标>` / `move <方向>`）：摆棋的命令在棋谱
文件里、在手敲时、在协议里都长这样，而且 `set` 的值域与坐标校验直接复用 `records` 的
解析器——不会出现"能写进棋谱但不能通过协议摆"这种两套规则（ADR 0036）。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import Self

import r2048_core

from r2048 import __version__, paths, records
from r2048.session import Session

PROTOCOL_VERSION = 1
NAME = "r2048-session"

# `_analyze` 自己 emit 应答行（流式），所以 handle 不能再补一行。
_STREAMED = object()


def _response(request_id: str, mark: str, body: str = "") -> str:
    prefix = f"{request_id} " if request_id else ""
    return f"{prefix}{mark} {body}" if body else f"{prefix}{mark}"


def _split_request(line: str) -> tuple[str, list[str]]:
    words = line.split()
    if words and words[0].isdigit():
        return words[0], words[1:]
    return "", words


def _split_response(text: str) -> tuple[str, str]:
    """把一行应答拆成（标记, 其余）。标记只有 `=` / `?` / `info` 三种。"""
    first, _, rest = text.lstrip().partition(" ")
    if first.isdigit():  # 回显的请求 id
        first, _, rest = rest.lstrip().partition(" ")
    return first, rest


class SessionServer:
    """把一行命令变成若干行应答。不碰 IO —— 所以可以脱离管道直接测。"""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._handlers: dict[str, Callable[[str, list[str]], object]] = {
            "protocol_version": self._protocol_version,
            "name": self._name,
            "version": self._version,
            "list_commands": self._list_commands,
            "state": self._state,
            "clear": self._clear,
            "set": self._set,
            "move": self._move,
            "genmove": self._genmove,
            "undo": self._undo,
            "load": self._load,
            "save": self._save,
            "analyze": self._analyze,
            "set_strategy": self._set_strategy,
            "set_simulations": self._set_simulations,
            "set_time_ms": self._set_time_ms,
        }
        self._emit: Callable[[str], None] = lambda _line: None

    def handle(self, line: str, emit: Callable[[str], None]) -> bool:
        """处理一行；用 `emit` 逐行输出。返回 False 表示应当退出。

        **每次应答都以一个空行结束**（GTP 约定）：协议靠它划界，少了它调用方就不知道
        这一条应答读完了没有——两个连续应答也会粘在一起。只有流式命令自己 emit 结尾。
        """
        self._emit = emit
        request_id, words = _split_request(line)
        if not words:  # 裸换行：没有异步分析要打断，忽略即可
            emit(_response(request_id, "="))
            emit("")
            return True
        verb, args = words[0], words[1:]
        if verb == "quit":
            emit(_response(request_id, "="))
            emit("")
            return False
        handler = self._handlers.get(verb)
        try:
            if handler is None:
                raise ValueError(f"未知命令 {verb!r}；用 list_commands 看有哪些")
            payload = handler(request_id, args)
        except (ValueError, RuntimeError, records.RecordError, OSError) as error:
            emit(_response(request_id, "?", str(error)))
            emit("")
            return True
        if payload is not _STREAMED:
            emit(_response(request_id, "=", json.dumps(payload, ensure_ascii=False)))
            emit("")
        return True

    # ---- 基础命令 ---------------------------------------------------------------

    def _protocol_version(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "protocol_version")
        return {"protocol_version": PROTOCOL_VERSION}

    def _name(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "name")
        return {"name": NAME}

    def _version(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "version")
        return {
            "python_package": __version__,
            "rules_version": int(r2048_core.RULES_VERSION),
            "protocol_version": int(r2048_core.PROTOCOL_VERSION),
            "build_id": str(r2048_core.BUILD_ID),
        }

    def _list_commands(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "list_commands")
        # `quit` 在 handle 里特判（它要结束循环），但它确实是一条命令，必须被列出来——
        # 否则"按 list_commands 驱动"的客户端会以为没有正常退出的办法。
        return {"commands": sorted(set(self._handlers) | {"quit"})}

    # ---- 盘面与棋谱 -------------------------------------------------------------

    def _state(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "state")
        return {"state": self.session.state()}

    def _clear(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "clear")
        self.session.reset()
        return {"state": self.session.state()}

    def _set(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 2:
            raise ValueError("set 需要 `<值> <坐标>` 两个参数（值是 null 或 2 的幂）")
        # 值域与坐标都走 records 的解析器：协议与棋谱因此共用同一套规则与同一份键表。
        exponent = None if args[0] == records.EMPTY_TOKEN else _face_to_exponent(args[0])
        self.session.set_tile(records.coordinate_to_index(args[1]), exponent)
        return {"state": self.session.state()}

    def _move(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("move 需要一个方向参数")
        if args[0] not in records.ACTION_OF_DIRECTION:
            raise ValueError(f"未知方向 {args[0]!r}，可选 {'/'.join(records.DIRECTIONS)}")
        step = self.session.play(records.ACTION_OF_DIRECTION[args[0]])
        return {"step": step, "state": self.session.state()}

    def _genmove(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "genmove")
        step = self.session.play(None)  # 树不新鲜时它会自己先分析一次
        report = self.session.last_report
        return {
            "step": step,
            "state": self.session.state(),
            "analysis": report.payload() if report is not None else None,
        }

    def _undo(self, request_id: str, args: list[str]) -> dict:
        self._expect(request_id, args, 0, "undo")
        return {"undone": self.session.undo(), "state": self.session.state()}

    def _load(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("load 需要棋谱路径")
        self.session.load(args[0])
        return {"state": self.session.state()}

    def _save(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("save 需要棋谱路径")
        return {"path": str(self.session.save(args[0])), "state": self.session.state()}

    # ---- 分析 -------------------------------------------------------------------

    def _analyze(self, request_id: str, args: list[str]) -> object:
        self._apply_settings(args)
        prefix = f"{request_id} " if request_id else ""
        self._emit(_response(request_id, "="))  # 起始行：先认领这次请求，再开始出帧

        def frame(report) -> None:
            self._emit(f"{prefix}info {json.dumps(report.payload(), ensure_ascii=False)}")

        frame(self.session.analyze(on_frame=frame))
        self._emit("")  # 空行结束整条应答
        return _STREAMED

    def _set_strategy(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("set_strategy 需要一个策略名")
        self.session.set_strategy(args[0])
        return {"state": self.session.state()}

    def _set_simulations(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("set_simulations 需要一个整数")
        self.session.config.simulations = max(0, int(args[0]))
        return {"state": self.session.state()}

    def _set_time_ms(self, request_id: str, args: list[str]) -> dict:
        if len(args) != 1:
            raise ValueError("set_time_ms 需要一个毫秒数（0 = 不限时）")
        self.session.config.time_ms = max(0, int(args[0]))
        return {"state": self.session.state()}

    def _apply_settings(self, args: list[str]) -> None:
        """`analyze simulations=64 time_ms=500 strategy=search` 这类一次性覆盖。"""
        for token in args:
            key, separator, value = token.partition("=")
            if not separator:
                raise ValueError(f"analyze 的参数应当是 key=value，收到 {token!r}")
            if key == "simulations":
                self.session.config.simulations = max(0, int(value))
            elif key == "time_ms":
                self.session.config.time_ms = max(0, int(value))
            elif key == "strategy":
                self.session.set_strategy(value)
            else:
                raise ValueError(f"analyze 不认识的参数 {key!r}（可选 simulations / time_ms / strategy）")

    @staticmethod
    def _expect(request_id: str, args: list[str], count: int, verb: str) -> None:
        if len(args) != count:
            raise ValueError(f"{verb} 不接受参数" if count == 0 else f"{verb} 需要 {count} 个参数")


def _face_to_exponent(text: str) -> int:
    if text not in records.FACE_TO_EXPONENT:
        raise ValueError(f"{text!r} 不是合法的块值：null 或 2 的 1..{records.MAX_EXPONENT} 次幂")
    return records.FACE_TO_EXPONENT[text]


def serve(session: Session, reader: Iterable[str], writer) -> int:
    """服务循环：逐行读、逐行写。返回退出码（`quit` 或输入结束都算 0）。"""
    server = SessionServer(session)

    def emit(line: str) -> None:
        writer.write(line + "\n")
        writer.flush()

    for raw in reader:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if not server.handle(line, emit):
            return 0
    return 0


# ---- 客户端 ---------------------------------------------------------------------


class SessionError(RuntimeError):
    """会话返回了 `?`、超时，或者进程已经退出。"""


class SessionClient:
    """驱动一个会话子进程。

    读在一个后台线程里、写在这里，于是：(a) `request` 能带超时——界面不该被一个卡住的
    会话冻住；(b) `stream` 可以在搜索还在进行时就把帧交出去，而 Gradio 的生成器回调
    正是这个形状。
    """

    def __init__(self, argv: list[str], env: dict | None = None) -> None:
        child_env = {**os.environ, paths.ENV_OUTPUT_DIR: str(paths.output_root()), **(env or {})}
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
        )
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stderr: list[str] = []
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._next_id = 1

    def _drain_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._queue.put(("line", line))
        self._queue.put(("eof", ""))

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr.append(line.rstrip("\n"))

    def stream(self, command: str, timeout: float = 120.0) -> Iterator[dict]:
        """发一条命令，产出应答里的帧。以空行为界；`?` 会抛 `SessionError`。"""
        request_id = str(self._next_id)
        self._next_id += 1
        if self._process.stdin is None or self._process.poll() is not None:
            raise SessionError("会话进程已经退出：" + self._stderr_tail())
        self._process.stdin.write(f"{request_id} {command}\n")
        self._process.stdin.flush()
        error: str | None = None
        while True:
            try:
                kind, text = self._queue.get(timeout=timeout)
            except queue.Empty as error_timeout:
                raise SessionError(f"会话在 {timeout:.0f} 秒内没有应答：{command!r}") from error_timeout
            if kind == "eof":
                raise SessionError("会话进程提前退出：" + self._stderr_tail())
            if not text.strip():
                break  # 空行 = 这次应答结束
            mark, body = _split_response(text)
            if mark == "?":
                # 记下错误，但**先把这次应答读完**：否则残留的结束空行会留在队列里，
                # 让下一条命令读到上一次的应答——错位之后每一条都对不上，且看不出来。
                error = body.strip()
                continue
            if mark == "info" or (mark == "=" and body.strip()):
                yield json.loads(body)
        if error is not None:
            raise SessionError(error)

    def request(self, command: str, timeout: float = 120.0) -> list[dict]:
        return list(self.stream(command, timeout=timeout))

    def _stderr_tail(self, lines: int = 5) -> str:
        tail = self._stderr[-lines:]
        return " / ".join(tail) if tail else "（stderr 无输出）"

    @property
    def returncode(self) -> int | None:
        """子进程的退出码；还在跑时为 None。"""
        return self._process.poll()

    def close(self) -> None:
        """收尾：先好好说 `quit`，它不走就 terminate，再不走才 kill。"""
        if self._process.poll() is None:
            try:
                self.request("quit", timeout=5.0)
            except (SessionError, OSError, ValueError):
                pass  # 已经读不出应答也没关系：下面按进程状态兜底
            try:
                self._process.wait(timeout=5)  # 给它时间自己退出（否则会白挨一次 terminate）
            except subprocess.TimeoutExpired:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - 只在子进程卡死时走到
                self._process.kill()
                self._process.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error) -> bool:
        self.close()
        return False


def session_argv(
    *,
    checkpoint: str = "",
    device: str = "cpu",
    strategy: str = "search",
    simulations: int = 256,
    time_ms: int = 0,
) -> list[str]:
    """拉起会话子进程的命令行。与 `r2048 session` 子命令的参数一一对应。"""
    argv: list[str] = [
        sys.executable,
        "-m",
        "r2048",
        "session",
        "--device",
        device,
        "--strategy",
        strategy,
        "--simulations",
        str(simulations),
        "--time-ms",
        str(time_ms),
    ]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    return argv
