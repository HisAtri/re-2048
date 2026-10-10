"""跨进程、跨语言的日期流水号：只在任务创建时加锁，计算阶段不串行（ADR 0048）。"""

from __future__ import annotations

import fcntl
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from r2048 import paths
from r2048.fileio import write_text_atomic


@contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def allocate_id(kind: str, root: Path | None = None, explicit: str = "") -> str:
    base = root if root is not None else paths.output_root()
    if explicit:
        if not re.fullmatch(r"[0-9]{11}", explicit) or explicit[8:] == "000":
            raise ValueError("身份必须为 YYYYMMDD + 三位非零流水号")
        datetime.strptime(explicit[:8], "%Y%m%d").replace(tzinfo=UTC)
    date = explicit[:8] if explicit else datetime.now(UTC).strftime("%Y%m%d")
    directory = base / ".identities" / kind
    with file_lock(directory / "lock"):
        counter = directory / f"{date}.counter"
        previous = int(counter.read_text()) if counter.exists() else 0
        if kind == "release":
            # 首次启用发号器时接纳已完成的旧发布；清单不是未完成预留的替代品。
            import json

            for manifest in base.glob("*/manifest.json"):
                identity = str(json.loads(manifest.read_text())["release_id"])
                if identity[:8] == date:
                    previous = max(previous, int(identity[8:]))
        number = int(explicit[8:]) if explicit else previous + 1
        if number <= previous:
            raise FileExistsError(f"{kind} 身份 {explicit} 已分配或低于高水位 {previous}")
        if number > 999:
            raise ValueError(f"{kind} 在 {date} 的三位流水号已耗尽")
        write_text_atomic(counter, f"{number}\n")
    return f"{date}{number:03d}"


def batch_id(policy: str, simulations: int) -> str:
    return f"{policy}-s{simulations}-{allocate_id('data')}"
