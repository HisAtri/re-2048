"""原子落盘与 JSON 读写：让"文件存在"等于"文件完整"。

C++ 侧早有 `write_text_atomic`（`cpp/core/fileio.cpp`），Python 侧此前是各处直接
`open(path, "w")`。两种最坏故障都来自非原子写：

* 进程被杀（这正是跑谱暂停、训练中断要面对的常态）留下一个**看起来存在**的半截文件，
  读回时才在随机位置崩掉；
* 状态文件（`status.json`）被外部读到一半，解析失败——而它存在的意义正是"随时可读"。

做法与 C++ 侧一致：写 `<path>.tmp`，成功后 `os.replace` 覆盖。不加 `fsync`：
这里防的是进程级中断，不是断电；C++ 侧同样不加，两边保持一致。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TEMP_SUFFIX = ".tmp"


def write_text_atomic(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + TEMP_SUFFIX)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    except BaseException:
        # 中断留下的 .tmp 一眼可辨，且绝不可能被当成正式产物读走。
        temp.unlink(missing_ok=True)
        raise
    return target


def write_json_atomic(path: str | Path, payload: object) -> Path:
    return write_text_atomic(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
