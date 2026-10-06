"""输出根目录的统一约定。

跑谱数据、训练产物、评测结果都归档在同一个输出根下；根目录可以用环境变量
`R2048_OUTPUT_DIR` 指到别的磁盘（默认由 Makefile 指到仓库外的本地文件系统）。
C++ 侧读取同一个环境变量，因此两边不需要互相传路径。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_OUTPUT_DIR = "R2048_OUTPUT_DIR"


def output_root() -> Path:
    return Path(os.environ.get(ENV_OUTPUT_DIR, Path.cwd() / "output"))


def data_dir(batch_id: str) -> Path:
    """跑谱数据批：分片 + manifest + 来源记录。"""
    return output_root() / "data" / batch_id


def run_dir(run_id: str) -> Path:
    """一次训练/评测运行：阶段状态、日志、模型产物。"""
    return output_root() / "runs" / run_id


def model_dir(model_id: str) -> Path:
    return output_root() / "models" / model_id


def eval_dir() -> Path:
    return output_root() / "eval"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
