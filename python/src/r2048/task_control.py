"""长任务合作式停止：在可恢复边界退出，不能把取消浏览器请求当成停止计算。"""

import os
from pathlib import Path


class TaskStopped(RuntimeError):
    pass


def stop_requested() -> bool:
    control = os.environ.get("R2048_TASK_CONTROL", "")
    return bool(control and Path(control).exists() and Path(control).read_text().strip() == "stop")


def check_stop() -> None:
    if stop_requested():
        raise TaskStopped("已停止并保存，任务可继续")
