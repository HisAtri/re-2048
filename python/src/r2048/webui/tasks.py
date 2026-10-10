"""服务器级持久队列：浏览器只提交/观察任务，计算在独立子进程运行（ADR 0049）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from r2048 import paths
from r2048.fileio import read_json, write_json_atomic, write_text_atomic
from r2048.identity import allocate_id, file_lock

TERMINAL = {"finished", "failed", "stopped", "cancelled", "interrupted"}


def worker_alive(directory: Path, status: dict) -> bool:
    """PID 可能复用，必须同时匹配进程创建时间、任务目录和随机提交 token。"""
    pid = status.get("pid")
    if not pid:
        return False
    try:
        process = Path(f"/proc/{pid}")
        started = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        argv = process.joinpath("cmdline").read_bytes().decode().split("\0")
        token = read_json(directory / "request.json")["submission_key"]
        return (
            started == status.get("process_start")
            and "r2048.webui.worker" in argv
            and str(directory) in argv
            and token in argv
        )
    except (OSError, KeyError):
        return False


class TaskManager:
    def __init__(self, root: Path | None = None, acquire=None, release=None):
        self.root = (root or paths.output_root()).resolve()
        self.directory = self.root / "jobs"
        self.acquire = acquire or (lambda: None)
        self.release = release or (lambda: None)
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closing = False
        self.thread = None
        self.guard = None

    def start(self):
        with self.lock:
            if self.thread is not None:
                return
            self.guard = file_lock(self.directory / ".manager.lock", blocking=False)
            self.guard.__enter__()
            for task in self.list():
                if task["state"] in ("running", "stopping") and not worker_alive(
                    Path(task["directory"]), task
                ):
                    self._status(Path(task["directory"]), "interrupted", error="上次执行中断，可继续任务")
            self.thread = threading.Thread(target=self._dispatch, name="experiment-queue", daemon=True)
            self.thread.start()

    def submit(self, request: dict, submission_key: str) -> str:
        from r2048.webui.worker import validate_request

        validate_request(request, self.root)
        self.start()
        with self.lock:
            for task in self.list():
                if read_json(Path(task["directory"]) / "request.json")["submission_key"] == submission_key:
                    return task["task_id"]
            identity = f"job-{allocate_id('job', self.root)}"
            directory = self.directory / identity
            directory.mkdir(parents=True)
            snapshot = json.loads(json.dumps(request))
            write_json_atomic(
                directory / "request.json",
                {
                    "version": 1,
                    "task_id": identity,
                    "submission_key": submission_key,
                    **snapshot,
                },
            )
            self._status(directory, "queued", created_at=time.time())
            self.wake.set()
            return identity

    def _status(self, directory: Path, state: str, **extra):
        with file_lock(directory / ".status.lock"):
            previous = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
            if previous.get("state") in TERMINAL and state in ("running", "stopping"):
                return
            write_json_atomic(
                directory / "status.json",
                {
                    **previous,
                    "task_id": directory.name,
                    "state": state,
                    "updated_at": time.time(),
                    **extra,
                },
            )

    def list(self) -> list[dict]:
        if not self.directory.exists():
            return []
        tasks = []
        for directory in self.directory.glob("job-*"):
            if (directory / "status.json").exists():
                tasks.append({**read_json(directory / "status.json"), "directory": str(directory)})
        return sorted(tasks, key=lambda item: item.get("created_at", 0))

    def detail(self, identity: str) -> dict:
        directory = self._path(identity)
        status = read_json(directory / "status.json")
        for field, file in (("progress", "work/status.json"), ("result", "result.json")):
            if (directory / file).exists():
                record = read_json(directory / file)
                status[field] = record.get("progress", record) if field == "progress" else record
        status["request"] = read_json(directory / "request.json")
        published = directory / "work/steps/publish.json"
        if "result" not in status and published.exists():
            status["result"] = read_json(published)
        manifest = status.get("result", {}).get("manifest")
        model = status.get("result", {}).get("model")
        if not model:
            model = status["request"].get("config", {}).get("checkpoint", "")
            manifest = (
                str(Path(model).parent / "manifest.json") if Path(model).suffix == ".safetensor" else ""
            )
        if model and manifest and Path(manifest).exists():
            identity = read_json(manifest)
            status["model_identity"] = {
                key: identity.get(key)
                for key in (
                    "model_id",
                    "architecture",
                    "current_dataset_samples",
                    "inherited_dataset_samples",
                    "dataset_samples",
                    "dataset_ids",
                    "parent_models",
                    "data_lineage",
                )
            }
        status["directory"] = str(directory)
        log = directory / "worker.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                status["log"] = "\n".join(
                    stream.read().decode(errors="replace").replace("\r", "\n").splitlines()[-100:]
                )
        return status

    def _path(self, identity: str) -> Path:
        if not identity:
            raise ValueError("请先选择任务")
        directory = self.directory / identity
        if directory.parent != self.directory or not (directory / "request.json").exists():
            raise ValueError("不认识的任务身份")
        return directory

    def stop(self, identity: str):
        with self.lock:
            directory = self._path(identity)
            state = read_json(directory / "status.json")["state"]
            if state == "queued":
                self._status(directory, "cancelled")
            elif state in ("running", "stopping"):
                self._status(directory, "stopping")
                write_text_atomic(directory / "control", "stop")
            self.wake.set()

    def resume(self, identity: str):
        self.start()
        with self.lock:
            directory = self._path(identity)
            state = read_json(directory / "status.json")["state"]
            if state not in ("stopped", "failed", "interrupted"):
                raise ValueError("仅已停止、失败或中断任务可以继续")
            if worker_alive(directory, read_json(directory / "status.json")):
                raise ValueError("原工作进程仍在运行，不能重复启动")
            (directory / "control").unlink(missing_ok=True)
            self._status(directory, "queued", error="", created_at=time.time())
            self.wake.set()

    def _dispatch(self):
        try:
            while not self.closing:
                tasks = self.list()
                active = next((item for item in tasks if item["state"] in ("running", "stopping")), None)
                queued = next((item for item in tasks if item["state"] == "queued"), None)
                item = active or queued
                if item is None:
                    self.wake.wait(1)
                    self.wake.clear()
                    continue
                directory = Path(item["directory"])
                try:
                    self.acquire()
                    process = None
                    if active is None:
                        with self.lock:
                            if read_json(directory / "status.json")["state"] != "queued" or self.closing:
                                continue
                            (directory / "control").unlink(missing_ok=True)
                            token = read_json(directory / "request.json")["submission_key"]
                            with (directory / "worker.log").open("ab") as log:
                                process = subprocess.Popen(
                                    [
                                        sys.executable,
                                        "-m",
                                        "r2048.webui.worker",
                                        "--job",
                                        str(directory),
                                        "--token",
                                        token,
                                    ],
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,
                                    # Ctrl+C 只停止服务器；工作进程须等控制文件，在安全边界保存。
                                    start_new_session=True,
                                    env={
                                        **os.environ,
                                        "R2048_OUTPUT_DIR": str(self.root),
                                        "R2048_TASK_CONTROL": str(directory / "control"),
                                        "R2048_SERVER_PID": str(os.getpid()),
                                    },
                                )
                            start = (
                                Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                            )
                            self._status(directory, "running", pid=process.pid, process_start=start)
                    while True:
                        status = read_json(directory / "status.json")
                        alive = process.poll() is None if process else worker_alive(directory, status)
                        if not alive:
                            if status["state"] not in TERMINAL:
                                self._status(directory, "failed", error="工作进程异常退出，查看日志后继续")
                            break
                        if self.closing:
                            write_text_atomic(directory / "control", "stop")
                        time.sleep(0.2)
                except Exception as error:  # noqa: BLE001 - 单个任务失败不能终止服务器队列。
                    self._status(directory, "failed", error=str(error))
                finally:
                    self.release()
        finally:
            self.guard.__exit__(None, None, None)

    def close(self):
        self.closing = True
        self.wake.set()
        if self.thread:
            self.thread.join()
