"""WSL 真子进程覆盖持久队列、停止恢复和正式产物衔接。"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from r2048.fileio import read_json, write_json_atomic
from r2048.webui.tasks import TERMINAL, TaskManager


def wait_until(predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("后台任务超时")


def finished(manager, identity):
    record = manager.detail(identity)
    if record["state"] not in TERMINAL:
        return None
    assert record["state"] == "finished", record.get("log", record)
    return record


def recipe():
    return {
        "train": {
            "architecture": "smnet-b9c64dt",
            "epochs": 1,
            "batch_size": 128,
            "max_samples": 512,
            "val_fraction": 0.25,
            "seed": 3,
            "device": "cpu",
            "log_every": 1,
        }
    }


def generation(samples=4000):
    return {
        "samples": samples,
        "data_seed": 3,
        "simulations": 8,
        "data_workers": 1,
        "device": "cpu",
        "lanes": 4,
        "in_flight": 4,
        "tree_nodes": 0,
    }


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_queue_deduplicates_cancels_and_runs_full_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path))
    gate = threading.Event()
    occupied = []
    manager = TaskManager(tmp_path, acquire=lambda: gate.wait(), release=lambda: occupied.append(False))
    request = {
        "kind": "iterate",
        "config": {
            "recipe": recipe(),
            "iteration": {
                **generation(),
                "eval_games": 1,
                "eval_seed": 3,
                "eval_simulations": 4,
                "baselines": ["random", "mcts4"],
            },
        },
    }
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: manager.submit(request, "same-preview"), range(4)))
        assert len(set(ids)) == 1 and len(manager.list()) == 1
        cancel = manager.submit({"kind": "selfplay", "config": {"selfplay": generation()}}, "cancel")
        manager.stop(cancel)
        assert manager.detail(cancel)["state"] == "cancelled"
        malformed = tmp_path / "eval" / "malformed.jsonl"
        malformed.parent.mkdir()
        malformed.write_text("invalid-json")
        failure = manager.submit({"kind": "rating", "config": {"files": [str(malformed)]}}, "fail")
        following = manager.submit({"kind": "selfplay", "config": {"selfplay": generation(100)}}, "following")
        gate.set()
        result = wait_until(lambda: finished(manager, ids[0]))
        assert Path(result["result"]["model"]).suffix == ".safetensor"
        manifest = read_json(result["result"]["manifest"])
        assert manifest["inherited_dataset_samples"] == 0
        assert manifest["parent_models"] == []
        assert manifest["dataset_samples"] == result["model_identity"]["dataset_samples"]
        iteration = Path(result["result"]["iteration"])
        assert read_json(iteration / "iteration.json")["name"].startswith("iter-")
        evaluation = read_json(iteration / "steps" / "evaluate.json")
        from r2048.workflows.rating import load_pool

        rows = load_pool(tmp_path / "rating/pool.jsonl")
        assert any(row["model_id"] == manifest["model_id"] for row in rows)
        assert evaluation["protocol"]["simulations"] == 4
        wait_until(lambda: manager.detail(failure)["state"] == "failed")
        wait_until(lambda: finished(manager, following))
        wait_until(lambda: len(occupied) >= 3)
        assert read_json(tmp_path / "rating/anchors.json")["tracks"].keys() == {"pure", "search"}
        # 独立双头训练调用同一工作流，继承并重复使用同一批数据只计一次。
        batch = read_json(iteration / "steps" / "selfplay.json")
        trained = manager.submit(
            {
                "kind": "train",
                "config": {
                    "recipe": recipe(),
                    "checkpoint": result["result"]["model"],
                    "datasets": [{"path": batch["data_dir"], "weight": 1, "limit": 0}],
                },
            },
            "standalone",
        )
        independent = wait_until(lambda: finished(manager, trained))
        inherited = independent["model_identity"]
        assert inherited["dataset_samples"] == manifest["dataset_samples"]
        assert inherited["inherited_dataset_samples"] == manifest["dataset_samples"]
    finally:
        gate.set()
        manager.close()


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_selfplay_stop_resume_keeps_batch_and_queue_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path))
    manager = TaskManager(tmp_path)
    identity = manager.submit({"kind": "selfplay", "config": {"selfplay": generation(40000)}}, "stoppable")
    try:
        batch = wait_until(lambda: next(iter((tmp_path / "data").glob("*/manifest.json")), None))
        manager.stop(identity)
        wait_until(lambda: manager.detail(identity)["state"] in TERMINAL)
        assert manager.detail(identity)["state"] == "stopped", manager.detail(identity)
        before = read_json(batch)
        assert before["state"] in ("paused", "stopped")
        manager.close()
        manager = TaskManager(tmp_path)
        manager.start()
        assert manager.detail(identity)["state"] == "stopped"
        manager.resume(identity)
        final = wait_until(lambda: finished(manager, identity))
        assert final["result"]["batch_id"] == before["batch_id"]
        assert read_json(batch)["samples"] >= 40000
        assert len(list((tmp_path / "data").glob("*/manifest.json"))) == 1
    finally:
        manager.close()


def test_restart_marks_unverified_process_interrupted(tmp_path):
    job = tmp_path / "jobs/job-20261010001"
    write_json_atomic(job / "request.json", {"version": 1, "submission_key": "unknown"})
    write_json_atomic(job / "status.json", {"task_id": job.name, "state": "running", "pid": os.getpid()})
    manager = TaskManager(tmp_path)
    try:
        manager.start()
        assert manager.detail(job.name)["state"] == "interrupted"
    finally:
        manager.close()


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_incomplete_evaluation_never_commits_and_completed_arm_is_reused(tmp_path, monkeypatch):
    from r2048.task_control import TaskStopped
    from r2048.workflows import assessment

    control = tmp_path / "control"
    control.write_text("stop")
    monkeypatch.setenv("R2048_TASK_CONTROL", str(control))
    parameters = {"games": 4, "seed": 37}
    arm = [("random", {"policy": "random"})]
    with pytest.raises(TaskStopped):
        assessment.evaluate_group(tmp_path / "eval", [], arm, parameters)
    assert not list((tmp_path / "eval").glob("*.jsonl"))
    control.unlink()
    result = assessment.evaluate_group(tmp_path / "eval", [], arm, parameters)
    assert len(Path(result["jsonl"]["random"]).read_text().splitlines()) == 4

    def unexpected(*args, **kwargs):
        pytest.fail("已完成的评测臂不应重跑")

    monkeypatch.setattr(assessment, "run_cpp", unexpected)
    assert assessment.evaluate_group(tmp_path / "eval", [], arm, parameters) == result


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_cpp_evaluation_stop_at_game_boundary_leaves_no_partial_jsonl(tmp_path):
    from r2048.jobs import run_cpp

    control = tmp_path / "control"
    control.write_text("stop")
    result = run_cpp(
        "evaluate",
        {
            "policy": "random",
            "games": 20,
            "workers": 2,
            "control": str(control),
            "jsonl": str(tmp_path / "partial.jsonl"),
        },
        echo=False,
    )
    import json

    assert json.loads(result.last_line)["games"] == 0
    assert not (tmp_path / "partial.jsonl").exists()


def test_validation_and_bounded_log_tail(tmp_path):
    from r2048.webui.worker import validate_request

    with pytest.raises(ValueError, match="samples"):
        validate_request({"kind": "selfplay", "config": {"selfplay": generation(-1)}}, tmp_path)
    with pytest.raises(ValueError, match="请选择完整模型"):
        validate_request({"kind": "assessment", "config": {}}, tmp_path)
    directory = tmp_path / "jobs/job-20261010001"
    write_json_atomic(directory / "request.json", {"version": 1, "submission_key": "tail"})
    write_json_atomic(directory / "status.json", {"state": "finished"})
    (directory / "worker.log").write_text("\n".join(str(index) * 100 for index in range(1000)))
    log = TaskManager(tmp_path).detail(directory.name)["log"]
    assert len(log.splitlines()) <= 100 and len(log.encode()) <= 65536


def test_compute_reservation_preserves_board_and_editor():
    from r2048.session_io import SessionError
    from r2048.webui.app import AppState

    state = AppState()
    state.last_state = {"board": [1] + [0] * 15}
    state.reserve_compute()
    try:
        assert state.auto_stop
        with pytest.raises(SessionError, match="实验任务"):
            state.attach("", "cpu", "search", 4, 0)
        with pytest.raises(SessionError, match="实验任务"):
            state.call("genmove")
        state.editor[4] = 3
        assert state.last_state["board"][0] == 1
    finally:
        state.release_compute()
        state.close()
