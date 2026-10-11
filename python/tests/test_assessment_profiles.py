"""固定口径、随机抽样与跨任务基线复用的回归验证。"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from r2048.fileio import read_json
from r2048.randomness import persist_seeds
from r2048.workflows import assessment


def test_random_seeds_are_fresh_for_new_tasks_and_retained_on_resume(tmp_path):
    first = persist_seeds(tmp_path / "first.json", {"seed": -1}, ("seed",))
    second = persist_seeds(tmp_path / "second.json", {"seed": -1}, ("seed",))
    resumed = persist_seeds(tmp_path / "first.json", {"seed": -1}, ("seed",))
    assert first == resumed
    assert first["seed"] >= 0 and first != second
    assert persist_seeds(tmp_path / "explicit.json", {"seed": 0}, ("seed",))["seed"] == 0
    with pytest.raises(ValueError):
        persist_seeds(tmp_path / "invalid.json", {"seed": -2}, ("seed",))


def test_p1000s64_is_fixed():
    definition = assessment.load_assessment_definition("P1000S64")
    assert definition["track"] == "nn_mcts" and definition["baseline"] == {"policy": "mcts"}
    profile = assessment.load_assessment_profile("P1000S64")
    assert (profile["games"], profile["simulations"], profile["seed"]) == (1000, 64, -1)
    assert (profile["lanes"], profile["in_flight"], profile["max_batch"]) == (64, 8, 8192)


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_concurrent_tasks_share_completed_baselines(tmp_path, monkeypatch):
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path))
    profile = {**assessment.load_assessment_profile("P1000S64"), "games": 2, "simulations": 4}
    monkeypatch.setattr(
        assessment,
        "load_assessment_definition",
        lambda _: {
            "evaluation": dict(profile),
            "track": "nn_mcts",
            "baseline": {"policy": "mcts"},
        },
    )
    calls = []
    run_cpp = assessment.run_cpp

    def counted(*args, **kwargs):
        calls.append(args[1])
        return run_cpp(*args, **kwargs)

    monkeypatch.setattr(assessment, "run_cpp", counted)

    def evaluate(name):
        return assessment.evaluate_group(tmp_path / name, [], [], {}, profile="small")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(evaluate, ("first", "second"))
    assert len(calls) == 1
    assert calls[0]["policy"] == "mcts"
    assert first["jsonl"] == second["jsonl"]
    assert first["seed"] != second["seed"]
    assert read_json(tmp_path / "first/seeds.json")["seed"] == first["seed"]
    initialize = {"search": next(iter(first["jsonl"]))}
    files = list(first["jsonl"].values())
    assert assessment.ingest_and_rate(tmp_path / "first", files, initialize=initialize)["ingested"] == 2
    assert assessment.ingest_and_rate(tmp_path / "second", files, initialize=initialize)["ingested"] == 0
    assert evaluate("third")["jsonl"] == first["jsonl"]
    assert len(calls) == 1
    profile["simulations"] = 8
    assert evaluate("changed")["jsonl"] != first["jsonl"]
    assert len(calls) == 2


@pytest.mark.skipif(not os.environ.get("R2048_CLI"), reason="需要 WSL C++ CLI")
def test_assessment_worker_uses_profile_and_reuses_baselines(tmp_path, monkeypatch):
    from r2048.nn.model import build, save_checkpoint
    from r2048.webui.worker import run_request, validate_request

    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path))
    profile = {
        **assessment.load_assessment_profile("P1000S64"),
        "games": 2,
        "simulations": 4,
        "lanes": 2,
        "in_flight": 1,
        "device": "cpu",
    }
    monkeypatch.setattr(
        assessment,
        "load_assessment_definition",
        lambda _: {
            "evaluation": dict(profile),
            "track": "nn_mcts",
            "baseline": {"policy": "mcts"},
        },
    )
    model = build("sh-s6c64", {"channels": 8, "blocks": 1, "head_width": 16, "groups": 2})
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(str(checkpoint), model, {**model.metadata("tiny"), "training_target": "dual"})
    request = {
        "kind": "assessment",
        "config": {
            "checkpoint": str(checkpoint),
            "track": "nn_mcts",
            "assessment_profile": "small",
            "evaluation": {"games": 999, "seed": 3},
        },
    }
    validate_request(request, tmp_path)
    first = run_request(tmp_path / "job-first", request)
    second = run_request(tmp_path / "job-second", request)
    assert first["assessment_profile"] == "small"
    assert first["games"] == second["games"] == 2
    assert first["seed"] != second["seed"]
    shared = {label: path for label, path in first["jsonl"].items() if label.startswith("small-")}
    assert len(shared) == 1
    assert all(second["jsonl"][label] == path for label, path in shared.items())
    assert first["ingested"] == 4 and second["ingested"] == 2
    rows = [json.loads(line) for line in Path(first["jsonl"]["job-first"]).read_text().splitlines()]
    assert len(rows) == 2 and all(row["seed"] == first["seed"] for row in rows)
    request["config"]["track"] = "nn_value"
    with pytest.raises(ValueError, match="只支持 nn_mcts"):
        run_request(tmp_path / "job-invalid", request)
