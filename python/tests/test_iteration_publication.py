"""真实迭代验证发布身份贯穿评测、继承与中断恢复，不用模拟发布结果代替产物。"""

import json
import os
from pathlib import Path

import pytest
import torch

from r2048.fileio import read_json, write_json_atomic
from r2048.identity import allocate_id
from r2048.iterate import IterateConfig, _step_publish, run_iteration
from r2048.nn.lineage import file_sha256


def test_two_iterations_publish_inherit_and_recover(tmp_path: Path, monkeypatch) -> None:
    if not os.environ.get("R2048_CLI"):
        pytest.skip("需要 WSL C++ CLI")
    torch.set_num_threads(1)
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path / "output"))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""train:
  architecture: smnet-b9c64dt
  epochs: 1
  batch_size: 128
  max_samples: 512
  val_fraction: 0.25
  seed: 3
  device: cpu
  log_every: 0
""")
    common = {
        "config_path": str(recipe),
        "samples": 4000,
        "data_seed": 3,
        "simulations": 8,
        "data_workers": 1,
        "device": "cpu",
        "lanes": 4,
        "in_flight": 4,
        "tree_nodes": 0,
        "eval_games": 1,
        "eval_seed": 3,
    }
    first = run_iteration(IterateConfig(**common))
    first_manifest = read_json(first.steps["publish"]["manifest"])
    assert first.name.startswith("iter-")
    batch = read_json(first.steps["selfplay"]["manifest"])
    assert first_manifest["dataset_samples"] == batch["samples"]
    assert first_manifest["inherited_dataset_samples"] == 0
    for label, artifact in first.steps["evaluate"]["jsonl"].items():
        # value 诊断保留训练身份，完整模型的两条臂必须使用发布身份。
        if label.endswith("policy") or "nnmcts" in label:
            row = json.loads(Path(artifact).read_text().splitlines()[0])
            assert row["model_id"] == first_manifest["model_id"]
    second = run_iteration(
        IterateConfig(
            **{**common, "samples": 2000},
            init_checkpoint=first.steps["publish"]["model"],
        )
    )
    second_manifest = read_json(second.steps["publish"]["manifest"])
    new_batch = read_json(second.steps["selfplay"]["manifest"])
    assert second_manifest["dataset_samples"] == batch["samples"] + new_batch["samples"]
    assert second_manifest["parent_models"][0]["model_id"] == first_manifest["model_id"]
    assert second_manifest["model_file"].endswith(f"-s{batch['samples'] + new_batch['samples']}.safetensor")

    # 模拟发布目录已经提交、步骤记录没落盘；恢复必须认出同一产物。
    directory = Path(second.directory)
    (directory / "steps" / "publish.json").unlink()
    restored = run_iteration(IterateConfig(config_path="", resume=str(directory), steps=("publish",)))
    assert restored.steps["publish"] == second.steps["publish"]
    assert len(list((tmp_path / "output" / "models").glob("*/manifest.json"))) == 2

    # 上游重做只失效下游记录，既有发布目录仍然保留。
    repeated = run_iteration(
        IterateConfig(config_path="", resume=str(directory), steps=("train",), force=True)
    )
    assert repeated.steps["train"]["model"] != second.steps["train"]["model"]
    assert not (directory / "steps" / "publish.json").exists()
    assert not (directory / "steps" / "evaluate.json").exists()
    assert not (directory / "steps" / "rate.json").exists()
    assert Path(second.steps["publish"]["model"]).exists()


def test_publication_recovery_rejects_changed_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path / "output"))
    model = tmp_path / "model.pt"
    torch.save({"model": {}, "metadata": {}}, model)
    config = IterateConfig(config_path="recipe.yaml", name="x", output=str(tmp_path / "iter"))
    root = tmp_path / "output" / "models"
    release = allocate_id("release", root)
    write_json_atomic(
        Path(config.output) / "requests" / "publish.json",
        {
            "release_id": release,
            "source_checkpoint_sha256": file_sha256(model),
        },
    )
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="已变化"):
        _step_publish(config, {"train": {"model": str(model)}}, None)
