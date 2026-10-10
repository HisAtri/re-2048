"""来源继承、历史补齐与跨语言分配使用真实文件，覆盖重试和并发的身份边界。"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from r2048.identity import allocate_id, batch_id
from r2048.nn.lineage import (
    file_sha256,
    lineage_fields,
    make_lineage,
    merge_lineages,
    resolve_lineage,
    sources_from_training,
)


def source(key: str, samples: int, batch: str = "") -> dict:
    return {"key": key, "batch_id": batch or key, "samples": samples}


def checkpoint(path: Path, identity: str, **metadata) -> Path:
    torch.save({"model": {}, "metadata": {"model_id": identity, "parent": "", **metadata}}, path)
    return path


def test_inheritance_union_and_half_merge() -> None:
    ancestor = make_lineage(sources_from_training([source("a", 100)]))
    repeated = make_lineage(sources_from_training([source("a", 100)]), [ancestor])
    assert lineage_fields(repeated)["dataset_samples"] == 100
    value = make_lineage(sources_from_training([source("a", 150), source("b", 50)]), [ancestor])
    other = make_lineage(sources_from_training([source("c", 40)]))
    policy = make_lineage(sources_from_training([source("b", 30)]), [other])
    merged = merge_lineages(value, policy)
    fields = lineage_fields(merged)
    assert fields["current_dataset_samples"] == 200
    assert fields["inherited_dataset_samples"] == 140
    assert fields["dataset_samples"] == 240
    assert sources_from_training([{**source("zero", 999), "weight": 0}]) == []
    alias = sources_from_training([source("a", 80, "copy-of-a")])
    assert lineage_fields(make_lineage(alias, [ancestor]))["dataset_samples"] == 100


def test_legacy_parent_reconstruction_and_self_containment(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path / "parent.pt", "parent", sources=[source("a", 100)])
    child = checkpoint(
        tmp_path / "child.pt",
        "child",
        sources=[source("b", 100)],
        parent="parent",
        config_digest="recipe",
        source_checkpoint_sha256=file_sha256(parent),
    )
    (tmp_path / "train.json").write_text(
        json.dumps(
            {
                "config_digest": "recipe",
                "config": {"init_checkpoint": str(parent)},
            }
        )
    )
    # 父模型自己的同目录报告不属于它，历史补齐不能错误引用子模型报告。
    parent_dir = tmp_path / "old"
    parent_dir.mkdir()
    parent = parent.rename(parent_dir / "parent.pt")
    (tmp_path / "train.json").write_text(
        json.dumps(
            {
                "config_digest": "recipe",
                "config": {"init_checkpoint": str(parent)},
            }
        )
    )
    lineage = resolve_lineage(child, strict=True)
    assert lineage_fields(lineage)["dataset_samples"] == 200
    checkpoint(tmp_path / "contained.pt", "contained", **lineage_fields(lineage))
    parent.unlink()
    assert resolve_lineage(tmp_path / "contained.pt", strict=True) == lineage
    assert not resolve_lineage(child)["complete"]
    with pytest.raises(ValueError, match="谱系不完整"):
        resolve_lineage(child, strict=True)


def test_bad_parent_hash_identity_and_cycle(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path / "parent.pt", "parent", sources=[source("a", 100)])
    child = checkpoint(
        tmp_path / "child.pt",
        "child",
        sources=[source("b", 50)],
        parent="parent",
        source_checkpoint_path=str(parent),
        source_checkpoint_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA256"):
        resolve_lineage(child, strict=True)
    checkpoint(
        child,
        "child",
        sources=[source("b", 50)],
        parent="wrong",
        source_checkpoint_path=str(parent),
        source_checkpoint_sha256=file_sha256(parent),
    )
    with pytest.raises(ValueError, match="ID 不匹配"):
        resolve_lineage(child, strict=True)
    with pytest.raises(ValueError, match="循环"):
        resolve_lineage(child, _stack=(child.resolve(),))


def test_cross_language_identity_allocation(tmp_path: Path, monkeypatch) -> None:
    binary = os.environ.get("R2048_CLI")
    if not binary:
        pytest.skip("需要 WSL C++ CLI")
    monkeypatch.setenv("R2048_OUTPUT_DIR", str(tmp_path))

    def allocate(index):
        if index % 2:
            return batch_id("nn", 64)
        output = subprocess.run(
            [binary, "generate", "--policy", "random", "--samples", "1", "--seed", str(index), "--quiet"],
            capture_output=True,
            text=True,
            check=True,
        )
        manifest = Path(output.stdout.strip().splitlines()[-1])
        identity = json.loads(manifest.read_text())["batch_id"]
        assert manifest.parent.name == identity
        return identity

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(allocate, range(6)))
    numbers = [int(identity[-3:]) for identity in ids]
    assert sorted(numbers) == [1, 2, 3, 4, 5, 6]
    assert len(set(ids)) == 6
    cpp_id = ids[0]
    manifest_path = tmp_path / "data" / cpp_id / "manifest.json"
    recorded = json.loads(manifest_path.read_text())
    recorded["state"] = "stopped"
    manifest_path.write_text(json.dumps(recorded))
    subprocess.run(
        [
            binary,
            "generate",
            "--policy",
            "random",
            "--samples",
            "1",
            "--seed",
            "0",
            "--out",
            str(tmp_path / "data" / cpp_id),
            "--continue",
            "--quiet",
        ],
        check=True,
        capture_output=True,
    )
    assert int(batch_id("nn", 64)[-3:]) == 7
    with pytest.raises(FileExistsError):
        allocate_id("release", tmp_path / "models", "20261010001")
        allocate_id("release", tmp_path / "models", "20261010001")
    allocate_id("data", explicit=ids[0][-11:-3] + "999")
    with pytest.raises(ValueError, match="耗尽"):
        batch_id("nn", 64)
