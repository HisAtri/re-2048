"""验证发布身份、两半数据谱系与无损转换；用真实文件校验发布目录的契约。"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from r2048.nn.lineage import lineage_fields, make_lineage, sources_from_training
from r2048.nn.model import DUAL_TOWER_ARCHITECTURE, create_model, load_checkpoint, save_checkpoint
from r2048.webui.inventory import list_models
from r2048.workflows.merge import merge_halves
from r2048.workflows.publish import publish_model


def test_publish_merged_model_preserves_weights_and_both_sources(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    model = create_model({"architecture": "dt-s3p3v3c64"})
    value, policy = tmp_path / "value.pt", tmp_path / "policy.pt"
    for path, target, sources in (
        (value, "value", [("a", 100), ("b", 200)]),
        (policy, "policy", [("a", 150), ("c", 300)]),
    ):
        metadata = model.metadata(target, parent="previous-release")
        metadata.update(
            training_target=target,
            sources=[
                {"batch_id": identity, "key": f"stable-{identity}", "samples": samples}
                for identity, samples in sources
            ],
            source_checkpoint_sha256="a" * 64,
        )
        if target == "value":
            metadata["architecture"] = "dt-s3p3v3c64"
            # 一半来自历史元数据时仍须按同一份稳定 key 合并，不能将 a 算两遍。
            del metadata["sources"][0]["batch_id"]
        metadata.update(
            lineage_fields(
                make_lineage(
                    sources_from_training(metadata["sources"]),
                    parent_models=[{"model_id": "previous-release", "sha256": "a" * 64}],
                )
            )
        )
        save_checkpoint(str(path), model, metadata)
    merged = tmp_path / "merged.pt"
    merge_halves(value, policy, merged)
    root = tmp_path / "models"
    result = publish_model(merged, "20261007001", root)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.model_path.name == "smnet-b9c64dt-20261007001-s650.safetensor"
    assert manifest["model_sha256"] == hashlib.sha256(result.model_path.read_bytes()).hexdigest()
    assert manifest["dataset_ids"] == ["a", "b", "c"]
    assert manifest["dataset_samples"] == 650
    assert manifest["parent_models"] == [{"model_id": "previous-release", "sha256": "a" * 64}]
    restored, metadata = load_checkpoint(str(result.model_path), torch.device("cpu"))
    assert metadata["architecture"] == DUAL_TOWER_ARCHITECTURE
    assert metadata["model_id"] == manifest["model_name"]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])
    board = torch.randn(4, 25, 4, 4)
    with torch.no_grad():
        for before, after in zip(model(board), restored(board)):
            assert torch.equal(before, after)
    assert list_models(root)[0].path == result.model_path
    with pytest.raises(FileExistsError):
        publish_model(merged, "20261007001", root)
    assert hashlib.sha256(result.model_path.read_bytes()).hexdigest() == manifest["model_sha256"]
    next_result = publish_model(merged, output=root)
    assert next_result.manifest_path.exists()
    assert not list(root.glob(".publish-*"))


def test_publish_rejects_incomplete_and_mislabeled_models(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    root = tmp_path / "models"
    model = create_model({"architecture": DUAL_TOWER_ARCHITECTURE, "architecture_params": {"channels": 32}})
    metadata = model.metadata("small")
    metadata["training_target"] = "value"
    save_checkpoint(str(checkpoint), model, metadata)
    with pytest.raises(ValueError, match="双头完整"):
        publish_model(checkpoint, output=root)
    metadata["training_target"] = "dual"
    save_checkpoint(str(checkpoint), model, metadata)
    with pytest.raises(ValueError, match="默认规格"):
        publish_model(checkpoint, output=root)
    assert not root.exists()
