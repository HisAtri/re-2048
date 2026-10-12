"""本地发布：权重与身份清单一并提交，目录出现就表示转换与校验已完成（ADR 0047）。"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_file

from r2048 import paths
from r2048.fileio import write_json_atomic
from r2048.identity import allocate_id
from r2048.nn.lineage import file_sha256, lineage_fields, resolve_lineage
from r2048.nn.metadata import default_params
from r2048.nn.model import load_checkpoint
from r2048.nn.registry import canonical_id, resolve


@dataclass(frozen=True)
class PublishResult:
    model_path: Path
    manifest_path: Path


def _json_metadata(value):
    # 半模型未训练的头会留下 NaN 指标；发布 JSON 用 null 表达缺席，避免非标准 JSON 与 NaN 比较。
    if isinstance(value, dict):
        return {key: _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_model(
    checkpoint: str | Path, release_id: str = "", output: str | Path = "", *, reserved: bool = False
) -> PublishResult:
    model, metadata = load_checkpoint(str(checkpoint), torch.device("cpu"))
    if metadata.get("training_target") not in ("dual", "joint"):
        raise ValueError("正式发布需要双头完整模型（training_target 为 dual 或 joint）")
    architecture = canonical_id(metadata["architecture"])
    params = model.metadata("publish")["architecture_params"]
    if params != default_params(resolve(architecture)):
        raise ValueError("当前尺寸与正式架构代号的默认规格不一致，请先登记对应的正式代号")
    lineage = resolve_lineage(checkpoint, strict=True)
    datasets = lineage["datasets"]
    samples = sum(entry["samples"] for entry in datasets)
    now = datetime.now(UTC)
    root = Path(output) if output else paths.output_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    if not reserved:
        release_id = allocate_id("release", root, release_id)
    name = f"{architecture}-{release_id}-s{samples}"
    target = root / name
    if target.exists():
        raise FileExistsError(f"发布目录已经存在，不允许覆盖：{target}")
    source_id = str(metadata["model_id"])
    source_sha256 = file_sha256(checkpoint)
    metadata = {
        **metadata,
        "architecture": architecture,
        "architecture_params": params,
        "source_model_id": source_id,
        "model_id": name,
        "release_id": release_id,
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "source_checkpoint_sha256": source_sha256,
        **lineage_fields(lineage),
    }
    metadata = _json_metadata(metadata)
    tensors = {key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()}
    # 在同一文件系统的临时目录完成转换、重载与逐张量比对；最后 rename 提交整个发布目录。
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=root) as temporary:
        staging = Path(temporary)
        model_path = staging / f"{name}.safetensor"
        save_file(
            tensors,
            str(model_path),
            metadata={"metadata": json.dumps(metadata, ensure_ascii=False, allow_nan=False)},
        )
        restored, restored_metadata = load_checkpoint(str(model_path), torch.device("cpu"))
        if restored_metadata != metadata or any(
            not torch.equal(tensor, restored.state_dict()[key]) for key, tensor in tensors.items()
        ):
            raise ValueError("safetensor 转换后权重或元数据不一致，发布取消")
        manifest = {
            "manifest_version": 2,
            "model_name": name,
            "model_id": name,
            "source_model_id": source_id,
            "source_checkpoint_sha256": source_sha256,
            "architecture": architecture,
            "architecture_params": params,
            "model_file": model_path.name,
            "model_sha256": _sha256(model_path),
            "release_id": release_id,
            "published_at": metadata["published_at"],
            "dataset_samples": samples,
            "dataset_ids": sorted(
                {identity for entry in datasets for identity in (entry["batch_ids"] or [entry["source_key"]])}
            ),
            "datasets": datasets,
            "parent_models": lineage["parent_models"],
            **lineage_fields(lineage),
            "weights_dtype": metadata.get("weights_dtype", "float32"),
            "training_target": metadata["training_target"],
        }
        write_json_atomic(staging / "manifest.json", manifest)
        staging.rename(target)
    return PublishResult(target / model_path.name, target / "manifest.json")
