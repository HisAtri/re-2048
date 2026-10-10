"""按稳定来源身份继承数据覆盖量；重复训练不增加规模，缺失谱系不伪装成从头训练。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from r2048.fileio import read_json

LINEAGE_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_sources(*groups: list[dict]) -> list[dict]:
    entries: dict[str, dict] = {}
    for group in groups:
        for source in group:
            key = str(source["source_key"])
            samples = int(source["samples"])
            if not key or samples <= 0:
                raise ValueError("来源缺少稳定 source_key 或有效样本数")
            entry = entries.setdefault(key, {"source_key": key, "batch_ids": [], "samples": 0})
            entry["samples"] = max(entry["samples"], samples)
            entry["batch_ids"] = sorted(set(entry["batch_ids"]) | set(source.get("batch_ids", [])))
    return [entries[key] for key in sorted(entries)]


def sources_from_training(sources: list[dict]) -> list[dict]:
    result = []
    for source in sources:
        if float(source.get("weight", 1)) <= 0:
            continue
        key = source.get("key") or source.get("source_key")
        if not key:
            raise ValueError("历史来源没有稳定 key，不能从批次名推断来源")
        identity = source.get("batch_id")
        result.append(
            {"source_key": key, "batch_ids": [identity] if identity else [], "samples": source["samples"]}
        )
    return union_sources(result)


def make_lineage(
    current: list[dict], parents: Sequence[dict] = (), parent_models: Sequence[dict] = (), issues=()
) -> dict:
    current = union_sources(current)
    inherited = union_sources(*(parent["datasets"] for parent in parents))
    problems = list(issues)
    for parent in parents:
        problems.extend(parent.get("issues", []))
        if not parent["complete"] and not parent.get("issues"):
            problems.append("父模型谱系不完整")
    return {
        "version": LINEAGE_VERSION,
        "complete": not problems,
        "issues": sorted(set(problems)),
        "current": current,
        "inherited": inherited,
        "datasets": union_sources(current, inherited),
        "parent_models": list(parent_models),
    }


def lineage_fields(lineage: dict) -> dict:
    return {
        "data_lineage": lineage,
        "current_dataset_samples": sum(entry["samples"] for entry in lineage["current"]),
        "inherited_dataset_samples": sum(entry["samples"] for entry in lineage["inherited"]),
        "dataset_samples": sum(entry["samples"] for entry in lineage["datasets"]),
    }


def require_complete(lineage: dict) -> dict:
    if not lineage["complete"]:
        raise ValueError("数据谱系不完整，不能正式发布：" + "；".join(lineage["issues"]))
    return lineage


def _reference_path(value: str, owner: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = owner.parent / path
    return local if local.exists() else path.resolve()


def resolve_lineage(checkpoint: str | Path, strict: bool = False, _stack: tuple[Path, ...] = ()) -> dict:
    """新产物自包含；旧产物只循明确路径恢复，绝不按名称搜索一个相似父模型。"""
    from r2048.nn.model import read_checkpoint_payload

    path = Path(checkpoint).resolve()
    if path in _stack:
        raise ValueError(f"数据谱系循环：{path}")
    payload = read_checkpoint_payload(path)
    metadata = dict(payload.get("metadata", {}))
    if "data_lineage" in metadata:
        lineage = metadata["data_lineage"]
        if lineage["version"] != LINEAGE_VERSION:
            raise ValueError(f"不支持的数据谱系版本：{lineage['version']}")
        if union_sources(lineage["current"], lineage["inherited"]) != lineage["datasets"]:
            raise ValueError(f"累计数据谱系与本轮、继承来源不一致：{path}")
        if lineage["complete"] and lineage.get("issues"):
            raise ValueError(f"谱系完整标志与缺口记录不一致：{path}")
        return require_complete(lineage) if strict else lineage
    issues = []
    current = []
    parents = []
    references = []
    stack = (*_stack, path)
    report = {}
    report_path = path.parent / "train.json"
    if report_path.exists():
        report = read_json(report_path)
        if metadata.get("config_digest") != report.get("config_digest"):
            issues.append(f"训练报告与 checkpoint 的配置身份不一致：{report_path}")
            report = {}
    halves = metadata.get("merged_from", {})
    if halves:
        half_lineages = []
        for name in ("value", "policy"):
            reference = halves.get(name, {})
            try:
                half_path = _reference_path(reference["path"], path)
                half_meta = read_checkpoint_payload(half_path)["metadata"]
                if half_meta.get("model_id") != reference["model_id"]:
                    raise ValueError(f"合并来源身份不一致：{half_path}")
                if reference.get("sha256") and file_sha256(half_path) != reference["sha256"]:
                    raise ValueError(f"合并来源摘要不一致：{half_path}")
                half_lineages.append(resolve_lineage(half_path, _stack=stack))
            except (OSError, ValueError, KeyError) as error:
                issues.append(f"{name} 半模型来源缺口：{error}")
        return _merge_lineages(half_lineages, issues, strict)
    try:
        raw = metadata.get("sources") or report.get("data", {}).get("sources", [])
        if not raw:
            raise ValueError(f"未记录本轮逐来源样本数：{path}")
        current = sources_from_training(raw)
    except (ValueError, KeyError) as error:
        issues.append(str(error))
    parent_id = metadata.get("parent", "")
    config = report.get("config", payload.get("config", {}))
    if "parent" not in metadata and "init_checkpoint" not in config:
        issues.append(f"缺少权重初始化来源，不能确认是否从头训练：{path}")
    parent_path_value = metadata.get("source_checkpoint_path") or config.get("init_checkpoint")
    if parent_id or parent_path_value:
        try:
            if not parent_path_value:
                raise ValueError(f"父模型 {parent_id} 缺少明确文件路径：{path}")
            parent_path = _reference_path(parent_path_value, path)
            digest = file_sha256(parent_path)
            expected = metadata.get("source_checkpoint_sha256")
            if expected and digest != expected:
                raise ValueError(f"父模型 SHA256 不匹配：{parent_path}")
            if not expected:
                raise ValueError(f"父模型缺少可验证的 SHA256：{parent_path}")
            parent_meta = read_checkpoint_payload(parent_path)["metadata"]
            if parent_id and parent_meta.get("model_id") != parent_id:
                raise ValueError(f"父模型 ID 不匹配：{parent_path}")
            parents.append(resolve_lineage(parent_path, _stack=stack))
            references.append(
                {"model_id": parent_meta["model_id"], "sha256": digest, "path": str(parent_path)}
            )
        except (OSError, ValueError, KeyError) as error:
            issues.append(str(error))
    lineage = make_lineage(current, parents, references, issues)
    return require_complete(lineage) if strict else lineage


def _merge_lineages(lineages: list[dict], issues=(), strict: bool = False) -> dict:
    current = union_sources(*(item["current"] for item in lineages))
    # 合并件的继承量来自两半各自的父模型；不能把两半的本轮数据再次当作继承量。
    inherited = union_sources(*(item["inherited"] for item in lineages))
    references = {
        (ref["model_id"], ref.get("sha256")): ref for item in lineages for ref in item["parent_models"]
    }
    problems = list(issues) + [problem for item in lineages for problem in item.get("issues", [])]
    for item in lineages:
        if not item["complete"] and not item.get("issues"):
            problems.append("半模型谱系不完整")
    lineage = make_lineage(current, [make_lineage(inherited)], list(references.values()), problems)
    return require_complete(lineage) if strict else lineage


def merge_lineages(*lineages: dict) -> dict:
    return _merge_lineages(list(lineages))
