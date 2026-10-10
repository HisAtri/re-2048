"""本机实测参数的共同入口；显式值优先，正式任务不隐式运行 benchmark。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

from r2048 import paths
from r2048.fileio import read_json, write_json_atomic
from r2048.identity import file_lock

DEFAULTS = {
    "selfplay": {"workers": 4, "lanes": 64, "in_flight": 8, "max_batch": 16384},
    "evaluate": {"lanes": 32, "in_flight": 4, "max_batch": 8192},
    "train": {"batch_size": 4096},
}


def profile_path():
    return paths.output_root() / "benchmarks/machines.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def machine_components(device):
    import r2048_core
    import torch

    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
    gpu = None
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("当前 CUDA 不可用，请选择 CPU")
        properties = torch.cuda.get_device_properties(torch.device(device))
        gpu = {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "memory_bytes": properties.total_memory,
        }
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "device": str(device),
        "build_id": str(r2048_core.BUILD_ID),
    }


def workload(stage, settings, checkpoint="", recipe=None):
    if checkpoint:
        from r2048.nn.model import read_checkpoint_payload

        model = read_checkpoint_payload(checkpoint)["metadata"]
    else:
        model = recipe or {}
    if stage == "train" and not model.get("architecture"):
        from r2048.nn.training import default_architecture

        model = {**model, "architecture": default_architecture(settings.get("mode", "value"))}
    shape = {key: model.get(key) for key in ("architecture", "architecture_params")}
    if shape["architecture"]:
        from r2048.nn.models import REGISTRY, canonical_id, default_params

        shape["architecture"] = canonical_id(shape["architecture"])
        shape["architecture_params"] = {
            **default_params(REGISTRY[shape["architecture"]]),
            **(shape["architecture_params"] or {}),
        }
    # lanes/in_flight 属于搜索协议。画像按固定工作负载匹配，不能把无标注的测量用于 R=8。
    keys = (
        (
            "simulations",
            "tree_nodes",
            "tree_label_plays",
            "tree_label_policy",
            "tree_label_rollout",
            "rollout_policy",
        )
        if stage == "selfplay"
        else ("track", "simulations")
        if stage == "evaluate"
        else ("mode", "freeze_value")
    )
    defaults = {
        "simulations": 64,
        "tree_nodes": 2,
        "tree_label_plays": 8,
        "tree_label_policy": "top_visits",
        "tree_label_rollout": "greedy",
        "rollout_policy": "greedy",
        "track": "nn_mcts",
        "mode": "value",
        "freeze_value": False,
    }
    context = {
        "stage": stage,
        "model": shape,
        "teacher": ("nn" if checkpoint else "mcts") if stage == "selfplay" else "model",
        "precision": "float32",
        **{key: settings.get(key, defaults[key]) for key in keys},
    }
    # v1 基准固定的协议默认值。非默认协议另建工作负载，不能沿用默认参数的建议。
    protocol_defaults = {
        "search_threads": 1,
        "c_puct": 1.5,
        "cache_entries": 131072,
        "temperature": 1.0,
        "temperature_moves": 30,
    }
    if stage in ("selfplay", "evaluate"):
        for key, fallback in protocol_defaults.items():
            if stage == "evaluate" and key in ("temperature", "temperature_moves"):
                continue
            value = settings.get(key, fallback)
            if value != fallback:
                context[key] = value
    return context


def resolve(stage, settings, checkpoint="", recipe=None, mode="auto", default_values=None):
    effective = dict(settings)
    defaults = {**DEFAULTS[stage], **(default_values or {})}
    missing = [key for key in defaults if effective.get(key) is None]
    info = {"mode": mode, "profile": str(profile_path()), "sources": {}, "reason": "全部参数已显式指定"}
    measured = {}
    if mode == "auto" and missing:
        machine = machine_components(
            "cpu" if stage == "selfplay" and not checkpoint else effective.get("device", "cuda")
        )
        context = workload(stage, effective, checkpoint, recipe)
        identity, key = digest(machine), digest(context)
        info.update(machine_id=identity, workload_id=key, workload=context)
        path = profile_path()
        payload = read_json(path) if path.exists() else {"version": 1, "machines": {}}
        if payload.get("version") != 1:
            raise ValueError("性能画像版本不匹配，请使用当前项目的 benchmark probe 重新测量")
        entry = payload["machines"].get(identity, {}).get("workloads", {}).get(key)
        measured = entry["recommended"] if entry else {}
        info["reason"] = "匹配本机实测画像" if entry else "没有匹配的本机与工作负载画像，使用默认值"
        if any(
            effective.get(key) is not None and effective[key] != measured[key]
            for key in ("lanes", "in_flight", "max_batch")
            if key in measured
        ):
            measured = {}
            info["reason"] = "显式搜索参数与实测配置不同，未指定项使用默认值"
    elif mode == "off":
        info["reason"] = "已关闭性能画像"
    for key, fallback in defaults.items():
        if effective.get(key) is not None:
            info["sources"][key] = "explicit"
        elif key in measured:
            effective[key] = measured[key]
            info["sources"][key] = "benchmark"
        else:
            effective[key] = fallback
            info["sources"][key] = "default"
        if not isinstance(effective[key], int) or effective[key] <= 0:
            raise ValueError(f"性能参数 {key} 必须为正整数")
    return effective, info


def save_measurement(stage, settings, checkpoint, measurements, recommended, recipe=None):
    machine = machine_components(settings.get("device", "cuda"))
    context = workload(stage, settings, checkpoint, recipe)
    path = profile_path()
    with file_lock(path.with_suffix(".lock")):
        payload = read_json(path) if path.exists() else {"version": 1, "machines": {}}
        if payload.get("version") != 1:
            raise ValueError("性能画像版本不匹配")
        entry = payload["machines"].setdefault(digest(machine), {"components": machine, "workloads": {}})
        entry["workloads"][digest(context)] = {
            "context": context,
            "checkpoint": checkpoint,
            "measurements": measurements,
            "recommended": recommended,
        }
        write_json_atomic(path, payload)
    return path
