"""评测组与评分提交共用入口；只有完整评测进入评分池。"""

import hashlib
import json
import os
from pathlib import Path

import yaml

from r2048 import paths
from r2048.fileio import write_json_atomic, write_text_atomic
from r2048.identity import file_lock
from r2048.jobs import run_cpp
from r2048.randomness import persist_seeds
from r2048.task_control import check_stop
from r2048.workflows import rating
from r2048.workflows.evaluate import EvalConfig, run_evaluation


def assessment_profiles() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[4] / "configs" / "assessment"
    return {path.stem: path for path in sorted(root.glob("*.yaml"))}


def load_assessment_definition(identity: str) -> dict:
    profiles = assessment_profiles()
    if identity not in profiles:
        raise ValueError(f"未知测评配置：{identity}")
    return yaml.safe_load(profiles[identity].read_text())


def load_assessment_profile(identity: str) -> dict:
    return load_assessment_definition(identity)["evaluation"]


def evaluate_group(directory: Path, arms: list, baselines: list, params: dict, job=None, profile="") -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if profile:
        definition = load_assessment_definition(profile)
        params = definition["evaluation"]
        if any(track != definition["track"] for _, _, track in arms):
            raise ValueError(f"{profile} 只支持 {definition['track']} 测评")
        baseline = definition["baseline"]
        baselines = [(f"mcts{params['simulations']}", {**baseline, "simulations": params["simulations"]})]
    params = persist_seeds(directory / "seeds.json", dict(params), ("seed",))
    files = {}
    for label, checkpoint, track in arms:
        check_stop()
        output = directory / f"{label}.jsonl"
        marker = directory / f"{label}.complete.json"
        if not marker.exists():
            # 正式模型的评分身份跟随发布清单；文件名仍区分此次评测的各条臂。
            model_label = label
            if Path(checkpoint).suffix == ".safetensor":
                from r2048.nn.model import read_checkpoint_payload

                model_label = read_checkpoint_payload(checkpoint)["metadata"]["model_id"]
            run_evaluation(
                EvalConfig(
                    **params,
                    label=model_label,
                    checkpoint=checkpoint,
                    track=track,
                    jsonl=str(output),
                    on_progress=job.callback() if job else None,
                )
            )
            write_json_atomic(marker, {"jsonl": str(output)})
        files[label] = str(output)
    for label, config in baselines:
        check_stop()
        if profile:
            # 基线只依赖规则与纯 C++ 搜索配置；模型和抽样种子不能制造新的基线身份。
            import r2048_core

            spec = {
                "profile": profile,
                "evaluation": load_assessment_profile(profile),
                "baseline": config,
                "rules_version": int(r2048_core.RULES_VERSION),
                "protocol_version": int(r2048_core.PROTOCOL_VERSION),
            }
            digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            label = f"{profile}-{digest}-{label}"
            shared = paths.output_root() / "rating" / "baselines" / label
            with file_lock(shared / ".build.lock"):
                baseline_params = persist_seeds(shared / "seeds.json", dict(params, seed=-1), ("seed",))
                result = evaluate_group(shared, [], [(label, config)], baseline_params, job)
            files[label] = result["jsonl"][label]
            continue
        output = directory / f"{label}.jsonl"
        marker = directory / f"{label}.complete.json"
        if not marker.exists():
            result = run_cpp(
                "evaluate",
                {
                    **config,
                    "games": params["games"],
                    "seed": params["seed"],
                    "label": label,
                    "jsonl": str(output),
                    "quiet": True,
                    "control": os.environ.get("R2048_TASK_CONTROL", ""),
                },
                job=job,
                progress=True,
            )
            check_stop()
            if json.loads(result.last_line)["games"] != params["games"]:
                raise ValueError("基线评测未完成，不能导入评分池")
            write_json_atomic(marker, {"jsonl": str(output)})
        files[label] = str(output)
    check_stop()
    result = {"arms": list(files), "jsonl": files, "games": params["games"], "seed": params["seed"]}
    if profile:
        result["assessment_profile"] = profile
    return result


def rating_report() -> dict:
    root = paths.output_root() / "rating"
    rows = rating.load_pool(root / "pool.jsonl")
    anchors = rating.load_anchors(root / "anchors.json")
    return {"tracks": rating.compute_ratings(rows, anchors), "anchors": anchors, "rows": len(rows)}


def ingest_and_rate(
    directory: Path, files: list, anchors: dict | None = None, initialize: dict | None = None
) -> dict:
    root = paths.output_root() / "rating"
    with file_lock(root / ".write.lock"):
        pool, anchor_path = root / "pool.jsonl", root / "anchors.json"
        rows = rating.load_pool(pool)
        keys = {rating.game_key(row) for row in rows}
        added = 0
        for file in files:
            for line in Path(file).read_text().splitlines():
                if not line.strip():
                    continue
                row = rating.normalize_row(json.loads(line))
                key = rating.game_key(row)
                if key not in keys:
                    rows.append(row)
                    keys.add(key)
                    added += 1
        known = rating.load_anchors(anchor_path)
        # 只初始化缺席的评分线；已有尺子必须通过明确的设置锚点操作替换。
        spec = {track: block["model_id"] for track, block in known.get("tracks", {}).items()}
        spec.update({track: identity for track, identity in (initialize or {}).items() if track not in spec})
        spec.update(anchors or {})
        # 先验证锚点，再原子提交评分池；失败请求不能留下半行或部分导入。
        updated = None
        if spec and (anchors or any(track not in known.get("tracks", {}) for track in spec)):
            temporary = root / "anchors.pending.json"
            updated = rating.save_anchors(temporary, rows, spec)
            temporary.unlink()
        if added:
            write_text_atomic(
                pool, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
            )
        if updated:
            write_json_atomic(anchor_path, updated)
        report = rating_report()
        output = directory / "rating.json"
        write_json_atomic(output, report)
        return {"pool": str(pool), "ingested": added, "rating_json": str(output), "pool_rows": len(rows)}
