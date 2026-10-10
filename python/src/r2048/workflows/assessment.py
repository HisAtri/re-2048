"""评测组与评分提交共用入口；只有完整评测进入评分池。"""

import json
import os
from pathlib import Path

from r2048 import paths
from r2048.fileio import write_json_atomic, write_text_atomic
from r2048.identity import file_lock
from r2048.jobs import run_cpp
from r2048.task_control import check_stop
from r2048.workflows import rating
from r2048.workflows.evaluate import EvalConfig, run_evaluation


def evaluate_group(directory: Path, arms: list, baselines: list, params: dict, job=None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
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
    return {"arms": list(files), "jsonl": files, "games": params["games"], "seed": params["seed"]}


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
