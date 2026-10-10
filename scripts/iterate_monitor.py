#!/usr/bin/env python3
"""连续自举的监控表：逐轮汇总标签诊断 + 四类评测臂 + 逐轮按局配对差。

为什么要有它：这条线每加一轮都要回答同一个问题——"有没有劣化、还涨不涨"。答案由三组量
共同给出，缺一组就会读错：

1. **标签诊断**（`scripts/advantage_probe.py`）：同根 Â std 中位、自洽度、Q̄ 均值、π̂ 熵。
   它们是 ADR 0030 §3 的塌缩预警信号——旧线的退化在评测变差之前就先在这里显现
   （同根 std 55.5 → 17.2 → 16.5）。**同根 std 是其中最有判别力的一个**：自洽度会随教师
   从纯 rollout（0.87）降到 NN 自举（0.58）并停在那里，健康线与退化线在这一项上是一样的
   （ADR 0033），所以只看自洽度会把退化读成正常。
2. **四类评测臂**：纯价值 / 纯策略 / NN-MCTS-64 / NN-MCTS-256。搜索臂是真判据，
   纯价值臂是分辨率高得多的先行指标（±270 对 ±1000，见 ADR 0030 §6.3）。
3. **固定外部锚点**：同预算的纯 rollout MCTS。它是唯一不随自举改变的尺子——只看逐轮配对差，
   两轮一起变坏也会显示"没有显著变化"；只有相对锚点的优势才能证明绝对水平在涨。

历史轮（r1–r3）早于本脚本，标签与数据批写在下面的 LEGACY 表里；r4 起从产物的步骤标记
（`steps/selfplay.json`、`steps/train.json`）自动发现，因此新增轮次不必改代码。

用法：
    python scripts/iterate_monitor.py r1 r2 r3 r4 r5 r6
    python scripts/iterate_monitor.py r6 --no-probe      # 跳过标签诊断（快）
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))
from r2048.workflows.evaluate import compare

EXP = Path("/root/r2048/output/exp")
# 缓存按输出字段的版本分文件：字段变了就换名字，避免读到旧格式的锚点串。
PROBE_CACHE = EXP / "label_anchors_v2.json"
# 锚点 = 同预算的纯 rollout MCTS（评分池里的 mcts256 记录，400 局、seed 20262053）。
ANCHOR = Path("/root/r2048/output/rating/pool.jsonl")
# 早于监控脚本的三轮：这里的名字是当时留下的产物名，不按新命名规则推。
LEGACY = {
    "r1": (
        "tG",
        "mcts64greedy-1m",
        {"value": "tG-value", "policy": "tG-policy", "nn64": "tG-nn64-if16", "nn256": "tG-nn256-if16"},
    ),
    "r2": (
        "g1-r2",
        "g1-r2",
        {
            "value": "g1r2-value",
            "policy": "g1-r2-policy",
            "nn64": "g1r2-nn64-if16",
            "nn256": "g1r2-nn256-if16",
        },
    ),
    "r3": (
        "g1-r3",
        "g1-r3",
        {
            "value": "g1r3-value",
            "policy": "g1-r3-policy",
            "nn64": "g1r3-nn64-if16",
            "nn256": "g1r3-nn256-if16",
        },
    ),
}
ARMS = ("value", "policy", "nn64", "nn256")


def round_spec(name: str) -> dict:
    if name in LEGACY:
        model, batch, labels = LEGACY[name]
        return {"name": name, "model": EXP / model, "batch_name": batch, "labels": labels}
    # 轮名与目录名不必相同：这条线的迭代名是 g1-r4 而人说的是 r4，两者都接受
    # （产物名是历史事实，不该逼着人每次去查）。
    directory = EXP / name if (EXP / name).is_dir() else EXP / f"g1-{name}"
    if not directory.is_dir():
        raise SystemExit(f"找不到轮目录：{EXP / name} 或 {EXP / f'g1-{name}'}")
    published = directory / "steps" / "publish.json"
    model_step = published if published.exists() else directory / "steps" / "train.json"
    model = json.loads(model_step.read_text())["model"]
    batch = Path(json.loads((directory / "steps" / "selfplay.json").read_text())["data_dir"])
    return {
        "name": name,
        "model": Path(model),
        "batch_name": batch.name,
        "batch": batch,
        "labels": {arm: f"{directory.name}-{arm}" for arm in ARMS},
    }


def label_anchors(spec: dict, enabled: bool) -> str:
    """标签诊断（同根 std / 自洽度 / Q̄ 均值）。结果按批缓存：探针要读整批数据，不便宜。"""
    cache = json.loads(PROBE_CACHE.read_text()) if PROBE_CACHE.exists() else {}
    if spec["batch_name"] in cache:
        return cache[spec["batch_name"]]
    if not enabled:
        return "—"
    batch = spec.get("batch") or Path("/root/r2048/output/data") / spec["batch_name"]
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "advantage_probe.py"), "--batch", str(batch)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    line = [row for row in out.splitlines() if row.startswith(batch.name)]
    if not line:
        return "（探针无输出）"
    anchors = " ".join(line[0].split()[2:])
    cache[spec["batch_name"]] = anchors
    PROBE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    return anchors


def stats(label: str, games: int = 400) -> dict | None:
    path = EXP / f"eval-{label}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < games:
        return None
    scores = sorted(row["score"] for row in rows)
    return {
        "path": path,
        "mean": sum(scores) / len(scores),
        "median": scores[len(scores) // 2],
        "steps": sum(row["steps"] for row in rows) / len(rows),
        "t2048": sum(1 for row in rows if (row.get("max_exponent") or 0) >= 11) / len(rows),
        "t4096": sum(1 for row in rows if (row.get("max_exponent") or 0) >= 12) / len(rows),
    }


def anchor_jsonl() -> Path:
    """把评分池里的 mcts256 记录转成逐局评测记录的形状（配对要求同 seed、同局号）。"""
    target = Path("/tmp/iterate_monitor_mcts256.jsonl")
    rows = []
    for line in ANCHOR.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("model_id") == "mcts256" and entry.get("track") == "search":
            rows.append(
                {
                    "episode": entry["episode_id"],
                    "score": entry["score"],
                    "steps": entry["steps"],
                    "max_exponent": entry.get("max_exp"),
                    "label": "mcts256",
                    "seed": entry.get("eval_seed"),
                }
            )
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rounds", nargs="+", help="轮名，按顺序给出（r1 r2 …）")
    parser.add_argument("--no-probe", action="store_true", help="跳过标签诊断（未缓存时显示 —）")
    args = parser.parse_args()

    specs = [round_spec(name) for name in args.rounds]
    print(
        "轮 | 数据批 | 标签诊断（同根 Â std / r std / 自洽度 A 与 r+A / r 翻转 / Q̄ 均值） | "
        "纯价值 | 纯策略 | NN-MCTS-64 | NN-MCTS-256 | 步数 | 到2048 | 到4096"
    )
    for spec in specs:
        anchors = label_anchors(spec, not args.no_probe)
        cells = []
        for arm in ARMS:
            info = stats(spec["labels"][arm])
            cells.append(f"{info['mean']:,.0f}" if info else "—")
        heavy = stats(spec["labels"]["nn256"])
        tail = (
            f"| {heavy['steps']:.0f} | {heavy['t2048'] * 100:.0f}% | {heavy['t4096'] * 100:.0f}%"
            if heavy
            else "| — | — | —"
        )
        print(f"{spec['name']} | {spec['batch_name']} | {anchors} | {' | '.join(cells)} {tail}")

    print("\n按局配对差（同 seed 20262053、400 局；纯网络臂分辨率 ±270、搜索臂 ±1000）：")
    for before, after in itertools.pairwise(specs):
        for arm in ARMS:
            first, second = stats(before["labels"][arm]), stats(after["labels"][arm])
            if not first or not second:
                continue
            result = compare(first["path"], second["path"])
            print(
                f"  {before['name']}→{after['name']} {arm:6s} {result['mean_difference']:+10.1f} "
                f"[{result['ci95'][0]:+9.1f}, {result['ci95'][1]:+9.1f}] "
                f"{'显著' if result['significant'] else '不显著'} {result['wins']}/{result['draws']}/{result['losses']}"
            )

    anchor = anchor_jsonl()
    print("\n相对固定外部锚点（同预算纯 rollout MCTS-256，均分 22,600，配对 400 局）：")
    for spec in specs:
        info = stats(spec["labels"]["nn256"])
        if not info:
            continue
        result = compare(str(anchor), info["path"])
        print(
            f"  {spec['name']:3s} NN-MCTS-256 {result['candidate_mean']:9.1f} "
            f"{result['candidate_mean'] / result['baseline_mean']:6.1%} 相对锚点  "
            f"{result['mean_difference']:+9.1f} [{result['ci95'][0]:+9.1f}, {result['ci95'][1]:+9.1f}] "
            f"{'显著' if result['significant'] else '不显著'}"
        )


if __name__ == "__main__":
    main()
