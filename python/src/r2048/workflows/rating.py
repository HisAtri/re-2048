"""评分：把逐局记录折算成单一标度上的评分，并给出量化差异。

口径沿用旧版 `src/ai2048/rating.py`，两处设计各有明确理由：

* **按局取 `log2(score + 1)` 再算均值**：2048 的终局得分是重尾分布，少数高分局能主导均值，
  直接比均分会让结论随一两局摆动。取对数后 230 分 = 一倍分数，尾部不再主导。
* **CI 用两样本独立公式**（不是配对公式）：同 seed 只对齐开局；2048 只有 480 种开局
  （D4 简并后见 `count_openings`），而第一步之后每次随机块生成都会改变后续轨迹，
  配对只对齐开头几步，方差缩减有限。因此这里按“局是独立样本”给区间，
  另用 `pairing_diagnostic` 把“配对到底省了多少方差”量化出来，而不是默认它有效。

评分 = 1000 + 230 × (模型的对数均值 − 锚点的对数均值)，锚点按 track 固定。
没有锚点的 track 不产出评分：宁可缺，也不要给没有尺度的数字。

“同一 track 才可比”这一条由**协议变量**落实：清单与校验在 `r2048.protocol`，
本模块只负责按 (track, protocol_hash) 分组。分组不是判“能不能比”，而是判
“比的是同一件事”：曲线上的每个模拟数都是不同的协议，它们都放在同一个锚点的尺子上。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from r2048.protocol import build_protocol, protocol_diff, protocol_hash

# 分数下界保护：理论上单局至少发生一次合并，+1 只是避免 score=0 的极端局把 log 拉到负无穷。
SCORE_EPSILON = 1.0
# 标度换算：1 个 log2 单位 = 230 评分点。它只是展示层常数，数据无法辨识该系数。
RATING_LAMBDA = 230.0
RATING_CENTER = 1000.0
Z95 = 1.96

# track 定义：同一 track 内的成绩才可比较。
#   pure   ：单次前向的纯网络策略（锚点 = 随机策略）
#   search ：同一次模拟预算下的搜索（锚点 = 纯 MCTS-rollout）
TRACKS = ("pure", "search")

# 哪些策略属于搜索线。这是从策略名到评分线的映射表，不是猜测：
# 旧实现靠 `row.get("policy") == "mcts"` 推断，而 C++ 评测那时根本不写 track，
# 推断错一次就会把纯 MCTS 的基准锚点从搜索线上挪走。
SEARCH_TRACKS = frozenset({"mcts", "nn_mcts"})

# 锚点文件格式版本。v2 起记录锚点自身的协议本体（v1 只有哈希，读不懂尺子是什么）。
ANCHORS_SCHEMA_VERSION = 2


def score_to_log(score: float) -> float:
    return math.log2(float(score) + SCORE_EPSILON)


def track_of(protocol: dict) -> str:
    """按协议里声明的策略归到评分线上。"""
    return "search" if protocol["track"] in SEARCH_TRACKS else "pure"


def normalize_row(row: dict) -> dict:
    """把评测 JSONL 的一行规范成评分池的一行。

    做这一层是为了让评分模块与评测模块的字段命名解耦：评测关心“这局怎么打的”，
    评分关心“这局值多少分、属于哪条协议”。

    协议变量由 `r2048.protocol` 统一校验：缺字段会在这里失败，而不是被默认值填上。
    """
    protocol = build_protocol(row)
    # 评分身份用 label：同一个 checkpoint 可能被以不同协议评测多次，
    # 而 label 是那次评测的命名（C++ 评测没有 model_id，只有 label）。
    identity = str(row.get("label") or row.get("model_id") or "unknown")
    return {
        "model_id": identity,
        "checkpoint_model_id": str(row.get("model_id", "")),
        "track": track_of(protocol),
        "source_track": protocol["track"],
        # 协议本体与它的哈希一起落盘：只有哈希的池子读不懂，也无法事后解释
        # “这两个 16 进制串为什么不同”。
        "protocol": protocol,
        "protocol_hash": protocol_hash(protocol),
        # eval_seed 不进协议（协议是“怎么测的”，seed 是“测了哪几局”），
        # 但它是去重键的一部分：同一协议下不同 seed 的局不能算重复。
        "eval_seed": int(row["seed"]),
        "episode_id": int(row["episode"]),
        "score": int(row["score"]),
        "max_exp": int(row.get("max_exponent", 0)),
        "steps": int(row.get("steps", 0)),
    }


def game_key(row: dict) -> tuple:
    return (
        row["model_id"],
        row["track"],
        row["protocol_hash"],
        row["eval_seed"],
        row["episode_id"],
    )


def load_pool(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_games(path: str | Path, rows: list[dict]) -> int:
    """追加去重后的局。重复 ingest 是常见操作（新增对局后全盘重算），
    不去重会把同一局算两次、人为压低区间。"""
    path = Path(path)
    existing = {game_key(row) for row in load_pool(path)}
    fresh = [row for row in rows if game_key(row) not in existing]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in fresh:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(fresh)


def ingest_jsonl(pool_path: str | Path, jsonl_paths: list[str | Path]) -> int:
    rows: list[dict] = []
    for path in jsonl_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(normalize_row(json.loads(line)))
    return append_games(pool_path, rows)


def summarize(rows: list[dict]) -> dict:
    logs = np.array([score_to_log(row["score"]) for row in rows], dtype=np.float64)
    return {
        "L": float(logs.mean()) if logs.size else float("nan"),
        "sigma": float(logs.std(ddof=1)) if logs.size > 1 else 0.0,
        "n": int(logs.size),
    }


def tile_reach_per_10k(rows: list[dict], exponents: tuple[int, ...] = (11, 12, 13, 14, 15)) -> dict:
    """每万局到达某块的次数：不同样本量的模型才可比，直接写次数是最常见的误读来源。"""
    count = len(rows)
    if count == 0:
        return {}
    return {
        str(1 << exponent): round(10_000.0 * sum(1 for row in rows if row["max_exp"] >= exponent) / count, 1)
        for exponent in exponents
    }


def load_anchors(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"schema_version": ANCHORS_SCHEMA_VERSION, "tracks": {}}
    anchors = json.loads(path.read_text(encoding="utf-8"))
    if int(anchors.get("schema_version", 0)) != ANCHORS_SCHEMA_VERSION:
        raise ValueError(
            f"锚点文件 {path} 的 schema_version={anchors.get('schema_version')}，"
            f"本实现要求 {ANCHORS_SCHEMA_VERSION}（旧版锚点没有记录协议本体，"
            "无法判断它固定的尺子是什么，只能重定锚点）"
        )
    return anchors


def save_anchors(path: str | Path, rows: list[dict], spec: dict[str, str]) -> dict:
    """按 track 指定锚点模型并落盘。锚点必须在同一 track 的同一协议下评测过。

    锚点就是“尺子”：它固定下来的协议是往后所有对比的量纲。所以这里要求锚点的
    所有记录协议完全一致——一个锚点混了两个协议，等于一把尺子有两个刻度。
    """
    anchors = {"schema_version": ANCHORS_SCHEMA_VERSION, "tracks": {}}
    for track, model_id in spec.items():
        matched = [row for row in rows if row["track"] == track and row["model_id"] == model_id]
        if not matched:
            raise ValueError(f"锚点 {model_id} 在 track {track} 下没有评测记录")
        protocols = {json.dumps(row.get("protocol"), sort_keys=True, ensure_ascii=False) for row in matched}
        if len(protocols) != 1:
            raise ValueError(
                f"锚点 {model_id} 在 track {track} 下的记录跨了 {len(protocols)} 个协议，"
                "锚点必须唯一确定一把尺子"
            )
        stats = summarize(matched)
        anchors["tracks"][track] = {
            "model_id": model_id,
            "protocol_hash": matched[0]["protocol_hash"],
            "protocol": matched[0].get("protocol"),
            "L_ref": stats["L"],
            "sigma_ref": stats["sigma"],
            "n_ref": stats["n"],
        }
    Path(path).write_text(json.dumps(anchors, ensure_ascii=False, indent=2))
    return anchors


def compute_ratings(rows: list[dict], anchors: dict, lam: float = RATING_LAMBDA) -> dict[str, list[dict]]:
    """按 track 归组、组内再按协议分组重算全部评分，返回 `{track: [协议组, ...]}`。

    **一个 track 可以有多组**，而且通常就是如此：曲线上的每个模拟数都是不同的协议。
    分组不是判“能不能比”，而是判“比的是同一件事”——所有组都放在该 track 锚点的
    尺子上（这正是“模拟数 → 评分曲线”的做法），但每组都会带上它与锚点协议真正不同的
    字段（`differs_from_anchor`），让“这条线是在什么测量条件下得到”变成可读的事实。

    组内按 (模拟数, 协议哈希) 排序：模拟数是曲线的横轴，按它排出来的表格才像曲线。
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["track"], row["protocol_hash"]), []).append(row)

    result: dict[str, list[dict]] = {}
    for (track, proto_hash), group in sorted(groups.items()):
        anchor = anchors.get("tracks", {}).get(track)
        if anchor is None:
            continue
        protocol = group[0].get("protocol")
        by_model: dict[str, list[dict]] = {}
        for row in group:
            by_model.setdefault(row["model_id"], []).append(row)
        models: dict[str, dict] = {}
        for model_id, model_rows in sorted(by_model.items()):
            stats = summarize(model_rows)
            rating = RATING_CENTER + lam * (stats["L"] - anchor["L_ref"])
            ci = (
                Z95
                * lam
                * math.sqrt(
                    stats["sigma"] ** 2 / max(1, stats["n"])
                    + anchor["sigma_ref"] ** 2 / max(1, anchor["n_ref"])
                )
            )
            models[model_id] = {
                **stats,
                "label": model_rows[0].get("label", model_id),
                "rating": rating,
                "ci95": ci,
                "mean_score": float(np.mean([row["score"] for row in model_rows])),
                "median_score": float(np.median([row["score"] for row in model_rows])),
                "mean_steps": float(np.mean([row["steps"] for row in model_rows])),
                "source_tracks": sorted({row["source_track"] for row in model_rows}),
                "tile_reach_per_10k": tile_reach_per_10k(model_rows),
            }
        result.setdefault(track, []).append(
            {
                "protocol_hash": proto_hash,
                "protocol": protocol,
                "differs_from_anchor": protocol_diff(anchor.get("protocol"), protocol or {})
                if anchor.get("protocol")
                else {},
                "anchor": anchor,
                "models": models,
            }
        )
    for blocks in result.values():
        blocks.sort(
            key=lambda block: ((block["protocol"] or {}).get("simulations") or 0, block["protocol_hash"])
        )
    return result


def pairing_diagnostic(rows: list[dict], model_a: str, model_b: str, track: str) -> dict:
    """量化“同 seed 配对”到底省了多少方差。

    做法：取两个模型在**同一 eval_seed** 下的逐局 log2 得分，按局号对齐，比较
    (a) 配对差分的标准误与 (b) 独立两样本差分的标准误。两者接近就说明配对没有起到
    方差缩减作用——这正是“同 seed 只对齐开局”的直接后果。同时给出相关系数。
    """
    by_model: dict[str, dict[int, float]] = {}
    for row in rows:
        if row["track"] != track or row["model_id"] not in (model_a, model_b):
            continue
        by_model.setdefault(row["model_id"], {})[row["episode_id"]] = score_to_log(row["score"])
    if model_a not in by_model or model_b not in by_model:
        raise ValueError("两个模型都必须在该 track 下有记录")
    shared = sorted(set(by_model[model_a]) & set(by_model[model_b]))
    if len(shared) < 2:
        raise ValueError("没有足够的共同局号：两个模型必须在同一 eval_seed 下评测过")

    a = np.array([by_model[model_a][episode] for episode in shared])
    b = np.array([by_model[model_b][episode] for episode in shared])
    difference = b - a
    n = len(shared)
    paired_se = float(difference.std(ddof=1) / math.sqrt(n))
    independent_se = float(math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n))
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    paired_mean = float(difference.mean())
    independent_mean = float(b.mean() - a.mean())

    def _block(mean: float, se: float) -> dict:
        # 同时给出 log2 单位与评分点单位：前者是统计量，后者是本项目的展示标度。
        return {
            "mean_log2_difference": mean,
            "se": se,
            "ci95_log2": [mean - Z95 * se, mean + Z95 * se],
            "rating_difference": RATING_LAMBDA * mean,
            "rating_ci95": [RATING_LAMBDA * (mean - Z95 * se), RATING_LAMBDA * (mean + Z95 * se)],
        }

    return {
        "track": track,
        "model_a": model_a,
        "model_b": model_b,
        "games": n,
        "paired": _block(paired_mean, paired_se),
        "independent": _block(independent_mean, independent_se),
        "log2_correlation": correlation,
        "variance_reduction": 1.0 - (paired_se / independent_se) ** 2 if independent_se > 0 else 0.0,
    }


def count_openings() -> dict:
    """开局到底有多少种：实测数字，用来判断“同 seed 对齐开局”还剩多少信息。

    标准 2048 开局：两次生成，第一次在 16 个空格里选一格，第二次在剩下 15 格里选一格，
    数值各自 90% 出 2、10% 出 4。棋盘层面（与生成顺序无关）就是 16 选 2 个格子 ×
    4 种数值组合。再按 4×4 的 D4 对称群（4 旋转 × 2 镜像）简并，剩下的才是真正不同的开局类别。
    """
    import itertools

    boards: set[tuple[int, ...]] = set()
    for first, second in itertools.permutations(range(16), 2):
        for value in (1, 2):
            for other in (1, 2):
                cells = [0] * 16
                cells[first] = value
                cells[second] = other
                boards.add(tuple(cells))

    def transform(board: tuple[int, ...], rotation: int, reflection: int) -> tuple[int, ...]:
        shaped = np.array(board, dtype=np.uint8).reshape(4, 4)
        rotated = np.rot90(shaped, rotation)
        if reflection:
            rotated = np.fliplr(rotated)
        return tuple(int(cell) for cell in rotated.reshape(16))

    orbits = {
        min(transform(board, rotation, reflection) for rotation in range(4) for reflection in range(2))
        for board in boards
    }
    return {"distinct_boards": len(boards), "d4_orbits": len(orbits)}
