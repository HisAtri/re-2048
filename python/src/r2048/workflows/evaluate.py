"""评测：三条 track 跑固定局数，并用配对 bootstrap 做模型间比较。

口径（每条都对应一次真实的误判）：

* **三条 track 不可互相解释为同一策略的成绩**：`nn_value` / `nn_policy` 是单次前向的纯网络
  策略，`nn_mcts` 是网络引导的搜索。把纯网络成绩当成“搜索的成绩”或反过来，都会得出错的结论。
* **同 seed 对齐开局**：局号 → 开局棋盘的映射与 C++ 评测完全一致，因此任意两个 track
  的成绩**按局配对**，比较用配对差而不是两个独立区间是否重叠。
* **必须用配对检验**：两个各自对基线的区间重叠不能说明模型之间没有差别；旧版的架构对照里
  离线指标几乎完全重合（MAE 65.5 vs 63.9），差别只在同 seed 对局里显现（+5,367 分）。
* **纯网络 track 需要对应的头**：策略头只在策略/联合训练里被训练过，拿一个只训过价值的
  checkpoint 跑 `nn_policy` 得到的只是随机先验。元数据里的 `training_target` 就是这道闸门。
* **逐局记录必须自证协议**：每行平铺声明全部协议变量（`r2048.protocol` 的清单），
  记录的是**实测生效值**（`lanes` 被局数夹过、`device` 可能回退到 cpu）。
  评分池按行 ingest，所以协议不能放在文件头；缺字段会被拒收，而不是用默认值补齐。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import r2048_core
import torch

from r2048.jobs import finish_console, progress_payload
from r2048.nn.model import load_checkpoint
from r2048.protocol import build_protocol, protocol_hash
from r2048.runtime.inference import BatchedEvaluator

TRACKS = {
    "nn_value": r2048_core.DECISION_VALUE_GREEDY,
    "nn_policy": r2048_core.DECISION_POLICY_GREEDY,
    "nn_mcts": r2048_core.DECISION_SEARCH,
}
POLICY_TRAINED_TARGETS = {"policy", "joint", "dual"}


@dataclass
class EvalConfig:
    games: int
    seed: int
    checkpoint: str = ""
    track: str = "nn_mcts"
    simulations: int = 64
    lanes: int = 32
    in_flight: int = 4
    c_puct: float = 1.5
    cache_entries: int = 1 << 17
    max_batch: int = 8192
    device: str = "cuda"
    label: str = ""
    jsonl: str = ""
    progress_every: float = 2.0
    # 结构化进度回调（见 r2048.jobs）：评测可能跑几十分钟，外部观察者不该只能解析
    # 那一行覆盖式打印。给了回调就不再打印，由调用方决定怎么呈现。
    on_progress: Callable[[dict], None] | None = None


@dataclass
class GameResult:
    episode: int
    score: int
    max_exponent: int
    steps: int


@dataclass
class EvalResult:
    label: str
    track: str
    games: list[GameResult] = field(default_factory=list)
    seconds: float = 0.0
    model_id: str = ""
    simulations: int = 0
    jsonl: str = ""
    protocol: dict = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return float(np.mean([game.score for game in self.games])) if self.games else float("nan")

    @property
    def median_score(self) -> float:
        return float(np.median([game.score for game in self.games])) if self.games else float("nan")

    def tile_counts(self) -> dict[int, int]:
        """达到各级块的局数（至少达到）。11 = 2048，12 = 4096。"""
        counts: dict[int, int] = {}
        if not self.games:
            return counts
        best = max(game.max_exponent for game in self.games)
        for exponent in range(11, best + 1):
            counts[1 << exponent] = sum(1 for game in self.games if game.max_exponent >= exponent)
        return counts

    def summary(self) -> dict:
        scores = np.array([game.score for game in self.games], dtype=np.float64)
        return {
            "label": self.label,
            "track": self.track,
            "model_id": self.model_id,
            "simulations": self.simulations,
            "protocol": self.protocol,
            "protocol_hash": protocol_hash(self.protocol) if self.protocol else "",
            "games": len(self.games),
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "p10": float(np.percentile(scores, 10)) if scores.size else float("nan"),
            "p90": float(np.percentile(scores, 90)) if scores.size else float("nan"),
            "mean_steps": float(np.mean([game.steps for game in self.games])) if self.games else 0.0,
            "tiles": {str(size): count for size, count in self.tile_counts().items()},
            "seconds": self.seconds,
            "games_per_second": len(self.games) / self.seconds if self.seconds > 0 else 0.0,
            "jsonl": self.jsonl,
        }


def build_eval_protocol(config: EvalConfig, lanes: int, device: torch.device, model) -> dict:
    """本次评测的协议变量（见 `r2048.protocol`）。

    记录的是**实测生效值**，不是请求值：`lanes` 会被局数夹住、`device` 会因 cuda
    不可用回退到 cpu。记请求值等于让协议与事实不符，而协议的全部价值就在于它属实。
    """
    search = config.track == "nn_mcts"
    precision = None
    if model is not None:
        precision = str(next(model.parameters()).dtype).removeprefix("torch.")
    return build_protocol(
        {
            "track": config.track,
            "rules_version": int(r2048_core.RULES_VERSION),
            "protocol_version": int(r2048_core.PROTOCOL_VERSION),
            # 纯网络 track 只做一次前向，没有搜索预算与探索系数，声明 null：
            # 旧实现把它们强行记成 0，等于把“没有这个变量”说成“这个变量的值是 0”。
            "simulations": config.simulations if search else None,
            "c_puct": config.c_puct if search else None,
            "temperature": 0.0 if search else None,
            "temperature_moves": 0 if search else None,
            "inference_precision": precision,
            "lanes": lanes,
            "in_flight": config.in_flight,
            "max_batch": config.max_batch,
            "cache_entries": config.cache_entries,
            "device": device.type,
        }
    )


def run_evaluation(config: EvalConfig) -> EvalResult:
    if config.track not in TRACKS:
        raise ValueError(f"未知 track: {config.track}（可选 {'/'.join(TRACKS)}）")
    if config.games <= 0:
        raise ValueError("对局数必须大于 0")

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("请求了 cuda 但不可用，回退到 cpu")
        device = torch.device("cpu")

    model = None
    metadata: dict = {}
    model_id = "random-init"
    if config.checkpoint:
        model, metadata = load_checkpoint(config.checkpoint, device)
        model_id = str(metadata.get("model_id", Path(config.checkpoint).parent.name))
        target = str(metadata.get("training_target", ""))
        if config.track == "nn_policy" and target not in POLICY_TRAINED_TARGETS:
            raise ValueError(
                f"checkpoint 的 training_target={target!r} 没有训练过策略头，"
                "用它跑 nn_policy 只会得到随机先验（元数据闸门拦下）"
            )
    elif config.track != "nn_mcts":
        raise ValueError("纯网络 track 需要 --checkpoint（没有网络就没有前向）")

    lanes = max(1, min(config.lanes, config.games))
    protocol = build_eval_protocol(config, lanes, device, model)
    engine_config = {
        "lanes": lanes,
        "simulations": config.simulations if config.track == "nn_mcts" else 1,
        "in_flight_per_lane": config.in_flight,
        "c_puct": config.c_puct,
        "temperature": 0.0,  # 评测必须确定性选招，不引入采样噪声
        "temperature_moves": 0,
        "cache_entries": config.cache_entries,
        "score_scale": float(r2048_core.SCORE_SCALE),
        "seed": config.seed,
        "decision_mode": TRACKS[config.track],
        "collect_samples": 0,  # 评测不写训练数据
        "episodes_target": config.games,
    }
    engine = r2048_core.BatchedEngine(engine_config, "", "", model_id, 0, 0)
    evaluator = BatchedEvaluator(model, device) if model is not None else None

    results: list[GameResult] = []
    started = time.monotonic()
    last_report = started
    while True:
        advance = engine.advance(config.max_batch)
        count = int(advance["request_ids"].size)
        if count:
            values, logits = evaluator.evaluate(advance["request_kinds"], advance["boards"])
            engine.submit(advance["request_ids"], values, logits)
        for episode, steps, tiles, score in zip(
            advance["episode_ids"],
            advance["episode_steps"],
            advance["episode_tiles"],
            advance["episode_scores"],
        ):
            results.append(
                GameResult(episode=int(episode), score=int(score), max_exponent=int(tiles), steps=int(steps))
            )
        now = time.monotonic()
        if now - last_report >= config.progress_every and not advance["done"]:
            last_report = now
            payload = progress_payload(
                "eval",
                len(results),
                config.games,
                "games",
                started,
                line=(
                    f"{config.track} {len(results)}/{config.games} 局，"
                    f"当前均分 {np.mean([g.score for g in results]) if results else 0:.0f}"
                ),
                extra={"track": config.track, "model_id": model_id},
            )
            if config.on_progress is None:
                print("\r" + payload["line"], end="", flush=True)
            else:
                config.on_progress(payload)
        if advance["done"]:
            break
    seconds = time.monotonic() - started
    if config.progress_every > 0 and config.on_progress is None:
        finish_console()

    results.sort(key=lambda game: game.episode)
    result = EvalResult(
        label=config.label or config.track,
        track=config.track,
        games=results,
        seconds=seconds,
        model_id=model_id,
        simulations=int(protocol["simulations"] or 0),
        jsonl=config.jsonl,
        protocol=protocol,
    )
    if config.jsonl:
        path = Path(config.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for game in results:
                # 协议变量平铺在每一行里：逐局记录是唯一能自证“这局怎么测出来的”的东西，
                # 把协议放在文件头会让下游（评分池按行 ingest）读不到它。
                handle.write(
                    json.dumps(
                        {
                            "episode": game.episode,
                            "label": result.label,
                            "model_id": model_id,
                            "seed": config.seed,
                            "score": game.score,
                            "max_exponent": game.max_exponent,
                            "steps": game.steps,
                            **protocol,
                        }
                    )
                    + "\n"
                )
    return result


# ---------------- 配对比较 ----------------


def load_jsonl(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows[int(entry["episode"])] = entry
    return rows


def paired_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, resamples: int = 20000, seed: int = 0
) -> dict:
    """按局配对差的 bootstrap 区间。局是独立样本，因此配对差的标准误直接从局间差异估计。"""
    if baseline.shape != candidate.shape:
        raise ValueError("配对比较要求两组逐局对齐（同 seed、同局数）")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    index = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[index].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    draws = int(np.sum(differences == 0))
    return {
        "games": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "significant": bool(lower > 0 or upper < 0),
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


def compare(
    baseline_path: str | Path,
    candidate_path: str | Path,
    baseline_label: str = "",
    candidate_label: str = "",
    resamples: int = 20000,
    seed: int = 0,
) -> dict:
    baseline_rows = load_jsonl(baseline_path)
    candidate_rows = load_jsonl(candidate_path)
    shared = sorted(set(baseline_rows) & set(candidate_rows))
    if len(shared) < 2:
        raise ValueError("两组记录没有足够的共同局号：配对比较要求同 seed、同局数（不同局的成绩不能配对）")
    baseline = np.array([baseline_rows[episode]["score"] for episode in shared], dtype=np.float64)
    candidate = np.array([candidate_rows[episode]["score"] for episode in shared], dtype=np.float64)
    result = {
        "baseline": baseline_label or str(baseline_rows[shared[0]].get("label", baseline_path)),
        "candidate": candidate_label or str(candidate_rows[shared[0]].get("label", candidate_path)),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "seed": int(baseline_rows[shared[0]].get("seed", 0)),
    }
    result.update(paired_bootstrap(baseline, candidate, resamples=resamples, seed=seed))
    return result
