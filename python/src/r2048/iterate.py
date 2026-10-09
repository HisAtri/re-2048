"""一次迭代 = 跑谱 → 训练 → 评测 → 评分。

**薄驱动**：四步各自都是对既有入口的一次调用（`run_selfplay` / `run_training` /
`run_evaluation` / `rating.ingest_jsonl`），不引入任何新机制。这样任何一步都能用原来的
CLI 单独重跑——这一点是刻意的：迭代失败时不该只有"重跑整个迭代"一个选项。

**幂等**：每步做完在迭代目录写 `steps/<步>.json`（记录这一步产出了什么）。重跑时跳过
已完成的步骤；`--steps` 只跑其中几步，`--force` 重做已完成的步骤。标记文件还有第二个
作用：单独重跑 `evaluate` 时，前面几步的产物路径从标记里读出来，不必让人重新敲一遍。

**为什么这套流程要进仓库**：它以前是 /tmp 里的脚本，里面混着三类东西——
批次路径靠 `tail -1` 从 stdout 里抓、评测协议参数散在命令行与注释里、两个半模型的合并
是一段内联 heredoc。它们都不是"人的判断"，而是可以被代码固定下来的约定
（stdout 末行是结果，见 ADR 0008；合并见 `workflows/merge.py`）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from r2048 import paths
from r2048.config import (
    DATASETS_SECTION,
    TRAIN_SECTION,
    load_config,
    parse_datasets,
    train_section_payload,
)
from r2048.fileio import write_json_atomic
from r2048.jobs import JobDir, run_cpp
from r2048.nn.training import DatasetSpec, TrainConfig, run_training
from r2048.runtime.selfplay import SelfplayConfig, run_selfplay
from r2048.workflows import rating as rating_module
from r2048.workflows.evaluate import EvalConfig, run_evaluation
from r2048.workflows.merge import merge_halves

# 步骤的规范顺序。`--steps` 只能从中挑选，不能改顺序：后一步的输入是前一步的产物。
STEPS = ("selfplay", "train", "evaluate", "rate")
# 默认基线臂：评分池的锚点就是这两个（ADR 0018 之后用新格式重跑过的那两条）。
DEFAULT_BASELINES = ("random", "mcts64")


@dataclass
class IterateConfig:
    name: str
    config_path: str  # 训练 YAML（ADR 0021）：datasets + train
    output: str  # 迭代目录（默认 <输出根>/iters/<name>）
    steps: tuple[str, ...] = STEPS
    force: bool = False
    # 上一轮的完整模型：跑谱用它当对手（NN 自对弈），训练用它当起点。
    # 不给则用纯 MCTS 当教师（首轮引导，见 ADR 0013 里的 c1）。
    init_checkpoint: str = ""
    samples: int = 0  # 跑谱目标样本数
    data_seed: int = 0  # 跑谱用的种子（与 --eval-seed 分开：改评测不该换掉训练数据）
    data_weight: float = 1.0  # 新数据混进训练池的权重
    # 协议参数：跑谱、NN 评测与基线评测**共用同一套**（统一注入并记录在 iteration.json 里）。
    # 分开配会让"这条曲线是在什么条件下得到的"重新变成注释里的口头约定。
    simulations: int = 64
    lanes: int = 64
    in_flight: int = 8
    search_threads: int = 1
    # 跑谱的并行度：>1 时起 N 个进程写同一批（每个 worker 单线程，数据可复现，ADR 0031）。
    # 样本目标是整批的，所以进程数可以按机器随便改，不需要分摊配额（ADR 0029 的订正）。
    data_workers: int = 4
    # 纯 C++ 教师的 rollout 续弈策略（uniform / greedy）。默认 greedy：它决定候选价值
    # 标签的同根区分度，而模型强度跟着区分度走（ADR 0030 的实测）。它进数据批的来源记录，
    # 所以改了它就是换了教师。
    rollout_policy: str = "greedy"
    # 记录里放几个树内节点。它是**格式参数**（决定记录字节数），进清单与续跑闸门；采集本身
    # 不改变任何一局的走法（ADR 0041）。默认打开的理由见 SelfplayConfig（ADR 0042 的订正）。
    tree_nodes: int = 2
    # 树内节点的独立标注（ADR 0043）：见 SelfplayConfig。
    tree_label_plays: int = 8
    tree_label_policy: str = "top_visits"
    tree_label_rollout: str = "greedy"
    c_puct: float = 1.5
    device: str = "cuda"
    eval_games: int = 400
    eval_seed: int = 0
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    anchors: dict[str, str] = field(default_factory=dict)
    on_progress: Callable[[dict], None] | None = None

    @property
    def iterations_root(self) -> Path:
        return paths.output_root() / "iters"

    def payload(self) -> dict:
        """写进 job.json / iteration.json 的参数快照（不含回调）。"""
        data = {name: value for name, value in vars(self).items() if name != "on_progress"}
        data["iterations_root"] = str(self.iterations_root)
        return data


@dataclass
class IterateResult:
    name: str
    directory: str
    steps: dict[str, dict] = field(default_factory=dict)
    rating_json: str = ""
    pool: str = ""
    seconds: float = 0.0


def run_iteration(config: IterateConfig) -> IterateResult:
    unknown = [step for step in config.steps if step not in STEPS]
    if unknown:
        raise ValueError(f"未知步骤 {unknown}，可选 {list(STEPS)}")
    if "selfplay" in config.steps and config.samples <= 0:
        # 迭代的跑谱必须有目标：一次迭代要在"这一轮产出多少数据"上闭合，
        # 不限量是给人守着终端用的（ADR 0020），不是流水线的形态。
        raise ValueError("迭代里的跑谱必须给出 --samples（> 0）：迭代要在数据量上闭合")
    directory = Path(config.output) if config.output else config.iterations_root / config.name
    started = time.monotonic()
    written = _load_steps(directory)
    result = IterateResult(name=config.name, directory=str(directory))

    with JobDir(config.name, "iterate", params=config.payload(), path=directory) as job:
        config.on_progress = job.callback()
        write_json_atomic(directory / "iteration.json", config.payload())
        for step in STEPS:
            if step not in config.steps:
                continue
            if step in written and not config.force:
                print(f"跳过 {step}（已完成，用 --force 重做）")
                continue
            print(f"[{step}] 开始")
            produced = _STEP_FUNCS[step](config, written, job)
            written[step] = produced
            write_json_atomic(directory / "steps" / f"{step}.json", produced)
            print(f"[{step}] 完成：{_summarize(step, produced)}")
        result.steps = {step: produced for step, produced in written.items() if step in config.steps}
        rating_json = written.get("rate", {}).get("rating_json", "")
        pool = written.get("rate", {}).get("pool", "")
        job.finish(steps=sorted(result.steps), rating_json=rating_json, pool=pool)

    result.rating_json = rating_json
    result.pool = pool
    result.seconds = time.monotonic() - started
    print(f"迭代 {config.name} 结束：{result.seconds:.1f} 秒，目录 {directory}")
    return result


# --- 各步 -----------------------------------------------------------------------


def _step_selfplay(config: IterateConfig, written: dict, job: JobDir) -> dict:
    """产出这一轮的数据批：有上一轮模型就 NN 自对弈，没有就纯 MCTS 当教师。"""
    import torch

    from r2048.nn.model import load_checkpoint

    batch_id = f"{config.name}-s{config.simulations}-{time.strftime('%Y%m%d-%H%M%S')}"
    data_dir = paths.data_dir(batch_id)
    if config.init_checkpoint:
        device = torch.device(config.device)
        model, metadata = load_checkpoint(config.init_checkpoint, device)
        model_id = str(metadata.get("model_id", "")) or Path(config.init_checkpoint).parent.name
        selfplay_config = SelfplayConfig(
            data_dir=str(data_dir),
            batch_id=batch_id,
            samples=config.samples,
            seed=config.data_seed,
            model_id=model_id,
            simulations=config.simulations,
            tree_nodes=config.tree_nodes,
            tree_label_plays=config.tree_label_plays,
            tree_label_policy=config.tree_label_policy,
            tree_label_rollout=config.tree_label_rollout,
            lanes=config.lanes,
            in_flight_per_lane=config.in_flight,
            search_threads=config.search_threads,
            c_puct=config.c_puct,
            # 接上进度回调：相位汇总（谁在等谁、GPU 忙占比）要落到作业日志里，
            # 而不是只打在终端上——跑完一轮还想回看时，终端已经滚掉了。
            on_progress=config.on_progress,
        )
        if config.data_workers > 1:
            # 并行跑谱走**多进程**：每个 worker 单线程，数据因此可复现（ADR 0031），
            # 而样本目标按整批判定，所以进程数随便改、不用分摊配额（ADR 0029 的订正）。
            from r2048.runtime.worker_pool import run_selfplay_workers

            workers_result = run_selfplay_workers(
                selfplay_config,
                config.init_checkpoint,
                config.device,
                config.data_workers,
                on_progress=config.on_progress,
            )
            manifest_path = workers_result.manifest_path
            samples, episodes, state = (
                workers_result.samples,
                workers_result.episodes,
                workers_result.state,
            )
            samples_per_second = workers_result.samples_per_second
        else:
            result = run_selfplay(selfplay_config, model, device)
            manifest_path = result.manifest_path
            samples, episodes, state = result.samples, result.episodes, result.state
            samples_per_second = result.samples_per_second
        return {
            "batch_id": batch_id,
            "data_dir": str(data_dir),
            "manifest": manifest_path,
            "teacher": f"nn:{model_id}",
            "samples": samples,
            "episodes": episodes,
            "state": state,
            "seed": config.data_seed,
            "data_workers": config.data_workers,
            "samples_per_second": round(samples_per_second, 1),
        }

    # 首轮：纯 C++ MCTS 教师。走 `generate` 而不是 Python 自对弈——这一轮没有模型，
    # 而纯 C++ 线是整数运算、几秒钟就能跑完的（ADR 0020 的停止语义对它同样成立）。
    #
    # 续弈策略默认 greedy：实测它决定候选价值标签的同根区分度（同种子同模拟数下
    # 同根标签 std 55.5 → 186.2），而模型强度跟着这个区分度走（NN-MCTS-64 8,306 → 13,156），
    # 教师棋力本身不解释差异（greedy-64 的棋力 22,357 与 uniform-256 的 22,863 相当，
    # 但后者训出的模型只有 6,388）。见 ADR 0030。
    command = run_cpp(
        "generate",
        {
            "policy": "mcts",
            "simulations": config.simulations,
            "rollout_policy": config.rollout_policy,
            "samples": config.samples,
            "seed": config.data_seed,
            "out": str(data_dir),
            "quiet": True,
        },
        job=job,
    )
    # 批次路径从 stdout 末行取（ADR 0008：stdout 只放可被脚本消费的结果）——
    # `generate` 的末行是 manifest.json 的路径，所以批次目录是它的父目录。
    manifest_path = Path(command.last_line)
    batch_dir = manifest_path.parent
    if not manifest_path.exists():
        raise RuntimeError(f"跑谱没有产出清单：stdout 末行是 {command.last_line!r}")
    summary = json.loads(manifest_path.read_text())
    return {
        # 批次号以清单里的为准：目录名是我们起的，清单才是写出方的记录（`generate` 自己
        # 决定 batch_id，从目录名反推会得到两个不同的说法）。
        "batch_id": str(summary["batch_id"]),
        "data_dir": str(batch_dir),
        "manifest": str(manifest_path),
        "teacher": f"cpp-mcts{config.simulations}",
        "samples": int(summary["samples"]),
        "episodes": int(summary["episodes"]),
        "state": str(summary.get("state", "complete")),
        "seed": config.data_seed,
    }


def _step_train(config: IterateConfig, written: dict, job: JobDir) -> dict:
    """价值 run → 策略 run（冻结价值）→ 合并成这一轮的完整模型。

    这套配方是 ADR 0010/0013 的结论：价值头与策略头分开训（共享主干一起优化会把价值头
    训坏），策略头在冻结价值上训。合并见 `workflows/merge.py`——两个 run 各自只在一个头上
    是新的，评测必须用"这一轮的价值 + 这一轮的策略"。
    """
    batch = written["selfplay"]
    directory = Path(config.output) if config.output else config.iterations_root / config.name
    file_config = load_config(config.config_path)
    train_params: dict[str, Any] = train_section_payload(
        config.config_path, file_config.get(TRAIN_SECTION, {})
    )
    # YAML 里的 datasets 是"已有的池子"（多来源，ADR 0021），这一轮新产出的数据按权重
    # 追加进去。YAML 里可以一个来源都不写——首轮就是这么用的：只有这一轮的数据。
    datasets = (parse_datasets(file_config[DATASETS_SECTION]) if DATASETS_SECTION in file_config else []) + [
        DatasetSpec(path=batch["data_dir"], weight=config.data_weight)
    ]
    train_params["datasets"] = datasets
    if config.on_progress is not None:
        train_params["on_progress"] = config.on_progress
    seed = int(train_params.get("seed", 0))

    value_run = Path(directory) / "runs" / "value"
    value_config = TrainConfig(
        output=str(value_run),
        init_checkpoint=config.init_checkpoint,
        mode="value",
        seed=seed,
        **{
            key: value
            for key, value in train_params.items()
            if key not in {"output", "mode", "seed", "init_checkpoint", "freeze_value"}
        },
    )
    value = run_training(value_config)

    # 策略 run 的起点：有上一轮模型就从它的策略路径继续，否则用**刚训好的价值模型**
    # （首轮没有别的东西可用；它的策略头是随机初始化的，正好就是"要训的那部分"）。
    policy_head_run = Path(directory) / "runs" / "policy-head"
    policy_config = TrainConfig(
        output=str(policy_head_run),
        init_checkpoint=config.init_checkpoint or value.best_checkpoint,
        mode="policy",
        freeze_value=True,
        # 两个 run 用相邻的种子：它们必须是不同的采样/增强序列，而把两个种子都塞进 CLI
        # 只会让 YAML 里的 seed 失去意义。
        seed=seed + 1,
        **{
            key: value
            for key, value in train_params.items()
            if key not in {"output", "mode", "seed", "freeze_value", "init_checkpoint"}
        },
    )
    policy = run_training(policy_config)

    merged = merge_halves(
        value.best_checkpoint,
        policy.best_checkpoint,
        Path(directory) / "runs" / "policy" / "model.pt",
        note=f"{config.name}：价值来自 value run（{seed}），策略来自 policy-head run（{seed + 1}）",
    )
    return {
        "value_checkpoint": value.best_checkpoint,
        "policy_head_checkpoint": policy.best_checkpoint,
        "model": merged.path,
        "model_id": merged.model_id,
        "seed": seed,
        "data": [spec.path for spec in datasets],
    }


def _step_evaluate(config: IterateConfig, written: dict, job: JobDir) -> dict:
    """评测三条 track + 基线臂，全部用同一套协议参数（`--eval-seed` 对齐开局）。"""
    directory = Path(config.output) if config.output else config.iterations_root / config.name
    train = written["train"]
    arms = _nn_arms(config, train)
    jsonls: dict[str, str] = {}
    eval_dir = directory / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for label, checkpoint, track in arms:
        # 逐局记录落在迭代目录里（不放全局 eval/）：一次迭代的产物应当自包含，
        # 否则第二次迭代会覆盖第一次的基线记录，而"事后回看那次迭代评了什么"就没了。
        jsonl = eval_dir / f"{label}.jsonl"
        result = run_evaluation(
            EvalConfig(
                games=config.eval_games,
                seed=config.eval_seed,
                checkpoint=checkpoint,
                track=track,
                simulations=config.simulations,
                lanes=config.lanes,
                in_flight=config.in_flight,
                search_threads=config.search_threads,
                c_puct=config.c_puct,
                device=config.device,
                label=label,
                jsonl=str(jsonl),
                on_progress=config.on_progress,
            )
        )
        jsonls[label] = result.jsonl
    for label, params in baseline_arms(config):
        jsonl = eval_dir / f"{label}.jsonl"
        run_cpp(
            "evaluate",
            {
                **params,
                "games": config.eval_games,
                "seed": config.eval_seed,
                "label": label,
                "jsonl": str(jsonl),
                "quiet": True,
            },
            job=job,
        )
        jsonls[label] = str(jsonl)

    return {
        "arms": [label for label, _, _ in arms] + [label for label, _ in baseline_arms(config)],
        "protocol": _protocol(config),
        "jsonl": jsonls,
        "games": config.eval_games,
        "seed": config.eval_seed,
        "directory": str(directory),
    }


def _nn_arms(config: IterateConfig, train: dict) -> list[tuple[str, str, str]]:
    """这一轮该评哪三条 NN 臂：价值头、策略头（合并模型）、NN-MCTS（合并模型）。"""
    return [
        (f"{config.name}-value", train["value_checkpoint"], "nn_value"),
        (f"{config.name}-policy", train["model"], "nn_policy"),
        (f"{config.name}-nnmcts{config.simulations}", train["model"], "nn_mcts"),
    ]


def baseline_arms(config: IterateConfig) -> list[tuple[str, dict]]:
    """基线臂：`random` / `mcts<模拟数>`，走纯 C++ 评测（几秒到几十秒）。

    它们同时是评分池的锚点数据来源：锚点必须每次迭代都在同一协议下重新评测，
    否则池子里的"尺子"会停留在很久以前的那次运行上。
    """
    arms = []
    for spec in config.baselines:
        if spec == "random":
            arms.append(("random", {"policy": "random"}))
            continue
        if spec.startswith("mcts") and spec[4:].isdigit():
            arms.append((spec, {"policy": "mcts", "simulations": int(spec[4:])}))
            continue
        raise ValueError(f"未知基线 {spec!r}：可选 'random' 或 'mcts<模拟数>'（例如 mcts64）")
    return arms


def _protocol(config: IterateConfig) -> dict:
    """统一注入的评测协议：记下来，否则"这条曲线是在什么条件下得到的"只能靠翻命令行。"""
    return {
        "simulations": config.simulations,
        "lanes": config.lanes,
        "in_flight_per_lane": config.in_flight,
        "search_threads": config.search_threads,
        "c_puct": config.c_puct,
        "device": config.device,
        "games": config.eval_games,
        "seed": config.eval_seed,
        "baselines": list(config.baselines),
    }


def _step_rate(config: IterateConfig, written: dict, job: JobDir) -> dict:
    """把这一步的评测追加进评分池，必要时刷新锚点，并把评分写进迭代目录。"""
    directory = Path(config.output) if config.output else config.iterations_root / config.name
    pool = paths.output_root() / "rating" / "pool.jsonl"
    anchors_path = paths.output_root() / "rating" / "anchors.json"
    jsonls = sorted(written["evaluate"]["jsonl"].values())
    added = rating_module.ingest_jsonl(pool, jsonls)
    rows = rating_module.load_pool(pool)
    if config.anchors:
        rating_module.save_anchors(anchors_path, rows, config.anchors)
    anchors = rating_module.load_anchors(anchors_path)
    ratings = rating_module.compute_ratings(rows, anchors)
    report = {
        "name": config.name,
        "protocol": written["evaluate"]["protocol"],
        "ingested": added,
        "pool": str(pool),
        "anchors": config.anchors,
        "tracks": {
            track: [
                {
                    "protocol_hash": block["protocol_hash"],
                    "anchor": block["anchor"]["model_id"],
                    "models": block["models"],
                }
                for block in blocks
            ]
            for track, blocks in ratings.items()
        },
    }
    rating_json = directory / "rating.json"
    write_json_atomic(rating_json, report)
    return {
        "pool": str(pool),
        "anchors": str(anchors_path) if config.anchors else "",
        "ingested": added,
        "rating_json": str(rating_json),
        "pool_rows": len(rows),
    }


_STEP_FUNCS = {
    "selfplay": _step_selfplay,
    "train": _step_train,
    "evaluate": _step_evaluate,
    "rate": _step_rate,
}


# --- 标记文件 -------------------------------------------------------------------


def _load_steps(directory: Path) -> dict[str, dict]:
    written: dict[str, dict] = {}
    for step in STEPS:
        marker = directory / "steps" / f"{step}.json"
        if marker.exists():
            written[step] = json.loads(marker.read_text())
    return written


def _summarize(step: str, produced: dict) -> str:
    if step == "selfplay":
        return f"{produced['teacher']} 产出 {produced['samples']} 条 / {produced['episodes']} 局 → {produced['data_dir']}"
    if step == "train":
        return f"完整模型 {produced['model']}"
    if step == "evaluate":
        return f"{len(produced['arms'])} 条臂 × {produced['games']} 局"
    return f"评分池新增 {produced['ingested']} 局 → {produced['rating_json']}"
