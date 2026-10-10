"""一次迭代 = 跑谱 → 训练 → 发布 → 评测 → 评分。

**薄驱动**：五步各自调用独立入口（跑谱、训练合并、发布、评测与评分），
任务身份与请求在执行前落盘，保证中断恢复复用同一产物。这样任何一步都能用原来的
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
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from r2048 import paths
from r2048.config import (
    DATASETS_SECTION,
    load_config,
    parse_datasets,
)
from r2048.fileio import read_json, write_json_atomic
from r2048.identity import allocate_id, file_lock
from r2048.identity import batch_id as allocate_batch_id
from r2048.jobs import JobDir, run_cpp
from r2048.nn.lineage import file_sha256
from r2048.nn.training import DatasetSpec
from r2048.runtime.selfplay import SelfplayConfig, run_selfplay
from r2048.task_control import check_stop
from r2048.workflows.publish import publish_model

# 步骤的规范顺序。`--steps` 只能从中挑选，不能改顺序：后一步的输入是前一步的产物。
STEPS = ("selfplay", "train", "publish", "evaluate", "rate")
# 默认基线臂：评分池的锚点就是这两个（ADR 0018 之后用新格式重跑过的那两条）。
DEFAULT_BASELINES = ("random", "mcts64")


@dataclass
class IterateConfig:
    config_path: str  # 训练 YAML（ADR 0021）：datasets + train
    name: str = ""
    output: str = ""  # 迭代目录（默认 <输出根>/iters/<name>）
    resume: str = ""
    steps: tuple[str, ...] | None = None
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
    lanes: int | None = None
    in_flight: int | None = None
    search_threads: int = 1
    # 跑谱的并行度：>1 时起 N 个进程写同一批（每个 worker 单线程，数据可复现，ADR 0031）。
    # 样本目标是整批的，所以进程数可以按机器随便改，不需要分摊配额（ADR 0029 的订正）。
    data_workers: int | None = None
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
    max_batch: int | None = None
    performance_profile: str = "auto"
    performance: dict = field(default_factory=dict)
    eval_games: int = 400
    eval_simulations: int = 0  # 独立评测预算；0 沿用跑谱预算，兼容既有 CLI 配置。
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
    if config.resume:
        directory = Path(config.resume).resolve()
        stored = read_json(directory / "iteration.json")
        for key in ("name", "config_path", "output", "init_checkpoint"):
            requested = getattr(config, key)
            before = stored.get(key)
            if key != "name" and requested and before:
                requested, before = str(Path(requested).resolve()), str(Path(before).resolve())
            if requested and requested != before:
                raise ValueError(f"续跑的 {key} 与原任务不一致")
        # 续跑只允许改变选择的步骤与 force；训练和搜索参数从首次快照恢复。
        steps, force = config.steps or tuple(stored.get("steps", STEPS)), config.force
        for key in vars(config):
            if key in stored and key not in ("steps", "force", "resume", "on_progress"):
                setattr(config, key, stored[key])
        config.steps, config.force = steps, force
        config.output = str(directory)
    elif not config.name:
        if config.output and (Path(config.output) / "iteration.json").exists():
            raise ValueError("任务目录已经存在，请使用 --resume")
        config.name = f"iter-{allocate_id('iteration')}"
    if not config.resume:
        from r2048.performance import resolve

        effective, config.performance = resolve(
            "selfplay",
            {**config.payload(), "workers": config.data_workers},
            config.init_checkpoint,
            mode=config.performance_profile,
        )
        config.data_workers = effective["workers"]
        for key in ("lanes", "in_flight", "max_batch"):
            setattr(config, key, effective[key])
    if not config.config_path:
        raise ValueError("创建迭代任务必须提供 --config；续跑使用 --resume")
    config.steps = config.steps or STEPS
    config.config_path = str(Path(config.config_path).resolve())
    if config.init_checkpoint:
        config.init_checkpoint = str(Path(config.init_checkpoint).resolve())
    config.output = str(
        (Path(config.output) if config.output else config.iterations_root / config.name).resolve()
    )
    with file_lock(Path(config.output) / ".execution.lock", blocking=False):
        return _run_iteration(config)


def _run_iteration(config: IterateConfig) -> IterateResult:
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
    snapshot_path = directory / "iteration.json"
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        for key in ("config_path", "init_checkpoint", "output"):
            if snapshot.get(key):
                snapshot[key] = str(Path(snapshot[key]).resolve())
        current = json.loads(json.dumps(config.payload()))
        differing = [
            key
            for key, value in snapshot.items()
            if key not in ("steps", "force", "resume") and key in current and current[key] != value
        ]
        if differing:
            raise ValueError(f"既有迭代任务的参数发生变化：{differing}；新实验请创建新任务")
    result = IterateResult(name=config.name, directory=str(directory))

    with JobDir(config.name, "iterate", params=config.payload(), path=directory) as job:
        config.on_progress = job.callback()
        if not snapshot_path.exists():
            write_json_atomic(snapshot_path, config.payload())
        for step in STEPS:
            if step not in config.steps:
                continue
            if step in written and not config.force:
                print(f"跳过 {step}（已完成，用 --force 重做）")
                continue
            if config.force:
                # 在新产物生成之前使下游失效；中断后也不会复用依赖旧上游的评测或发布。
                for downstream in STEPS[STEPS.index(step) :]:
                    written.pop(downstream, None)
                    (directory / "steps" / f"{downstream}.json").unlink(missing_ok=True)
                    (directory / "requests" / f"{downstream}.json").unlink(missing_ok=True)
            required = {
                "train": ("selfplay",),
                "publish": ("train",),
                "evaluate": ("train", "publish"),
                "rate": ("evaluate",),
            }.get(step, ())
            missing = [name for name in required if name not in written]
            if missing:
                raise ValueError(f"{step} 缺少前置步骤：{missing}")
            print(f"[{step}] 开始")
            check_stop()
            produced = _STEP_FUNCS[step](config, written, job)
            if step == "selfplay" and produced["state"] != "complete":
                raise InterruptedError(f"数据批处于 {produced['state']}，保留原身份；使用 --resume 继续")
            written[step] = produced
            write_json_atomic(directory / "steps" / f"{step}.json", produced)
            check_stop()
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

    from r2048.nn.model import load_checkpoint, read_checkpoint_payload

    if any(
        value is None for value in (config.data_workers, config.lanes, config.in_flight, config.max_batch)
    ):
        from r2048.performance import resolve

        effective, config.performance = resolve(
            "selfplay",
            {**config.payload(), "workers": config.data_workers},
            config.init_checkpoint,
            mode=config.performance_profile,
        )
        config.data_workers = effective["workers"]
        for key in ("lanes", "in_flight", "max_batch"):
            setattr(config, key, effective[key])
    directory = Path(config.output)
    request_path = directory / "requests" / "selfplay.json"
    if request_path.exists():
        batch_id = read_json(request_path)["batch_id"]
    else:
        batch_id = allocate_batch_id("nn" if config.init_checkpoint else "mcts", config.simulations)
        write_json_atomic(request_path, {"batch_id": batch_id})
    data_dir = paths.data_dir(batch_id)
    continuing = (data_dir / "manifest.json").exists()
    if continuing:
        summary = read_json(data_dir / "manifest.json")
        if summary["batch_id"] != batch_id:
            raise ValueError("数据 manifest 与任务预留批次身份不一致")
        if summary.get("state") == "complete":
            # 增量 manifest 已提交但步骤标记未提交：补回结果，不能向完整批次追加数据。
            return {
                "batch_id": batch_id,
                "data_dir": str(data_dir),
                "manifest": str(data_dir / "manifest.json"),
                "teacher": f"nn:{summary['model_id']}"
                if config.init_checkpoint
                else f"cpp-mcts{config.simulations}",
                "samples": int(summary["samples"]),
                "episodes": int(summary["episodes"]),
                "state": "complete",
                "seed": config.data_seed,
            }
    if config.init_checkpoint:
        device = torch.device(config.device)
        if config.data_workers > 1:
            metadata = read_checkpoint_payload(config.init_checkpoint)["metadata"]
            model = None
        else:
            model, metadata = load_checkpoint(config.init_checkpoint, device)
        model_id = str(metadata.get("model_id", "")) or Path(config.init_checkpoint).parent.name
        selfplay_config = SelfplayConfig(
            data_dir=str(data_dir),
            batch_id=batch_id,
            resume=continuing,
            control_path=os.environ.get("R2048_TASK_CONTROL", ""),
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
            max_batch=config.max_batch,
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
            "workers": config.data_workers,
            "seed": config.data_seed,
            "out": str(data_dir),
            "batch_id": batch_id,
            "continue": continuing,
            "control": os.environ.get("R2048_TASK_CONTROL", ""),
            "quiet": True,
        },
        job=job,
        progress=True,
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
    from r2048.workflows.dual_training import train_dual

    file_config = load_config(config.config_path)
    datasets = (parse_datasets(file_config[DATASETS_SECTION]) if DATASETS_SECTION in file_config else []) + [
        DatasetSpec(path=written["selfplay"]["data_dir"], weight=config.data_weight)
    ]
    return train_dual(
        Path(config.output),
        config.name,
        config.config_path,
        datasets,
        config.init_checkpoint,
        config.on_progress,
    )


def _step_publish(config: IterateConfig, written: dict, job: JobDir) -> dict:
    checkpoint = written["train"]["model"]
    request_path = Path(config.output) / "requests" / "publish.json"
    root = paths.output_root() / "models"
    digest = file_sha256(checkpoint)
    if request_path.exists():
        request = read_json(request_path)
        if request["source_checkpoint_sha256"] != digest:
            raise ValueError("待发布 checkpoint 已变化，请重做 publish 及下游步骤")
    else:
        request = {"release_id": allocate_id("release", root), "source_checkpoint_sha256": digest}
        write_json_atomic(request_path, request)
    # 目录提交成功、步骤标记未提交就中断时，按预留身份与源摘要恢复，不能重新发号。
    matches = list(root.glob(f"*-{request['release_id']}-s*/manifest.json"))
    if matches:
        if len(matches) != 1:
            raise ValueError("预留发布身份对应多个清单")
        manifest_path = matches[0]
        manifest = read_json(manifest_path)
        model_path = manifest_path.parent / manifest["model_file"]
        if (
            manifest.get("source_checkpoint_sha256") != digest
            or file_sha256(model_path) != manifest["model_sha256"]
        ):
            raise ValueError("已提交发布与预留源模型或文件摘要不一致")
    else:
        result = publish_model(checkpoint, request["release_id"], root, reserved=True)
        model_path, manifest_path = result.model_path, result.manifest_path
        manifest = read_json(manifest_path)
    return {
        "model": str(model_path),
        "manifest": str(manifest_path),
        "release_id": manifest["release_id"],
        "model_id": manifest["model_id"],
        "model_sha256": manifest["model_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
    }


def _step_evaluate(config: IterateConfig, written: dict, job: JobDir) -> dict:
    """评测三条 track + 基线臂，全部用同一套协议参数（`--eval-seed` 对齐开局）。"""
    directory = Path(config.output) if config.output else config.iterations_root / config.name
    train = written["train"]
    if "publish" not in written:
        raise ValueError("完整模型评测需要先完成 publish 步骤")
    train = {**train, "model": written["publish"]["model"]}
    from r2048.workflows.assessment import evaluate_group

    request = directory / "requests" / "evaluate.json"
    if request.exists():
        attempt = read_json(request)["attempt"]
    else:
        attempt = f"eval-{allocate_id('eval')}"
        write_json_atomic(request, {"attempt": attempt})
    params = {
        "games": config.eval_games,
        "seed": config.eval_seed,
        "simulations": config.eval_simulations or config.simulations,
        "lanes": config.lanes,
        "in_flight": config.in_flight,
        "search_threads": config.search_threads,
        "c_puct": config.c_puct,
        "device": config.device,
    }
    result = evaluate_group(
        directory / "eval" / attempt, _nn_arms(config, train), baseline_arms(config), params, job
    )
    return {**result, "protocol": _protocol(config), "directory": str(directory)}


def _nn_arms(config: IterateConfig, train: dict) -> list[tuple[str, str, str]]:
    """这一轮该评哪三条 NN 臂：价值头、策略头（合并模型）、NN-MCTS（合并模型）。"""
    return [
        (f"{config.name}-value", train["value_checkpoint"], "nn_value"),
        (f"{config.name}-policy", train["model"], "nn_policy"),
        (f"{config.name}-nnmcts{config.eval_simulations or config.simulations}", train["model"], "nn_mcts"),
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
        "simulations": config.eval_simulations or config.simulations,
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
    from r2048.workflows.assessment import ingest_and_rate

    labels = written["evaluate"]["jsonl"]
    initialize = {}
    for label in labels:
        if label == "random" or label.endswith("-random"):
            initialize["pure"] = label
        if label.split("-")[-1] == f"mcts{config.eval_simulations or config.simulations}":
            initialize["search"] = label
    return ingest_and_rate(Path(config.output), sorted(labels.values()), config.anchors, initialize)


_STEP_FUNCS = {
    "selfplay": _step_selfplay,
    "train": _step_train,
    "publish": _step_publish,
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
    if step in ("train", "publish"):
        return f"完整模型 {produced['model']}"
    if step == "evaluate":
        return f"{len(produced['arms'])} 条臂 × {produced['games']} 局"
    return f"评分池新增 {produced['ingested']} 局 → {produced['rating_json']}"
