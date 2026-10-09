"""`python -m r2048 <子命令>`：Python 侧的入口。

只放需要神经网络与 PyTorch 的事情；规则、搜索、纯随机/UCT 跑谱全部在 C++ CLI 里，
两边不重复实现同一件事。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from r2048 import __version__, paths

RATING_LAMBDA = 230.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m r2048", description="2048 自博弈 Python 侧入口")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="打印 Python 侧与 C++ 内核的版本信息")
    sub.add_parser("env", help="打印输出目录等路径解析结果")

    train = sub.add_parser(
        "train",
        help="训练价值头 / 策略头 / 联合 / 双塔",
        description=(
            "训练。默认值取自 TrainConfig（见 nn/training.py），--config 的 train 小节与命令行同名；"
            "优先级 CLI > YAML > 默认值。"
        ),
    )
    # 所有参数都用 SUPPRESS 默认值：只有显式给出的才会出现在 namespace 里，
    # 于是"用户到底指定了什么"是精确可知的，YAML 与 CLI 的合并才谈得上"谁覆盖谁"。
    train.add_argument("--config", default="", help="训练配置文件（YAML：datasets + train）")
    train.add_argument(
        "--data", default="", help="数据批目录（含 manifest.json）；与 --config 的 datasets 二选一"
    )
    train.add_argument("--output", required=True, help="运行目录（写 checkpoint 与 train.json）")
    train.add_argument("--mode", default=argparse.SUPPRESS, choices=["value", "policy", "joint", "dual"])
    train.add_argument("--init-checkpoint", default=argparse.SUPPRESS, help="从已有 checkpoint 初始化")
    train.add_argument("--architecture", default=argparse.SUPPRESS, help="覆盖结构（默认按模式选）")
    train.add_argument(
        "--freeze-value",
        action="store_true",
        default=argparse.SUPPRESS,
        help="只训练策略路径（策略模式的推荐做法）",
    )
    train.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    train.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    train.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    train.add_argument("--lr-schedule", default=argparse.SUPPRESS, choices=["cosine", "constant"])
    train.add_argument("--warmup-steps", type=int, default=argparse.SUPPRESS)
    train.add_argument("--grad-clip", type=float, default=argparse.SUPPRESS)
    train.add_argument("--value-loss", default=argparse.SUPPRESS, choices=["huber", "mse"])
    train.add_argument(
        "--value-target",
        default=argparse.SUPPRESS,
        choices=["auto", "candidate", "trajectory"],
        help="价值回归用候选目标还是主轨迹回报（auto：有候选监督就用候选）",
    )
    train.add_argument("--value-weight", type=float, default=argparse.SUPPRESS)
    train.add_argument(
        "--value-selection",
        default=argparse.SUPPRESS,
        choices=["best_hit", "q_best_hit"],
        help="价值头 checkpoint 的挑选口径：q_best_hit = r+A 口径（默认，与引擎的选择值同口径）；"
        "best_hit = A 口径（旧口径，仅用于复现历史产物）",
    )
    train.add_argument("--policy-weight", type=float, default=argparse.SUPPRESS)
    train.add_argument(
        "--value-ranking-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Q 空间的同根成对排序项权重（0 = 关，默认）；与 Huber 同量纲，见 ADR 0035",
    )
    train.add_argument(
        "--policy-target",
        default=argparse.SUPPRESS,
        choices=["visits", "improved"],
        help="策略目标：visits = 搜索根访问分布（纯蒸馏，默认）；improved = 一步策略改进 π̃∝π̂·exp(Â/β)",
    )
    train.add_argument(
        "--improve-beta",
        type=float,
        default=argparse.SUPPRESS,
        help="改进目标的温度（无量纲，乘在同根 Â 的 std 上；β→∞ 退化为 visits 口径）",
    )
    train.add_argument(
        "--improve-floor",
        type=float,
        default=argparse.SUPPRESS,
        help="改进目标里 π̂ 的均匀伪访问数（让搜索未访问的候选也能拿到质量）",
    )
    train.add_argument("--val-fraction", type=float, default=argparse.SUPPRESS)
    train.add_argument("--split-seed", type=int, default=argparse.SUPPRESS)
    train.add_argument(
        "--samples-per-epoch",
        type=int,
        default=argparse.SUPPRESS,
        help="每轮采样多少条（0 = 各来源训练池之和）",
    )
    train.add_argument("--no-augment", action="store_true", default=argparse.SUPPRESS, help="关闭 D4 增强")
    train.add_argument(
        "--max-samples",
        type=int,
        default=argparse.SUPPRESS,
        help="限制训练样本数（调试用，按全局下标顺序截断）",
    )
    train.add_argument("--device", default=argparse.SUPPRESS)
    train.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    train.add_argument("--amp", default=argparse.SUPPRESS, choices=["off", "bf16"])
    train.add_argument("--no-channels-last", action="store_true", default=argparse.SUPPRESS)
    train.add_argument(
        "--compile",
        action="store_true",
        default=argparse.SUPPRESS,
        help="torch.compile（首次调用有编译开销）",
    )
    train.add_argument("--log-every", type=int, default=argparse.SUPPRESS)
    train.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="从 --output 里的 last.pt 接着跑（默认从最后一个保存点，含轮内）",
    )
    train.add_argument(
        "--save-every-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="每多少步写一次可续跑的 checkpoint（0 = 只用时间兜底，约 10 分钟）",
    )

    evaluate = sub.add_parser("eval", help="用 checkpoint 跑固定局数评测（三条 track）")
    evaluate.add_argument("--checkpoint", default="", help="模型 checkpoint（纯网络 track 必需）")
    evaluate.add_argument("--track", default="nn_mcts", choices=["nn_value", "nn_policy", "nn_mcts"])
    evaluate.add_argument("--games", type=int, required=True)
    evaluate.add_argument("--seed", type=int, default=0, help="与 C++ 评测用同一个 seed 才能按局配对")
    evaluate.add_argument("--simulations", type=int, default=64, help="仅 nn_mcts 使用")
    evaluate.add_argument("--lanes", type=int, default=32)
    evaluate.add_argument("--in-flight", type=int, default=4)
    evaluate.add_argument("--c-puct", type=float, default=1.5)
    evaluate.add_argument("--cache-entries", type=int, default=1 << 17)
    evaluate.add_argument("--search-threads", type=int, default=1, help="推进 lane 的线程数")
    evaluate.add_argument("--max-batch", type=int, default=8192)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--label", default="")
    evaluate.add_argument("--jsonl", default="", help="逐局记录输出路径")

    rating = sub.add_parser("rating", help="评分池：按旧版口径算评分（对数标度 + 锚点）")
    rating.add_argument(
        "--pool", default="", help="评分池（append-only JSONL），默认 $R2048_OUTPUT_DIR/rating/pool.jsonl"
    )
    rating.add_argument("--anchors", default="", help="锚点文件，默认 $R2048_OUTPUT_DIR/rating/anchors.json")
    rating.add_argument("--ingest", nargs="*", default=[], help="把评测 JSONL 追加进评分池（自动去重）")
    rating.add_argument("--anchor", action="append", default=[], help="指定锚点，形如 track=model_id")
    rating.add_argument("--pair-a", default="", help="配对诊断的基准模型")
    rating.add_argument("--pair-b", default="", help="配对诊断的对照模型")
    rating.add_argument("--pair-track", default="search", choices=["search", "pure"])
    rating.add_argument("--json", default="", help="把评分结果写到文件")

    compare = sub.add_parser("compare", help="两组逐局记录做按局配对 bootstrap")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline-label", default="")
    compare.add_argument("--candidate-label", default="")
    compare.add_argument("--resamples", type=int, default=20000)

    probe = sub.add_parser("probe", help="对单个局面打印网络输出（推理自检用）")
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--board", default="", help="16 个指数，空格或逗号分隔；不给则用新开局")
    probe.add_argument("--seed", type=int, default=1, help="生成新开局时用的种子")
    probe.add_argument("--device", default="cuda")

    selfplay = sub.add_parser(
        "selfplay",
        help="NN 引导的批量自对弈跑谱",
        description=(
            "自动对弈产出训练数据。样本目标为 0 表示不限量：跑到被暂停/终止为止，"
            "之后用同样的参数加 --continue 接着写进同一个批次目录。"
            "暂停/终止＝把 pause/stop 写进控制文件，或者按一次 Ctrl-C："
            "进行中的局会走完，已产出的数据一定被提交。"
        ),
    )
    selfplay.add_argument("--out", default="", help="输出目录（默认 $R2048_OUTPUT_DIR/data/<批次号>）")
    selfplay.add_argument("--batch-id", default="", help="批次号（默认 nn-<模拟次数>-<时间>）")
    selfplay.add_argument(
        "--samples",
        type=int,
        default=0,
        help=(
            "目标样本条数（0 = 不限量，跑到被叫停为止）。多进程 --join 下按**整批**判定："
            "进程数随便改，不需要分摊配额；实际总量会多出在途的几局（达标后进行中的局必须走完）"
        ),
    )
    selfplay.add_argument("--seed", type=int, default=0, help="随机种子（0 表示按时间生成）")
    selfplay.add_argument("--checkpoint", default="", help="模型 checkpoint；不给则用随机初始化模型")
    selfplay.add_argument("--random-init", action="store_true", help="显式使用随机初始化模型（仅链路自检）")
    selfplay.add_argument("--simulations", type=int, default=64)
    selfplay.add_argument("--lanes", type=int, default=64)
    selfplay.add_argument("--in-flight", type=int, default=8, help="每个 lane 在途模拟数（批大小旋钮）")
    selfplay.add_argument("--c-puct", type=float, default=1.5)
    selfplay.add_argument("--temperature", type=float, default=1.0)
    selfplay.add_argument("--temperature-moves", type=int, default=30)
    selfplay.add_argument("--cache-entries", type=int, default=1 << 17)
    selfplay.add_argument(
        "--search-threads",
        type=int,
        default=1,
        help=(
            "推进 lane 的线程数（1 = 单线程；>1 会改变调度，属协议参数）。"
            "注意：>1 产出的数据不可复现（ADR 0031），只能用于吞吐实验；"
            "要并行跑谱请用 --join 多进程（每进程 1 线程）"
        ),
    )
    selfplay.add_argument(
        "--tree-nodes",
        type=int,
        default=2,
        help="记录里放几个搜索树内部节点（0 = 不采集；默认 2，见 ADR 0041/0042）",
    )
    selfplay.add_argument(
        "--tree-label-plays",
        type=int,
        default=8,
        help="树内节点每个候选跑几次独立 playout 作为非循环标签（0 = 不标注，见 ADR 0043）",
    )
    selfplay.add_argument(
        "--tree-label-policy",
        default="depth_stratified",
        choices=["depth_stratified", "top_visits"],
        help="树内节点怎么挑：按深度分层（默认）或按访问数取 top-K",
    )
    selfplay.add_argument(
        "--tree-label-rollout",
        default="greedy",
        choices=["greedy", "uniform"],
        help="树内标注用的续弈策略（默认 greedy，见 ADR 0030 §6.1）",
    )
    selfplay.add_argument("--shard-samples", type=int, default=100000)
    selfplay.add_argument("--max-batch", type=int, default=16384)
    selfplay.add_argument("--device", default="cuda", help="cuda / cpu（默认 cuda）")
    selfplay.add_argument("--model-id", default="", help="写入数据来源记录的模型标识")
    selfplay.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help="续跑：在既有批次上接着编号写下去（参数逐项比对，不一致直接拒绝）",
    )
    selfplay.add_argument(
        "--join",
        action="store_true",
        help="加入已有批次并行写（多个进程指向同一个 --out）：局号跨进程统一发放，仍是一个批次",
    )
    selfplay.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "起 N 个进程写同一批数据（默认 1 = 本进程自己跑）。N>1 时本进程只做监督："
            "起 N 个 --join worker、报批级进度、把 Ctrl-C 变成一次 stop 广播。"
            "样本目标是整批的，进程数随便改（ADR 0029 的订正）；worker 一律单线程（ADR 0031）"
        ),
    )
    selfplay.add_argument(
        "--control", default="", help="控制文件路径（默认 <数据批目录>/control，内容是 pause/stop）"
    )

    merge = sub.add_parser(
        "merge-halves",
        help="把两个半模型拼成完整模型（价值取一边、策略取另一边）",
        description=(
            "双塔下价值头与策略头分开训练时，两个 run 各自只在一个头上是新的。"
            "评测“这一轮的搜索强度”必须用这一轮的价值 + 这一轮的策略。"
        ),
    )
    merge.add_argument("--value", required=True, help="提供非策略张量的 checkpoint")
    merge.add_argument("--policy", required=True, help="提供策略张量的 checkpoint")
    merge.add_argument("--output", required=True, help="合并后的 checkpoint 路径")
    merge.add_argument("--note", default="", help="写进元数据的说明（为什么这样合）")

    iterate = sub.add_parser(
        "iterate",
        help="一次迭代：跑谱 → 训练 → 评测 → 评分",
        description=(
            "薄驱动：四步各自都是对既有入口的一次调用，每步做完在迭代目录留一个标记，"
            "重跑时跳过已完成的步骤。--steps 只跑其中几步，--force 重做已完成的步骤。"
            "训练数据集 = --config 里 datasets 声明的池子 + 这一轮新产出的数据（按 --data-weight）。"
        ),
    )
    iterate.add_argument("--name", required=True, help="迭代名（例如 c3）：产物路径与评测标签都由它派生")
    iterate.add_argument("--config", required=True, help="训练配置文件（YAML：datasets + train）")
    iterate.add_argument("--output", default="", help="迭代目录（默认 $R2048_OUTPUT_DIR/iters/<name>）")
    iterate.add_argument(
        "--steps",
        default="",
        help="要跑的步骤，逗号分隔（默认全部四个：selfplay,train,evaluate,rate）",
    )
    iterate.add_argument("--force", action="store_true", help="重做已经完成的步骤")
    iterate.add_argument(
        "--init-checkpoint",
        default="",
        help="上一轮的完整模型：跑谱用它当对手，训练用它当起点。不给则用纯 MCTS 当教师（首轮）",
    )
    iterate.add_argument("--samples", type=int, default=0, help="跑谱目标样本数（> 0）")
    iterate.add_argument("--data-seed", type=int, default=0, help="跑谱的种子")
    iterate.add_argument("--data-weight", type=float, default=1.0, help="新数据在训练池里的权重")
    iterate.add_argument("--simulations", type=int, default=64, help="跑谱/评测共用的模拟数")
    iterate.add_argument("--lanes", type=int, default=64)
    iterate.add_argument("--in-flight", type=int, default=8)
    iterate.add_argument("--search-threads", type=int, default=1, help="推进 lane 的线程数（协议参数）")
    iterate.add_argument(
        "--data-workers",
        type=int,
        default=4,
        help="跑谱并行度：起 N 个进程写同一批（每 worker 单线程，数据可复现；样本目标按整批算）",
    )
    iterate.add_argument(
        "--rollout-policy",
        default="greedy",
        choices=["uniform", "greedy"],
        help="首轮纯 C++ 教师的 rollout 续弈策略（默认 greedy，见 ADR 0030）",
    )
    iterate.add_argument(
        "--tree-nodes",
        type=int,
        default=2,
        help="记录里放几个搜索树内部节点（0 = 不采集；默认 2，见 ADR 0041/0042）",
    )
    iterate.add_argument(
        "--tree-label-plays", type=int, default=8, help="树内节点每个候选的 playout 次数（ADR 0043）"
    )
    iterate.add_argument(
        "--tree-label-policy",
        default="depth_stratified",
        choices=["depth_stratified", "top_visits"],
    )
    iterate.add_argument("--tree-label-rollout", default="greedy", choices=["greedy", "uniform"])
    iterate.add_argument("--c-puct", type=float, default=1.5)
    iterate.add_argument("--device", default="cuda")
    iterate.add_argument("--eval-games", type=int, default=400)
    iterate.add_argument("--eval-seed", type=int, default=0, help="评测种子（各臂对齐开局用）")
    iterate.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="基线臂：random 或 mcts<模拟数>（可重复；默认 random 与 mcts64）",
    )
    iterate.add_argument(
        "--anchor", action="append", default=[], help="刷新锚点，形如 track=model_id（可重复）"
    )

    session = sub.add_parser(
        "session",
        help="启动会话进程（文本协议，供界面与脚本驱动；见 ADR 0038）",
        description=(
            "一个盘面 + 一条棋谱 + 一个引擎。命令走 stdin/stdout 的文本协议，一行一条："
            "clear / newgame [seed=N]（标准开局）/ set <值> <坐标> / move <方向> / genmove / "
            "analyze / undo / state / load <路径> / save <路径> / list_commands / quit。"
            "编辑与复盘不需要 --checkpoint；分析与落子需要。"
        ),
    )
    session.add_argument("--checkpoint", default="", help="模型 checkpoint；不给则这个会话只能编辑与复盘")
    session.add_argument("--device", default="cpu", help="cpu / cuda（交互分析默认 cpu）")
    session.add_argument("--strategy", default="search", choices=["search", "policy", "value"])
    session.add_argument("--simulations", type=int, default=256, help="每次分析的模拟数")
    session.add_argument("--time-ms", type=int, default=0, help="每次分析的时间上限（毫秒，0 = 不限时）")
    session.add_argument("--in-flight", type=int, default=4, help="每 lane 在途模拟数（批大小旋钮）")
    session.add_argument("--cache-entries", type=int, default=1 << 16)
    session.add_argument("--seed", type=int, default=0)

    webui = sub.add_parser(
        "webui",
        help="启动 Gradio 交互界面（可选组件：需要 webui 依赖组，见 ADR 0039）",
        description=(
            "交互分析界面：计算/分析、棋盘编辑器、复盘三个标签页。它只跟会话进程说话"
            "（`r2048 session`），自己不做棋规、不选招。默认只绑 127.0.0.1。"
        ),
    )
    webui.add_argument("--host", default="127.0.0.1", help="监听地址（默认只绑本机）")
    webui.add_argument("--port", type=int, default=7860)
    webui.add_argument("--share", action="store_true", help="生成公网链接（默认不对外暴露）")
    return parser


def _command_version() -> int:
    import r2048_core

    info = {
        "python_package": __version__,
        "rules_version": int(r2048_core.RULES_VERSION),
        "protocol_version": int(r2048_core.PROTOCOL_VERSION),
        "build_id": str(r2048_core.BUILD_ID),
        "torch": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch 是必装依赖，这里只是稳妥
        pass
    print(json.dumps(info, ensure_ascii=False))
    return 0


def _command_env() -> int:
    import r2048_core

    info = {
        "output_root": str(paths.output_root()),
        "ext_build_id": str(r2048_core.BUILD_ID),
        "python": sys.executable,
    }
    print(json.dumps(info, ensure_ascii=False))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    from r2048.config import build_train_config
    from r2048.jobs import JobDir
    from r2048.nn.training import config_payload, data_label, run_training

    # 参数合并（CLI > YAML > 默认值）全在配置层：命令行这里不逐个字段搬运，
    # 少一处"新加了字段但忘了接线"的地方。
    config = build_train_config(args)
    # 训练的运行目录本来就是 --output，作业状态与日志直接落在同一个目录里：
    # 事后排查“这次训练是怎么跑的”时，job.json/status.json 与 train.json 在同一处。
    with JobDir(Path(args.output).name, "train", params=config_payload(config), path=args.output) as job:
        config.on_progress = job.callback()
        result = run_training(config)
        job.finish(best_checkpoint=result.best_checkpoint, train_json=result.train_json)
    print(f"完成：{data_label(config)} 最佳 checkpoint {result.best_checkpoint}")
    print(f"运行记录 {result.train_json}")
    # stdout 只放可被脚本消费的结果。
    print(result.best_checkpoint)
    return 0


def _command_eval(args: argparse.Namespace) -> int:
    import time

    from r2048.jobs import JobDir
    from r2048.workflows.evaluate import EvalConfig, run_evaluation

    config = EvalConfig(
        games=args.games,
        seed=args.seed,
        checkpoint=args.checkpoint,
        track=args.track,
        simulations=args.simulations,
        lanes=args.lanes,
        in_flight=args.in_flight,
        c_puct=args.c_puct,
        cache_entries=args.cache_entries,
        search_threads=args.search_threads,
        max_batch=args.max_batch,
        device=args.device,
        label=args.label,
        jsonl=args.jsonl,
    )
    label = config.label or config.track
    job_id = f"eval-{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    with JobDir(job_id, "eval", params={**config.__dict__, "job_id": job_id}) as job:
        config.on_progress = job.callback()
        result = run_evaluation(config)
        job.finish(
            jsonl=result.jsonl, games=len(result.games), protocol_hash=result.summary()["protocol_hash"]
        )
    summary = result.summary()
    tiles = "，".join(f"{size}: {count}" for size, count in summary["tiles"].items())
    print(
        f"{result.label}（{result.track}）{summary['games']} 局：均分 {summary['mean_score']:.0f}，"
        f"中位 {summary['median_score']:.0f}，平均步数 {summary['mean_steps']:.0f}，"
        f"耗时 {summary['seconds']:.1f}s（{summary['games_per_second']:.1f} 局/秒）"
        + (f"，达到 {tiles}" if tiles else "")
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _command_rating(args: argparse.Namespace) -> int:
    """评分池：追加评测、定锚点、全盘重算，并打印评分表与配对诊断。

    池是 append-only 的：新增对局后全盘重算，评分因此可以随数据增长变准，也能事后追溯。
    """
    from r2048.workflows import rating as rating_module

    root = paths.output_root() / "rating"
    pool_path = Path(args.pool) if args.pool else root / "pool.jsonl"
    anchors_path = Path(args.anchors) if args.anchors else root / "anchors.json"

    if args.ingest:
        added = rating_module.ingest_jsonl(pool_path, args.ingest)
        print(f"新增 {added} 局到评分池 {pool_path}")

    rows = rating_module.load_pool(pool_path)
    if not rows:
        raise SystemExit(f"评分池为空：{pool_path}")

    if args.anchor:
        spec = {}
        for entry in args.anchor:
            track, _, model_id = entry.partition("=")
            spec[track] = model_id
        rating_module.save_anchors(anchors_path, rows, spec)
        print(f"锚点已写入 {anchors_path}: {spec}")

    anchors = rating_module.load_anchors(anchors_path)
    ratings = rating_module.compute_ratings(rows, anchors)
    openings = rating_module.count_openings()
    print(
        f"开局：{openings['distinct_boards']} 种（D4 简并后 {openings['d4_orbits']} 种）——"
        "这就是“同 seed 对齐开局”能提供的全部信息"
    )

    report = {"openings": openings, "tracks": {}}
    for track, blocks in ratings.items():
        for block in blocks:
            anchor = block["anchor"]
            print(
                f"\n[{track}] 锚点 {anchor['model_id']}（{anchor['n_ref']} 局）= 1000，"
                f"协议 {block['protocol_hash']}"
            )
            # 同一 track 的不同协议都会放到这把尺子上（曲线就是这么来的），
            # 所以差异必须打出来：不然“这条线是在什么条件下得到的”只能靠比对两个哈希。
            if block["differs_from_anchor"]:
                difference = "，".join(
                    f"{name} {pair[0]}→{pair[1]}" for name, pair in block["differs_from_anchor"].items()
                )
                print(f"  与锚点协议不同的字段：{difference}")
            print(
                f"{'模型':<22}{'评分':>10}{'±95%':>8}{'局数':>7}{'对数均值':>10}{'均分':>10}"
                f"{'中位':>9}{'平均步数':>9}  每万局到达 2048/4096"
            )
            ordered = sorted(block["models"].items(), key=lambda item: -item[1]["rating"])
            for model_id, stats in ordered:
                tiles = stats["tile_reach_per_10k"]
                print(
                    f"{model_id:<22}{stats['rating']:>10.1f}{stats['ci95']:>8.1f}{stats['n']:>7}"
                    f"{stats['L']:>10.3f}{stats['mean_score']:>10.0f}{stats['median_score']:>9.0f}"
                    f"{stats['mean_steps']:>9.0f}  {tiles.get('2048', 0):>5.1f} / {tiles.get('4096', 0):>5.1f}"
                )
            report["tracks"].setdefault(track, []).append(
                {
                    "anchor": anchor["model_id"],
                    "protocol_hash": block["protocol_hash"],
                    "protocol": block["protocol"],
                    "differs_from_anchor": block["differs_from_anchor"],
                    "models": {
                        model_id: {
                            "rating": stats["rating"],
                            "ci95": stats["ci95"],
                            "n": stats["n"],
                            "L": stats["L"],
                            "mean_score": stats["mean_score"],
                            "median_score": stats["median_score"],
                            "tile_reach_per_10k": stats["tile_reach_per_10k"],
                        }
                        for model_id, stats in block["models"].items()
                    },
                }
            )

    # 模型间差异：同时给“评分口径（独立两样本）”与“同 seed 配对差分”，
    # 后者用来量化配对到底省了多少方差，而不是默认它有效。
    if args.pair_a and args.pair_b:
        diagnostic = rating_module.pairing_diagnostic(rows, args.pair_a, args.pair_b, args.pair_track)
        paired = diagnostic["paired"]
        independent = diagnostic["independent"]
        print(
            f"\n配对诊断 [{diagnostic['track']}] {args.pair_b} − {args.pair_a}，"
            f"{diagnostic['games']} 个共同局号（同 eval_seed）"
        )
        print(
            f"  评分口径（独立两样本）：{independent['rating_difference']:+.1f}，"
            f"95% CI [{independent['rating_ci95'][0]:+.1f}, {independent['rating_ci95'][1]:+.1f}]，"
            f"标准误 {independent['se']:.4f} log2"
        )
        print(
            f"  同 seed 配对差分        ：{paired['rating_difference']:+.1f}，"
            f"95% CI [{paired['rating_ci95'][0]:+.1f}, {paired['rating_ci95'][1]:+.1f}]，"
            f"标准误 {paired['se']:.4f} log2"
        )
        print(
            f"  逐局 log2 得分相关系数 {diagnostic['log2_correlation']:+.3f} → "
            f"配对把标准误压低了 {diagnostic['variance_reduction'] * 100:+.1f}%"
        )
        report["pairing"] = diagnostic

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n评分结果写入 {args.json}")
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    from r2048.workflows.evaluate import compare

    result = compare(
        args.baseline,
        args.candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        resamples=args.resamples,
    )
    print(
        f"{result['candidate']} − {result['baseline']} = {result['mean_difference']:+.1f} "
        f"（95% CI [{result['ci95'][0]:+.1f}, {result['ci95'][1]:+.1f}]，"
        f"{result['wins']} 胜 {result['draws']} 平 {result['losses']} 负，"
        f"{result['games']} 局配对）" + ("，显著" if result["significant"] else "，区间跨零：不可下结论")
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _command_probe(args: argparse.Namespace) -> int:
    """打印一个局面的网络输出：每个方向的合并得分、afterstate 的价值、以及策略分布。

    计算在 `r2048.runtime.probe` 里（界面与脚本复用同一条口径），这里只负责排版。
    """
    from r2048.runtime.probe import DIRECTION_NAMES, parse_board, probe_position

    board = parse_board(args.board) if args.board else None
    result = probe_position(args.checkpoint, board=board, seed=args.seed, device=args.device)

    print("局面（指数）:", " ".join(str(int(cell)) for cell in result.board))
    print(f"合法方向掩码 {result.legal_mask:04b}，模型 {result.model_id}，结构 {result.architecture}")
    scores = result.directional_scores()
    for action in range(4):
        mark = " 合法" if (result.legal_mask >> action) & 1 else " 非法"
        print(
            f"  {DIRECTION_NAMES[action]}({action}){mark}: "
            f"合并得分 {result.merge_scores[action]:6d}  "
            f"A(u) {result.afterstate_values[action]:9.1f}  "
            f"r+A(u) {scores[action]:9.1f}  "
            f"π {result.policy[action]:.3f}"
        )
    return 0


def _command_selfplay(args: argparse.Namespace) -> int:
    import time

    import torch

    from r2048.jobs import JobDir
    from r2048.nn.model import create_model, load_checkpoint
    from r2048.runtime.selfplay import SelfplayConfig, run_selfplay

    if args.device == "cuda" and not torch.cuda.is_available():
        print("请求了 cuda 但当前不可用，回退到 cpu", file=sys.stderr)
        args.device = "cpu"
    device = torch.device(args.device)

    seed = args.seed if args.seed else int(time.time())
    if args.checkpoint:
        model, metadata = load_checkpoint(args.checkpoint, device)
        model_id = args.model_id or str(metadata.get("model_id", ""))
    else:
        # 随机初始化的权重也必须由种子决定：否则同一条命令跑两次得到两份不同的数据，
        # 而"没有 checkpoint 也能复现整条链路"正是这个选项存在的理由。
        torch.manual_seed(seed)
        model = create_model().to(device).eval()
        model_id = args.model_id or "random-init"
        if not args.random_init:
            print("未提供 --checkpoint：使用随机初始化模型（只用于链路自检）", file=sys.stderr)

    batch_id = args.batch_id or f"nn-s{args.simulations}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.out) if args.out else paths.data_dir(batch_id)

    config = SelfplayConfig(
        data_dir=str(out_dir),
        batch_id=batch_id,
        samples=args.samples,
        seed=seed,
        model_id=model_id,
        simulations=args.simulations,
        lanes=args.lanes,
        in_flight_per_lane=args.in_flight,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        cache_entries=args.cache_entries,
        search_threads=args.search_threads,
        shard_samples=args.shard_samples,
        max_batch=args.max_batch,
        tree_nodes=args.tree_nodes,
        tree_label_plays=args.tree_label_plays,
        tree_label_policy=args.tree_label_policy,
        tree_label_rollout=args.tree_label_rollout,
        control_path=args.control or "",
        resume=args.continue_,
        join=args.join,
    )
    print(
        f"计划：NN 自对弈，模型 {model_id}，模拟 {config.simulations}，lane {config.lanes}，"
        f"样本 {config.samples or '不限量'}，种子 {seed}，设备 {args.device}，输出 {out_dir}"
        + ("（续跑）" if config.resume else ""),
        file=sys.stderr,
    )

    # 作业目录与数据批同名：一次跑谱的现场记录（status/log）和它产出的数据放在
    # 两个并列的目录里，事后能一眼对上。**join 时按进程分目录**：多个进程共用一个
    # 作业目录会互相踩 status.json.tmp（实测直接崩掉两个进程）。
    job_id = f"{batch_id}-p{os.getpid()}" if args.join else batch_id
    if args.workers > 1:
        # 多进程：本进程只做监督，worker 是各自独立的进程（它们会各自建自己的作业目录）。
        from r2048.runtime.worker_pool import run_selfplay_workers

        with JobDir(job_id, "selfplay", params={**config.__dict__, "workers": args.workers}) as job:
            workers_result = run_selfplay_workers(
                config, args.checkpoint, args.device, args.workers, on_progress=job.callback()
            )
            job.finish(
                manifest=workers_result.manifest_path,
                state=workers_result.state,
                samples=workers_result.samples,
                episodes=workers_result.episodes,
                seconds=round(workers_result.seconds, 3),
            )
        print(
            f"完成：{args.workers} 个进程写同一批，状态 {workers_result.state}，"
            f"样本 {workers_result.samples}，局数 {workers_result.episodes}，"
            f"耗时 {workers_result.seconds:.1f} 秒"
            f"（{workers_result.samples_per_second:.0f} 样本/秒）",
            file=sys.stderr,
        )
        for index, log in enumerate(workers_result.logs):
            print(f"  worker-{index} 日志：{log}", file=sys.stderr)
        print(workers_result.manifest_path)
        return 0

    with JobDir(job_id, "selfplay", params=config.__dict__) as job:
        config.on_progress = job.callback()
        result = run_selfplay(config, model, device)
        job.finish(
            manifest=result.manifest_path,
            state=result.state,
            samples=result.samples,
            episodes=result.episodes,
            seconds=round(result.seconds, 3),
        )
    stats = result.stats
    print(
        f"完成：状态 {result.state}，样本 {result.samples}，局数 {result.episodes}，"
        f"耗时 {result.seconds:.1f} 秒"
        f"（{result.samples_per_second:.0f} 样本/秒），前向 {result.forward_batches} 次，"
        f"平均每批 {result.rows_per_forward:.0f} 行，"
        f"叶子缓存命中 {stats['leaf_cache_rate'] * 100:.1f}%",
        file=sys.stderr,
    )
    if result.state != "complete":
        print(
            f"这批数据没有跑满目标（{result.state}）：可以用同样的参数加 --continue 接着跑，"
            "数据会写进同一个批次目录",
            file=sys.stderr,
        )
    # stdout 只放可被脚本消费的结果：manifest 路径。
    print(result.manifest_path)
    return 0


def _command_merge_halves(args: argparse.Namespace) -> int:
    from r2048.workflows.merge import merge_halves

    result = merge_halves(args.value, args.policy, args.output, note=args.note)
    print(
        f"合并完成：策略张量 {result.policy_tensors} 个、价值张量 {result.value_tensors} 个，"
        f"标识 {result.model_id}",
        file=sys.stderr,
    )
    # stdout 只放可被脚本消费的结果（ADR 0008）：合并后的 checkpoint 路径。
    print(result.path)
    return 0


def _command_iterate(args: argparse.Namespace) -> int:
    from r2048.iterate import DEFAULT_BASELINES, STEPS, IterateConfig, run_iteration

    steps = tuple(part.strip() for part in args.steps.split(",") if part.strip()) or STEPS
    anchors: dict[str, str] = {}
    for entry in args.anchor:
        track, _, model_id = entry.partition("=")
        if not track or not model_id:
            raise SystemExit(f"--anchor 必须是 track=model_id 的形状，收到 {entry!r}")
        anchors[track] = model_id
    config = IterateConfig(
        name=args.name,
        config_path=args.config,
        output=args.output,
        steps=steps,
        force=args.force,
        init_checkpoint=args.init_checkpoint,
        samples=args.samples,
        data_seed=args.data_seed,
        data_weight=args.data_weight,
        simulations=args.simulations,
        lanes=args.lanes,
        in_flight=args.in_flight,
        search_threads=args.search_threads,
        data_workers=args.data_workers,
        rollout_policy=args.rollout_policy,
        tree_nodes=args.tree_nodes,
        tree_label_plays=args.tree_label_plays,
        tree_label_policy=args.tree_label_policy,
        tree_label_rollout=args.tree_label_rollout,
        c_puct=args.c_puct,
        device=args.device,
        eval_games=args.eval_games,
        eval_seed=args.eval_seed,
        baselines=tuple(args.baseline) or DEFAULT_BASELINES,
        anchors=anchors,
    )
    result = run_iteration(config)
    for step, produced in result.steps.items():
        print(f"{step}: {produced}")
    print(f"评分 {result.rating_json}")
    # stdout 只放可被脚本消费的结果：这一次迭代产出的完整模型路径（没有训练步骤时为空）。
    print(result.steps.get("train", {}).get("model", ""))
    return 0


def _command_session(args: argparse.Namespace) -> int:
    """会话进程：把 stdin/stdout 交给文本协议。

    **stdout 是协议通道**，所以这里一句都不往上写（就像 ADR 0008 里 stdout 是结果通道那样）；
    给人看的就绪信息走 stderr。
    """
    from r2048.session import Session, SessionConfig
    from r2048.session_io import NAME, serve

    config = SessionConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        strategy=args.strategy,
        simulations=args.simulations,
        time_ms=args.time_ms,
        in_flight=args.in_flight,
        cache_entries=args.cache_entries,
        seed=args.seed,
    )
    session = Session(config)
    print(
        f"{NAME} 就绪：策略 {config.strategy}，设备 {config.device}，"
        + (f"模型 {session.state()['model_id']}" if args.checkpoint else "无模型（只能编辑与复盘）"),
        file=sys.stderr,
        flush=True,
    )
    return serve(session, sys.stdin, sys.stdout)


def _command_webui(args: argparse.Namespace) -> int:
    """启动 Gradio 界面。gradio 在可选依赖组里，没装就给一条能照做的提示。"""
    try:
        from r2048.webui.app import build_app
    except ImportError as error:  # pragma: no cover - 只在没装 webui 组时走到
        raise SystemExit(
            "WebUI 是可选组件（ADR 0039）：先 `uv sync --group webui` 装上 gradio，或直接用 `make webui`"
        ) from error

    demo, state = build_app()
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_api=False,
        )
    finally:
        state.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "version":
        return _command_version()
    if args.command == "env":
        return _command_env()
    if args.command == "train":
        return _command_train(args)
    if args.command == "eval":
        return _command_eval(args)
    if args.command == "rating":
        return _command_rating(args)
    if args.command == "compare":
        return _command_compare(args)
    if args.command == "probe":
        return _command_probe(args)
    if args.command == "selfplay":
        return _command_selfplay(args)
    if args.command == "merge-halves":
        return _command_merge_halves(args)
    if args.command == "iterate":
        return _command_iterate(args)
    if args.command == "session":
        return _command_session(args)
    if args.command == "webui":
        return _command_webui(args)
    raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
