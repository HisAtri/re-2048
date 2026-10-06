"""`python -m r2048 <子命令>`：Python 侧的入口。

只放需要神经网络与 PyTorch 的事情；规则、搜索、纯随机/UCT 跑谱全部在 C++ CLI 里，
两边不重复实现同一件事。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from r2048 import __version__, paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m r2048", description="2048 自博弈 Python 侧入口")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="打印 Python 侧与 C++ 内核的版本信息")
    sub.add_parser("env", help="打印输出目录等路径解析结果")

    train = sub.add_parser("train", help="训练价值头 / 策略头 / 联合 / 双塔")
    train.add_argument("--data", required=True, help="数据批目录（含 manifest.json）")
    train.add_argument("--output", required=True, help="运行目录（写 checkpoint 与 train.json）")
    train.add_argument("--mode", default="value", choices=["value", "policy", "joint", "dual"])
    train.add_argument("--init-checkpoint", default="", help="从已有 checkpoint 初始化")
    train.add_argument("--architecture", default="", help="覆盖结构（默认按模式选）")
    train.add_argument("--freeze-value", action="store_true", help="只训练策略路径（策略模式的推荐做法）")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--value-loss", default="huber", choices=["huber", "mse"])
    train.add_argument("--value-weight", type=float, default=1.0)
    train.add_argument("--policy-weight", type=float, default=1.0)
    train.add_argument("--val-fraction", type=float, default=0.05)
    train.add_argument("--split-seed", type=int, default=0)
    train.add_argument("--no-augment", action="store_true", help="关闭 D4 增强")
    train.add_argument("--max-samples", type=int, default=0, help="限制训练样本数（调试用）")
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--amp", default="off", choices=["off", "bf16"])
    train.add_argument("--no-channels-last", action="store_true")
    train.add_argument("--compile", action="store_true", help="torch.compile（首次调用有编译开销）")
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--resume", action="store_true")

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
    evaluate.add_argument("--max-batch", type=int, default=8192)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--label", default="")
    evaluate.add_argument("--jsonl", default="", help="逐局记录输出路径")

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

    selfplay = sub.add_parser("selfplay", help="NN 引导的批量自对弈跑谱")
    selfplay.add_argument("--out", default="", help="输出目录（默认 $R2048_OUTPUT_DIR/data/<批次号>）")
    selfplay.add_argument("--batch-id", default="", help="批次号（默认 nn-<模拟次数>-<时间>）")
    selfplay.add_argument("--samples", type=int, required=True, help="目标样本条数")
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
    selfplay.add_argument("--shard-samples", type=int, default=100000)
    selfplay.add_argument("--max-batch", type=int, default=16384)
    selfplay.add_argument("--device", default="cuda", help="cuda / cpu（默认 cuda）")
    selfplay.add_argument("--model-id", default="", help="写入数据来源记录的模型标识")
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
    from r2048.nn.training import TrainConfig, run_training

    config = TrainConfig(
        data=args.data,
        output=args.output,
        mode=args.mode,
        init_checkpoint=args.init_checkpoint,
        architecture=args.architecture,
        freeze_value=args.freeze_value,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        value_loss=args.value_loss,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        augment=not args.no_augment,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        amp=args.amp,
        channels_last=not args.no_channels_last,
        compile=args.compile,
        log_every=args.log_every,
        resume=args.resume,
    )
    result = run_training(config)
    print(f"完成：最佳 checkpoint {result.best_checkpoint}")
    print(f"运行记录 {result.train_json}")
    # stdout 只放可被脚本消费的结果。
    print(result.best_checkpoint)
    return 0


def _command_eval(args: argparse.Namespace) -> int:
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
        max_batch=args.max_batch,
        device=args.device,
        label=args.label,
        jsonl=args.jsonl,
    )
    result = run_evaluation(config)
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
        f"{result['games']} 局配对）"
        + ("，显著" if result["significant"] else "，区间跨零：不可下结论")
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _command_probe(args: argparse.Namespace) -> int:
    """打印一个局面的网络输出：每个方向的合并得分、afterstate 的价值、以及策略分布。

    它只做“打印”，不做选招——选招口径只有 C++ 引擎一处，避免探针与引擎给出不同答案。
    """
    import numpy as np
    import torch

    import r2048_core

    from r2048.nn.encode import encode_boards
    from r2048.nn.model import load_checkpoint

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model, metadata = load_checkpoint(args.checkpoint, device)
    scale = float(r2048_core.SCORE_SCALE)

    if args.board:
        values = [int(part) for part in args.board.replace(",", " ").split()]
        if len(values) != 16:
            raise SystemExit("--board 需要 16 个指数")
        board = np.array(values, dtype=np.uint8)
    else:
        # 用与跑谱一致的开局生成口径（同一 seed 与局号 → 同一开局）。
        import random

        rng = random.Random(args.seed)
        board = np.zeros(16, dtype=np.uint8)
        for _ in range(2):
            empty = [i for i in range(16) if board[i] == 0]
            board[rng.choice(empty)] = 2 if rng.random() < 0.1 else 1
    legal = int(r2048_core.legal_mask(board))

    rows = [board]
    scores = []
    for action in range(4):
        outcome = r2048_core.apply_move(board, action)
        rows.append(np.asarray(outcome["board"], dtype=np.uint8))
        scores.append(int(outcome["score"]))
    with torch.no_grad():
        value, logits = model(encode_boards(torch.from_numpy(np.stack(rows)).to(torch.long).to(device)))
    afterstate_values = (value[1:] * scale).tolist()
    logits = logits[0].float().cpu()
    masked = torch.where(
        torch.tensor([(legal >> action) & 1 == 1 for action in range(4)], dtype=torch.bool),
        logits,
        torch.tensor(float("-inf")),
    )
    policy = torch.softmax(masked, dim=-1).tolist()

    names = ["上", "右", "下", "左"]
    print("局面（指数）:", " ".join(str(int(cell)) for cell in board))
    print(f"合法方向掩码 {legal:04b}，模型 {metadata.get('model_id', '')}，结构 {metadata.get('architecture', '')}")
    for action in range(4):
        mark = " 合法" if (legal >> action) & 1 else " 非法"
        print(
            f"  {names[action]}({action}){mark}: 合并得分 {scores[action]:6d}  "
            f"A(u) {afterstate_values[action]:9.1f}  r+A(u) {scores[action] + afterstate_values[action]:9.1f}  "
            f"π {policy[action]:.3f}"
        )
    return 0


def _command_selfplay(args: argparse.Namespace) -> int:
    import time

    import torch

    from r2048.nn.model import create_model, load_checkpoint
    from r2048.runtime.selfplay import SelfplayConfig, run_selfplay

    if args.device == "cuda" and not torch.cuda.is_available():
        print("请求了 cuda 但当前不可用，回退到 cpu", file=sys.stderr)
        args.device = "cpu"
    device = torch.device(args.device)

    if args.checkpoint:
        model, metadata = load_checkpoint(args.checkpoint, device)
        model_id = args.model_id or str(metadata.get("model_id", ""))
    else:
        model = create_model().to(device).eval()
        model_id = args.model_id or "random-init"
        if not args.random_init:
            print("未提供 --checkpoint：使用随机初始化模型（只用于链路自检）", file=sys.stderr)

    seed = args.seed if args.seed else int(time.time())
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
        shard_samples=args.shard_samples,
        max_batch=args.max_batch,
    )
    print(
        f"计划：NN 自对弈，模型 {model_id}，模拟 {config.simulations}，lane {config.lanes}，"
        f"样本 {config.samples}，种子 {seed}，设备 {args.device}，输出 {out_dir}",
        file=sys.stderr,
    )

    result = run_selfplay(config, model, device)
    stats = result.stats
    print(
        f"完成：样本 {result.samples}，局数 {result.episodes}，耗时 {result.seconds:.1f} 秒"
        f"（{result.samples_per_second:.0f} 样本/秒），前向 {result.forward_batches} 次，"
        f"平均每批 {result.rows_per_forward:.0f} 行，"
        f"叶子缓存命中 {stats['leaf_cache_rate'] * 100:.1f}%",
        file=sys.stderr,
    )
    # stdout 只放可被脚本消费的结果：manifest 路径。
    print(result.manifest_path)
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
    if args.command == "compare":
        return _command_compare(args)
    if args.command == "probe":
        return _command_probe(args)
    if args.command == "selfplay":
        return _command_selfplay(args)
    raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
