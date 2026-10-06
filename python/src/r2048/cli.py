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

    selfplay = sub.add_parser("selfplay", help="NN 引导的批量自对弈跑谱")
    selfplay.add_argument("--out", default="", help="输出目录（默认 $R2048_OUTPUT_DIR/data/<批次号>）")
    selfplay.add_argument("--batch-id", default="", help="批次号（默认 nn-<模拟次数>-<时间>）")
    selfplay.add_argument("--samples", type=int, required=True, help="目标样本条数")
    selfplay.add_argument("--seed", type=int, default=0, help="随机种子（0 表示按时间生成）")
    selfplay.add_argument("--checkpoint", default="", help="模型 checkpoint；不给则用随机初始化模型")
    selfplay.add_argument("--random-init", action="store_true", help="显式使用随机初始化模型（仅链路自检）")
    selfplay.add_argument("--simulations", type=int, default=64)
    selfplay.add_argument("--lanes", type=int, default=8)
    selfplay.add_argument("--in-flight", type=int, default=8, help="每个 lane 在途模拟数（批大小旋钮）")
    selfplay.add_argument("--c-puct", type=float, default=1.5)
    selfplay.add_argument("--temperature", type=float, default=1.0)
    selfplay.add_argument("--temperature-moves", type=int, default=30)
    selfplay.add_argument("--cache-entries", type=int, default=1 << 17)
    selfplay.add_argument("--shard-samples", type=int, default=100000)
    selfplay.add_argument("--max-batch", type=int, default=8192)
    selfplay.add_argument("--threads", type=int, default=0)
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
        threads=args.threads,
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
    if args.command == "selfplay":
        return _command_selfplay(args)
    raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
