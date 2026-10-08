"""NN 自对弈的端到端测试：跑通两阶段批推理协议，并守住“数据可复现”的边界。

覆盖三件事：
1. 一次小规模自对弈能产出通过 C++ 侧结构校验的数据批；
2. 每批请求的平均行数远大于 1（这是两阶段协议存在的意义，退化成逐样本调用就是失败）；
3. 相同配置与种子在同一设备上重复运行，产出的分片逐字节相同。
   注意范围：NN 自对弈的可复现性以“模型、设备、批处理配置都相同”为前提——
   浮点批大小变化会改变归约顺序，因此它不像纯 C++ 跑谱那样无条件逐字节可复现。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import r2048_core
import torch

from r2048.nn.model import create_model
from r2048.runtime.selfplay import SelfplayConfig, run_selfplay
from r2048.runtime.worker_pool import run_selfplay_workers

# 小规模：测试只验证链路与不变式，不追求成绩。
SMALL = {"samples": 1200, "simulations": 8, "lanes": 8, "in_flight_per_lane": 4, "shard_samples": 1200}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _config(data_dir: Path, seed: int) -> SelfplayConfig:
    return SelfplayConfig(
        data_dir=str(data_dir),
        batch_id="selfplay-test",
        samples=SMALL["samples"],
        seed=seed,
        model_id="random-init",
        simulations=SMALL["simulations"],
        lanes=SMALL["lanes"],
        in_flight_per_lane=SMALL["in_flight_per_lane"],
        shard_samples=SMALL["shard_samples"],
        max_batch=4096,
        progress_every=0.0,  # 测试里不要进度输出
    )


def _model() -> torch.nn.Module:
    torch.manual_seed(20261006)  # 固定随机初始化权重，保证两次运行用同一个模型
    model = create_model().to(_device()).eval()
    return model


def _record_bytes() -> int:
    from r2048.data.shards import RECORD_BYTES

    return RECORD_BYTES


def _verify(data_dir: Path) -> dict:
    binary = os.environ.get("R2048_TEST_BINARY")
    if not binary:
        pytest.skip("未设置 R2048_TEST_BINARY：通过 make test 运行")
    completed = subprocess.run(
        [binary.replace("r2048_tests", "r2048"), "verify", "--data", str(data_dir), "--no-replay"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, f"数据批校验失败：\n{completed.stdout}\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_selfplay_produces_verifiable_batch(tmp_path: Path) -> None:
    data_dir = tmp_path / "batch"
    result = run_selfplay(_config(data_dir, seed=12345), _model(), _device())

    assert result.samples == SMALL["samples"], "样本数必须恰好等于目标条数"
    assert result.episodes > 0
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["policy"] == "nn"
    assert manifest["simulations"] == SMALL["simulations"]
    # 记录长度由 schema 版本决定（v2 起含候选块）；这里不写死数字，
    # 只要求与本地布局一致，避免每次升 schema 都要改这条断言。
    assert manifest["record_bytes"] == int(r2048_core.NUM_ACTIONS) * 0 + _record_bytes()
    # 批处理协议的意义就在这里：平均每批请求行数必须远大于 1。
    assert result.rows_per_forward > 20, f"平均每批只有 {result.rows_per_forward:.1f} 行，批处理失效"
    # 叶子缓存命中率：同一局面在搜索中被反复查询，命中率过低说明缓存没起作用。
    assert result.stats["leaf_cache_rate"] > 0.1, f"叶子缓存命中率过低：{result.stats['leaf_cache_rate']}"

    verified = _verify(data_dir)
    assert verified["ok"] is True
    assert verified["samples"] == SMALL["samples"]


def test_selfplay_reproducible_for_same_config(tmp_path: Path) -> None:
    """同配置同设备下重复运行必须产出相同数据。

    这是“这批数据是哪次实验产出的”这一问题的下限保证；跨设备或改动批处理参数
    就不在保证范围内（见模块文档与 ADR 0003）。
    """
    hashes = []
    for run in range(2):
        data_dir = tmp_path / f"run{run}"
        run_selfplay(_config(data_dir, seed=20261006), _model(), _device())
        manifest = json.loads((data_dir / "manifest.json").read_text())
        hashes.append([shard["sha256"] for shard in manifest["shards"]])

    assert hashes[0] == hashes[1], "相同配置两次自对弈的分片哈希必须一致"

    # 换种子必须换数据。
    other_dir = tmp_path / "other"
    run_selfplay(_config(other_dir, seed=999), _model(), _device())
    other = json.loads((other_dir / "manifest.json").read_text())
    assert [s["sha256"] for s in other["shards"]] != hashes[0], "不同种子应当产出不同数据"


def test_extension_exposes_protocol_versions() -> None:
    # 跨语言契约的版本号必须由 C++ 内核给出，Python 侧不做假定。
    assert int(r2048_core.RULES_VERSION) >= 1
    assert int(r2048_core.PROTOCOL_VERSION) >= 1
    assert int(r2048_core.NUM_ACTIONS) == 4
    assert float(r2048_core.SCORE_SCALE) == 1024.0
    # 动作顺序是硬契约：UP=0, RIGHT=1, DOWN=2, LEFT=3。
    assert (int(r2048_core.UP), int(r2048_core.RIGHT), int(r2048_core.DOWN), int(r2048_core.LEFT)) == (
        0,
        1,
        2,
        3,
    )


def test_selfplay_workers_share_one_batch(tmp_path: Path) -> None:
    """多进程跑谱：N 个 worker 写同一批，且 `--samples` 是**整批**目标。

    这条同时守住两件事：
    1. `run_selfplay_workers` 起的是 N 个真进程（每个单线程，ADR 0031 的可复现前提），
       它们通过 `--join` 写进同一个批次目录；
    2. 目标按整批判定——旧实现是每个进程各跑满一次，2 个 worker 会产出接近 2× 目标。
       上界 1.4× 正是用来把两种语义分开的：现在的超出量只有"在途的局"（达标后进行中的局
       必须走完，否则局号会出现空洞）。
    """
    if not os.environ.get("R2048_TEST_BINARY"):
        pytest.skip("未设置 R2048_TEST_BINARY：通过 make test 运行")
    data_dir = tmp_path / "workers-batch"
    target = 6000
    config = SelfplayConfig(
        data_dir=str(data_dir),
        batch_id="workers-test",
        samples=target,
        seed=4242,
        model_id="random-init",
        simulations=4,
        lanes=4,
        in_flight_per_lane=2,
        shard_samples=1000,
        max_batch=4096,
        search_threads=1,
        progress_every=0.0,
    )
    result = run_selfplay_workers(config, "", _device().type, 2)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert result.worker_exits == [0, 0], f"worker 退出码异常：{result.worker_exits}"
    assert result.samples == manifest["samples"], "监督进程的汇总必须与清单同源"
    assert manifest["state"] == "complete", "达标停车应当把状态记成 complete"
    assert manifest["samples"] >= target, "应当至少产出到目标"
    # 2 个 worker × 4 lane 的在途局 + 刷新滞后，远小于"每个进程各跑满一次"的两倍。
    assert manifest["samples"] <= target * 1.4, (
        f"批级总量 {manifest['samples']} 超出目标 {target} 太多——"
        "像是又变回了按进程计目标（每个进程各跑满一次）"
    )
