"""协议变量的清单化：写出方与评分侧必须对同一份清单负责。

为什么值得单独一个测试文件：协议字段漂移**不报错**。写出方少写一个字段、
或者读者给它补一个默认值，都只会让两个真正不同的协议在评分线上静默合并，
下游拿到的数字看起来完全正常。所以这里把四件事钉成测试：

1. 缺字段必须拒收（不报错就等于用默认值伪造协议身份）；
2. 整型与浮点必须归一化到同一个哈希（`json.dumps` 对 1 与 1.0 字节不同）；
3. 非整浮点不能被截断（`int(1.5)` 会静默变成 1，那就是另一个协议）；
4. `eval_seed` 不进协议、但参与去重键。

第 5 条是跨语言的：C++ `r2048 evaluate` 写出的真实记录必须能通过 Python 侧的校验。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from r2048.protocol import PROTOCOL_FIELDS, build_protocol, protocol_hash
from r2048.workflows.rating import game_key, normalize_row

# 一条完整的纯 rollout 记录：纯 C++ 线只有这几项有值，其余声明 null。
PURE_ROW = {
    "track": "mcts",
    "rules_version": 1,
    "protocol_version": 1,
    "simulations": 64,
    "temperature_moves": 0,
    "c_puct": None,
    "temperature": None,
    "inference_precision": None,
    "lanes": None,
    "in_flight": None,
    "max_batch": None,
    "cache_entries": None,
    "device": None,
}


def _row(**overrides) -> dict:
    return {**PURE_ROW, "episode": 0, "label": "mcts64", "seed": 7, "score": 100, **overrides}


def test_missing_field_is_refused() -> None:
    partial = {name: PURE_ROW[name] for name in ("track", "rules_version")}
    with pytest.raises(ValueError) as error:
        build_protocol(partial)
    assert "缺少协议字段" in str(error.value)
    # 报错必须点名缺了哪些字段，否则面对十几个字段只能靠数。
    assert "c_puct" in str(error.value)


def test_null_means_not_applicable_not_unknown() -> None:
    """`null` 与缺字段是两件事：前者是“该策略没有这个变量”的声明，后者是漏洞。"""
    explicit_null = build_protocol(PURE_ROW)
    assert explicit_null["c_puct"] is None
    without_key = {name: value for name, value in PURE_ROW.items() if name != "c_puct"}
    with pytest.raises(ValueError, match="c_puct"):
        build_protocol(without_key)


def test_int_and_float_normalize_to_same_hash() -> None:
    """同一个协议值不因为写出方的语言不同而劈成两条线。"""
    as_int = build_protocol({**PURE_ROW, "c_puct": 1, "temperature": 0, "simulations": 64})
    as_float = build_protocol({**PURE_ROW, "c_puct": 1.0, "temperature": 0.0, "simulations": 64.0})
    assert protocol_hash(as_int) == protocol_hash(as_float)


def test_fractional_value_is_not_truncated() -> None:
    with pytest.raises(ValueError):
        build_protocol({**PURE_ROW, "simulations": 64.5})


def test_eval_seed_is_not_part_of_protocol() -> None:
    """seed 是同一协议下的重复抽样：换 seed 不换协议，但换局（去重键必须区分）。"""
    first = normalize_row(_row(seed=7))
    second = normalize_row(_row(seed=8))
    assert first["protocol_hash"] == second["protocol_hash"]
    assert first["track"] == "search" and first["source_track"] == "mcts"
    assert game_key(first) != game_key(second)
    assert first["eval_seed"] == 7


def test_protocol_change_splits_the_line() -> None:
    """批组成进协议：这正是 ADR 0017 要求的结构性隔离。"""
    assert protocol_hash(build_protocol({**PURE_ROW, "simulations": 64})) != protocol_hash(
        build_protocol({**PURE_ROW, "simulations": 128})
    )


def test_cpp_writer_satisfies_the_contract(tmp_path: Path) -> None:
    """C++ 写出的逐局记录必须零缺失字段——防的是「改了 C++ 写出方忘记改 Python 读取方」。"""
    cli = os.environ.get("R2048_CLI")
    if not cli or not Path(cli).exists():
        pytest.skip("未设置 R2048_CLI：通过 make test 运行，或先 make build")
    jsonl = tmp_path / "random.jsonl"
    subprocess.run(
        [
            cli,
            "evaluate",
            "--policy",
            "random",
            "--games",
            "4",
            "--seed",
            "11",
            "--label",
            "random",
            "--jsonl",
            str(jsonl),
            "--quiet",
        ],
        check=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    for row in rows:
        protocol = build_protocol(row)
        assert set(protocol) == set(PROTOCOL_FIELDS)
        # 纯 C++ 线只能这样声明：有值的只有规则/协议版本与它的两个旋钮。
        assert protocol["track"] == "random"
        assert protocol["simulations"] is None and protocol["device"] is None
        # 必须能被评分侧直接消费（不抛异常即通过校验）。
        normalized = normalize_row(row)
        assert normalized["track"] == "pure"
        assert normalized["protocol"] == protocol
