"""评测协议：一条逐局记录要被后人正确解释，必须带哪些变量。

本模块是协议字段的**唯一事实来源**：评测写出方按 `build_protocol` 声明，
评分侧按同一份清单校验并哈希。两侧各写一遍清单是行不通的——漂移一次，
两次不同设置的评测就会静默合并成一条评分线。

两条规则，都是被真实误判换来的（ADR 0016 决策 4/5/6、ADR 0017）：

* **缺字段直接报错，不用默认值兜底。** 旧实现写 `row.get("c_puct", 1.5)`：
  一次用 c_puct=2.5 跑的评测因为没记录该字段，被当成 1.5 写进哈希，
  于是两个真正不同的协议在评分线上合并。这种错误不报错，只给出一个无意义的数字。
* **值为 null 表示“该策略没有这个变量”**（随机策略没有 c_puct），
  与“缺字段”是两件不同的事。必须显式声明 null，不能省略。

字段清单为什么是这些：一个量能被加进来，必须同时满足两条——
改动它会改变逐局结果，且它不能被当成抽样误差吸收（同参数重复跑逐局一致，
说明给定配置是确定的，所以差异是**配置**依赖）。判据见 ADR 0017。
"""

from __future__ import annotations

import hashlib
import json

# 字段 → 归一化类型。
#
# 整型与浮点必须分开：`json.dumps` 对 1 与 1.0 给出不同的字节，而它们是同一个
# 协议值。不归一化的话，同一个旋钮按写出方（C++ 的 Json 只支持整数、Python 写浮点）
# 劈成两条永不合并的评分线。
PROTOCOL_FIELDS: dict[str, type] = {
    # 产出这条记录的策略：random / mcts / nn_value / nn_policy / nn_mcts。
    # 它进哈希而不是只当标签，是为了让“同一策略”成为可比性的前提之一：
    # 两个策略即使在同一个评分线（track）上，也不该共用同一份协议身份。
    "track": str,
    # 规则语义与动作顺序契约，C++ 的 kRulesVersion。数据批靠它拒绝用错误语义解释旧数据。
    "rules_version": int,
    # 实现语义契约，C++ 的 kProtocolVersion。**没有单独入口的实现常量都挂在它上面**：
    # 纯 rollout 的探索常数 `RolloutMctsConfig::exploration`、rollout 策略本身、
    # 批调度形式，改动它们都会改变逐局结果，但它们不是配置项，没有别的地方可记。
    # 因此改这些常量时必须同时抬高 kProtocolVersion，否则评分线会跨语义合并。
    "protocol_version": int,
    # 搜索预算。纯网络 track 没有搜索，声明 null（不是 0：0 是“搜了 0 次”这个合法值，
    # 而 null 说的是“这个策略根本没有搜索预算这个变量”）。
    "simulations": int,
    "c_puct": float,
    "temperature": float,
    "temperature_moves": int,
    # 推理精度：网络前向实际用的 dtype。目前恒为 float32（没有开启降精度的入口），
    # 记录它是为了将来开了 bf16 之后，两批数据不会带着同一个协议身份混在一起。
    "inference_precision": str,
    # NN 路径的批组成。实测能改变逐局结果，通道是在途模拟的调度切分，不是浮点
    # （ADR 0017）；纯 C++ 路径没有这些东西，全部声明 null。
    "lanes": int,
    "in_flight": int,
    "max_batch": int,
    "cache_entries": int,
    # 计算设备。CPU 与 GPU 在同参数下逐局结果不同（浮点实现不同），因此属于协议。
    "device": str,
}

# 明确不属于协议的字段，列在这里是为了防止“感觉它该进来”而反复讨论：
#   eval_seed —— 同一协议下的重复抽样，它的位置是评分池的去重键 game_key。
#   games    —— 样本量，只影响区间宽度（同一 seed 加大局数时前 N 局逐局不变）。
#   workers  —— 纯 C++ 线的并发数，实测逐字节一致，只是性能参数。


def _as_int(value: object) -> int:
    """整型转换：拒绝非整的浮点，避免 1.5 被截断成 1 而静默改变协议身份。"""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{value!r} 不是整数")
    return int(value)  # type: ignore[arg-type]


_CASTERS = {int: _as_int, float: float, str: str}


def build_protocol(row: dict) -> dict:
    """从一条逐局记录里取出协议变量，缺字段或类型不符都直接失败。

    返回的 dict 已归一化：整型就是整型、浮点就是浮点，与写出方的语言无关。
    """
    missing = [name for name in PROTOCOL_FIELDS if name not in row]
    if missing:
        raise ValueError(
            "逐局记录缺少协议字段："
            + "、".join(missing)
            + f"（写出方必须声明全部 {len(PROTOCOL_FIELDS)} 项，该策略没有的变量写 null）。"
            "协议清单见 r2048.protocol；协议收口之前采集的记录无法补齐，只能重跑。"
        )
    protocol: dict = {}
    for name, kind in PROTOCOL_FIELDS.items():
        value = row[name]
        if value is None:
            protocol[name] = None
            continue
        try:
            protocol[name] = _CASTERS[kind](value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"协议字段 {name} 的值 {value!r} 不是 {kind.__name__}") from error
    return protocol


def protocol_hash(protocol: dict) -> str:
    payload = json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def protocol_diff(reference: dict | None, other: dict) -> dict:
    """两组协议之间真正不同的字段（含“一方有值、另一方为 null”）。

    存在的理由：评分表要能说清“这两组只是模拟数不同、其余测量条件一致”，
    而不是让人去比对两个 16 进制哈希。
    """
    if not reference:
        return {}
    return {
        name: [reference.get(name), other.get(name)]
        for name in PROTOCOL_FIELDS
        if reference.get(name) != other.get(name)
    }
