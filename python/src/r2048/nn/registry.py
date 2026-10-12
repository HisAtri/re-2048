"""网络定义的注册表。

**名字与结构分开：名字只是索引，超参一律从 metadata 读。** 具体是两条互不重叠的职责：

* **名字**（`VERSION_ID` 与它的别名）**只用于"按名字找定义"这一处查表**，以及给人看。
  代码不解析名字、不据它推导结构——**这条是硬线**：名字写错只是找不到定义（报错），
  绝不会静默建成另一个网络。
* **超参**（`architecture_params`）是构造网络的全部依据；配置可调整尺寸，正式发布新尺寸
  时要登记对应的正式代号，避免展示名与实际体量不符。

名字要**对人可读**：双塔正式代号 `smnet-b9c64dt` 表示总共 9 块、64 通道、双塔，
默认拓扑是共享 3 块 + 两塔各 3 块（ADR 0047）。换宽度本来就是一个新模型、就该有新名字，
所以"改宽度要改名"不是代价，而是正确行为（ADR 0026 记录命名分层与发布文件名语法）。

**同一份定义可以有多个名字**：正式别名（canonical）与历史 ID（`ALIASES`）。
历史产物记着旧名字，改不了也不该改，所以旧名字必须继续解析；而**新产物只写 canonical**。
测试版时期用日期流水号（`20261006001`）当名字，转正后 canonical 换成正式别名、日期号进别名表。

**读到不支持的 ID 就退出**：这是加载闸门的一部分（与 `input_encoding` / `score_scale` /
`rules_version` 同处判定）。理由与 ADR 0018 的"缺字段必须报错"一致——用一个不认识的定义
去解释一份权重，最坏情况不是抛异常，而是 `load_state_dict` 恰好成功（形状撞上），
得到一个"能跑但语义全错"的模型。

**一个定义模块必须提供**（一版一个模块，见 `docs/adr/0024`）：

1. `VERSION_ID`（正式别名）+ `ALIASES`（历史名字）：名字只用于查表，代码不解析它；
2. 一个 `nn.Module` 子类，沿用 `metadata.py` 的元数据生成方式（超参在签名里、存成同名属性、
   metadata 由 `definition_metadata()` 生成），并且：
   * `forward(boards) -> (value, logits)`：输入是原始指数棋盘 `(N, 16)` 整数张量，
     输出 `(N,)` 与 `(N, 4)`。**这段内部的一切都是定义自己的事**——层数、塔的对称性、
     输入编码、位置编码、注意力都在里面实现，外部不会多传一个参数；
   * 已发布的定义不允许改语义（改形状/改归一化都会让旧权重"加载成功但不再等价"）。
"""

from __future__ import annotations

from torch import nn

from r2048.nn.models.dual_tower import VERSION_ID as DUAL_TOWER_ID
from r2048.nn.models.dual_tower import DualTowerResNet
from r2048.nn.models.shared_resnet import VERSION_ID as SHARED_ID
from r2048.nn.models.shared_resnet import DualHeadResNet


class UnknownModelDefinition(ValueError):
    """checkpoint 的版本 ID 在本版本里没有对应的定义。"""


# canonical 名字 → 定义类。这张表就是"本版本支持哪些网络定义"的唯一清单：
# 加一个新定义在这里加一行；改宽度/深度**不需要**动它。
REGISTRY: dict[str, type[nn.Module]] = {
    DualHeadResNet.VERSION_ID: DualHeadResNet,
    DualTowerResNet.VERSION_ID: DualTowerResNet,
}
# 别名（旧 ID / 测试版日期号）→ canonical。查表用，不影响任何计算。
ALIAS_INDEX: dict[str, str] = {}


def supported_ids() -> tuple[str, ...]:
    """canonical 名字（对外展示与索引优先用它们）。"""
    return tuple(REGISTRY)


def known_names() -> dict[str, str]:
    """别名 → canonical 的全表（给人看"还有哪些旧名字能用"）。"""
    return dict(ALIAS_INDEX)


def canonical_id(name: str) -> str:
    """把任何被接受的名字（canonical 或别名）规约成 canonical。"""
    if name in REGISTRY:
        return name
    canonical = ALIAS_INDEX.get(name)
    if canonical is None:
        raise UnknownModelDefinition(
            f"不认识的模型定义名 {name!r}：本版本支持 {list(supported_ids())}"
            f"（别名：{sorted(ALIAS_INDEX)}）。这份权重通常来自更新的代码或另一条分支——"
            "要么用产出它的版本加载它，要么把定义搬进 nn/models/ 并在 nn/registry.py 里登记。"
        )
    return canonical


def resolve(version_id: str) -> type[nn.Module]:
    return REGISTRY[canonical_id(version_id)]


def build(version_id: str, params: dict | None = None) -> nn.Module:
    """按名字选定义、按参数构造。参数里的未知键直接报错（拼错的名字不该被忽略）。"""
    return resolve(version_id)(**(params or {}))


def _check_registry() -> None:
    """导入时自检：ID 不能为空、类与模块声明必须一致、注册表不能有重复。

    这类错误（复制一个定义时忘了改 ID）在别处不会暴露：两个类抢同一个 ID 时，
    注册表只会安静地少一个定义。
    """
    pairs = ((SHARED_ID, DualHeadResNet), (DUAL_TOWER_ID, DualTowerResNet))
    for module_id, definition in pairs:
        if not module_id:
            raise RuntimeError(f"{definition.__name__} 的 VERSION_ID 为空")
        if definition.VERSION_ID != module_id:
            raise RuntimeError(
                f"{definition.__name__} 的类名 {definition.VERSION_ID!r} 与模块声明的 {module_id!r} 不一致"
            )
    if len(REGISTRY) != len(pairs):
        raise RuntimeError(f"REGISTRY 里有重复的 canonical 名字：{sorted(REGISTRY)}")
    # 别名必须唯一，且不能与任何 canonical 名字撞车：两个名字指同一份定义是允许的，
    # 一个名字指两份定义则是灾难（调度会随书写顺序变）。
    for _module_id, definition in pairs:
        for alias in definition.ALIASES:
            if alias in REGISTRY:
                raise RuntimeError(f"别名 {alias!r} 与 canonical 名字撞车")
            if ALIAS_INDEX.get(alias, definition.VERSION_ID) != definition.VERSION_ID:
                raise RuntimeError(f"别名 {alias!r} 同时指向两份定义")
            ALIAS_INDEX[alias] = definition.VERSION_ID


_check_registry()
