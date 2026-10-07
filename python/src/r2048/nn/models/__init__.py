"""网络定义的注册表。

**版本 ID 是索引，不是规格。** 它只回答"这份权重要用哪个定义来构造"，
**不用来反推结构或超参**：超参一律从 checkpoint metadata 的 `architecture_params` 读，
ID 只是给人一眼看出"这是哪一族网络"的名字。这条分界很重要：

* 把结构写进 ID（旧版那种 `s3p3v3-c64`）看起来信息量更大，但它逼着 ID 随着任何宽度/深度
  调整一起改名——每换一次宽度，ID、代码分派、产物目录、评分表都要跟着动，而"这具权重
  当初是什么结构"仍然只能靠 ID 反推，算错就静默建成另一个网络（旧版 `create_model()`
  忽略 `shared_blocks/tower_blocks`、永远按 3/3 建网，就是这个坑，写在它自己的注释里）。
* 把超参放在 metadata、把名字留给 ID，则"改宽度/改深度"是**纯配置行为**（改一处参数），
  ID 只在一个**新拓扑形状**出现时才新增。

**读到不支持的 ID 就退出**：这是加载闸门的一部分（与 `input_encoding` / `score_scale` /
`rules_version` 同处判定）。理由与 ADR 0018 的"缺字段必须报错"一致——用一个不认识的定义
去解释一份权重，最坏情况不是抛异常，而是 `load_state_dict` 恰好成功（形状撞上），
得到一个"能跑但语义全错"的模型。

**一个定义模块必须提供**（一版一个模块，见 `docs/adr/0024`）：

1. `VERSION_ID`：唯一版本 ID，写进 metadata，**一旦有产物依赖它就不再改名**；
2. 一个 `nn.Module` 子类，满足 `contract.py` 里的三条约定（超参在签名里、存成同名属性、
   metadata 由 `definition_metadata()` 生成），并且：
   * `forward(encoded) -> (value, logits)`：输入是 `encode_boards` 的输出 `(N, 25, 4, 4)`，
     输出 `(N,)` 与 `(N, 4)`。**这段内部的一切都是定义自己的事**——层数、塔的对称性、
     位置编码、注意力都在里面实现，外部不会多传一个参数；
   * 已发布的定义不允许改语义（改形状/改归一化都会让旧权重"加载成功但不再等价"）。
"""

from __future__ import annotations

from torch import nn

from r2048.nn.models.contract import (  # noqa: F401 - 对外转发契约工具
    SCORE_SCALE,
    default_params,
    definition_metadata,
    instance_params,
)
from r2048.nn.models.dual_tower import VERSION_ID as DUAL_TOWER_ID
from r2048.nn.models.dual_tower import DualTowerResNet
from r2048.nn.models.shared_resnet import VERSION_ID as SHARED_ID
from r2048.nn.models.shared_resnet import DualHeadResNet


class UnknownModelDefinition(ValueError):
    """checkpoint 的版本 ID 在本版本里没有对应的定义。"""


# ID → 定义类。这张表就是"本版本支持哪些网络定义"的唯一清单：
# 加一个新版本在这里加一行；改宽度/深度**不需要**动它。
REGISTRY: dict[str, type[nn.Module]] = {
    DualHeadResNet.VERSION_ID: DualHeadResNet,
    DualTowerResNet.VERSION_ID: DualTowerResNet,
}


def supported_ids() -> tuple[str, ...]:
    return tuple(REGISTRY)


def resolve(version_id: str) -> type[nn.Module]:
    definition = REGISTRY.get(version_id)
    if definition is None:
        raise UnknownModelDefinition(
            f"不认识的模型版本 ID {version_id!r}：本版本支持 {list(supported_ids())}。"
            "这份权重通常来自更新的代码或另一条分支——要么用产出它的版本加载它，"
            "要么把那个定义搬进 nn/models/ 并在 REGISTRY 里登记。"
        )
    return definition


def build(version_id: str, params: dict | None = None) -> nn.Module:
    """按 ID 选定义、按参数构造。参数里的未知键直接报错（拼错的名字不该被忽略）。"""
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
                f"{definition.__name__} 的类 ID {definition.VERSION_ID!r} 与模块 ID {module_id!r} 不一致"
            )
    if len(REGISTRY) != len(pairs):
        raise RuntimeError(f"REGISTRY 里有重复的版本 ID：{sorted(REGISTRY)}")


_check_registry()
