"""定义契约的机械部分：超参怎么读、metadata 怎么写全。

单独成模块是为了让定义模块只依赖契约、不依赖注册表（否则 `models/__init__` 与各定义
互相导入）。这里没有任何网络结构知识，只有三条约定：

1. **超参写在 `__init__` 签名里**（都有默认值），所以 `default_params()` 能读出来；
2. **每个超参存成同名实例属性**，所以 `instance_params()` 能读回来——漏存会在构造时
   报错，而不是某天用默认值静默重建出另一个网络；
3. **metadata 由 `definition_metadata()` 生成**，于是 `architecture` / `architecture_params`
   / `input_encoding` / `score_scale` 一定齐全，不必让每个定义各自记得写。
"""

from __future__ import annotations

import inspect
from typing import Any

from torch import nn

from r2048.nn.encode import INPUT_ENCODING

# 与 cpp/core/version.h 的 kScoreScale 同源：写进 metadata 并在加载时校验。
SCORE_SCALE = 1024.0

# 权重在文件里的精度。**现在就记下来**（不是等用到才加）：一旦出现量化对齐过的产物，
# "这份权重是不是被降精度过"只能靠记录判断，而事后补记对已有文件是不可能的（ADR 0026）。
# 目前恒为 float32；加载侧把缺失当 float32（历史文件全都没这个键），
# 但**认得出别的值就拒绝**——`load_state_dict` 会静默转换 dtype，静默比报错贵得多。
WEIGHTS_DTYPE = "float32"


def default_params(definition: type[nn.Module]) -> dict[str, Any]:
    """从 `__init__` 签名读超参默认值：超参只在签名里写一次，metadata 与重建都从这里来。"""
    signature = inspect.signature(definition.__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def instance_params(module: nn.Module) -> dict[str, Any]:
    """读回一个实例实际的超参值（要求每个超参都是同名实例属性）。"""
    params: dict[str, Any] = {}
    for name in default_params(type(module)):
        if not hasattr(module, name):
            raise AttributeError(
                f"{type(module).__name__} 没有把超参 {name!r} 存成同名属性："
                "定义必须写成 self.<参数名> = <参数名>，metadata 才能自动记全"
            )
        params[name] = getattr(module, name)
    return params


def definition_metadata(module: nn.Module, version_id: str, model_id: str, parent: str = "") -> dict:
    """各定义统一的 metadata：字段由这里保证齐全，定义自己不必逐个想起。"""
    return {
        "architecture": version_id,
        "architecture_params": instance_params(module),
        "weights_dtype": WEIGHTS_DTYPE,
        "input_encoding": INPUT_ENCODING,
        "score_scale": SCORE_SCALE,
        "model_id": model_id,
        "parent": parent,
    }
