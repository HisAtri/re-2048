"""模型定义的注册表与加载闸门。

这里守的是"模型迭代的解耦"：加一个新版本应当只动 `nn/models/` 里的东西，而**不会**
悄悄影响到训练、自对弈、评测或旧产物。所以判据围绕三件事：

* 每个登记在册的定义都能**自己往返**（构造 → metadata → 重建 → 前向）；
* 版本 ID 的三种异常（不认识 / 缺失 / 类与模块不一致）都必须报错，不能静默兜底；
* 超参通过配置能真的传到模型上（否则"改通道数"就只是句空话）。
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest
import torch

from r2048.config import build_train_config
from r2048.nn.encode import INPUT_CHANNELS
from r2048.nn.model import (
    SHARED_ARCHITECTURE,
    UnknownModelDefinition,
    create_model,
    load_checkpoint,
    supported_ids,
)
from r2048.nn.models import REGISTRY, build, default_params, instance_params


def encoded_batch(count: int = 3) -> torch.Tensor:
    return torch.zeros(count, INPUT_CHANNELS, 4, 4)


@pytest.mark.parametrize("version_id", sorted(REGISTRY))
def test_every_definition_round_trips(version_id: str) -> None:
    """在册的每个定义：构造 → metadata → 用 metadata 重建 → 前向。

    这条测试对**将来新增的定义自动生效**，是"加架构只动 nn/models/"的守卫：
    漏记超参（用默认值重建出另一个网络）、前向形状不对、metadata 字段不全，
    都会在这里失败，而不是等到训练出问题才发现。
    """
    model = build(version_id)
    metadata = model.metadata("round-trip")
    assert metadata["architecture"] == version_id
    assert metadata["architecture_params"] == instance_params(model)

    rebuilt = create_model(metadata)
    assert type(rebuilt) is type(model)
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    actual = {name: tuple(t.shape) for name, t in rebuilt.state_dict().items()}
    assert actual == expected, "按 metadata 重建出的参数形状必须与原模型逐一相同"
    # 重建出来的模型要能直接吃旧权重（strict）：形状对了但键名不同的话，这里会失败。
    rebuilt.load_state_dict(model.state_dict())

    with torch.no_grad():  # 前向不带梯度：这里只关心形状与量纲
        value, logits = rebuilt(encoded_batch())
    assert value.shape == (3,) and logits.shape == (3, 4)
    assert torch.isfinite(value).all() and torch.isfinite(logits).all()
    # 本项目的值语义：价值是**归一化**的分数（score / score_scale），不是原始分数。
    assert float(value.abs().max()) < 100.0


def test_hyperparameters_are_recorded_not_inferred() -> None:
    """超参必须逐个记进 metadata（ID 只选定义、不反推结构）。

    这条正是"改了宽度但 metadata 没记 → 用默认值静默重建"那个坑的守卫：
    旧版 `create_model()` 忽略 `shared_blocks/tower_blocks`，永远按 3/3 建网。
    """
    for version_id in sorted(REGISTRY):
        signature = default_params(REGISTRY[version_id])
        assert signature, f"{version_id} 没有可记录的构造超参（定义必须把超参写在 __init__ 签名里）"
        assert "head_width" in signature, "head_width 也必须进参数表，否则重建时静默回落默认值"
        assert "groups" in signature and signature["groups"] == 8, "分组数必须可记录且默认值不变"
        # 定义必须把每个超参存成同名属性（contract.instance_params 靠它读回来）。
        model = build(version_id)
        for name in signature:
            assert getattr(model, name) == signature[name]

    non_default = build(SHARED_ARCHITECTURE, {"channels": 32, "blocks": 2, "head_width": 64})
    metadata = non_default.metadata("custom")
    assert metadata["architecture_params"] == {"channels": 32, "blocks": 2, "head_width": 64, "groups": 8}
    rebuilt = create_model(metadata)
    assert rebuilt.stem[0].out_channels == 32
    assert rebuilt.value_head[1].out_features == 64
    with pytest.raises(RuntimeError, match="mat|size mismatch"):
        # 换尺寸之后旧权重当然装不上：这正是"换结构 = 换一条训练线"在代码上的样子。
        rebuilt.load_state_dict(build(SHARED_ARCHITECTURE).state_dict())


def test_unknown_definition_is_refused() -> None:
    """不认识的 ID 必须退出，且消息里要说清"支持什么"——否则读者只能去翻代码。"""
    with pytest.raises(UnknownModelDefinition) as error:
        create_model({"architecture": "dual_tower_v9"})
    message = str(error.value)
    assert "dual_tower_v9" in message
    for version_id in supported_ids():
        assert version_id in message

    # 加载路径上同样要拦住：ID 不支持的权重不能"试试看能不能装进去"。
    payload = {"checkpoint_version": 1, "model": {}, "metadata": {"architecture": "nope_v1"}}
    path = Path("/tmp") / "r2048-unknown-model.pt"
    torch.save(payload, path)
    with pytest.raises(UnknownModelDefinition, match="nope_v1"):
        load_checkpoint(str(path), torch.device("cpu"))

    # 而且**缺失** ID 也算失败：没有 ID 就无法确认该用哪个定义解释这份权重。
    torch.save({"checkpoint_version": 1, "model": {}, "metadata": {}}, path)
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(str(path), torch.device("cpu"))
    path.unlink()


def test_registry_self_check_guards_copy_paste() -> None:
    """自检要能抓住"复制一个定义忘了改 ID"：类与模块的 ID 必须一致。"""
    from r2048.nn.models import dual_tower, shared_resnet

    for module in (dual_tower, shared_resnet):
        assert module.VERSION_ID == getattr(module, REGISTRY[module.VERSION_ID].__name__).VERSION_ID
        # 每个定义的 __init__ 必须只暴露"有默认值的具名参数"：**kwargs 会让超参记不全。
        for parameter in inspect.signature(REGISTRY[module.VERSION_ID].__init__).parameters.values():
            assert parameter.kind is not inspect.Parameter.VAR_KEYWORD
    assert len(set(REGISTRY)) == len(REGISTRY) == 2, "两个已发布定义必须都在册且 ID 唯一"


def test_architecture_conflict_is_precise() -> None:
    """从 checkpoint 初始化时的结构检查：只拦**真的不同**的，同一条线的配置照常带上。

    过严会挡住正常用法（一份训练线的 YAML 会被反复带上，包括"从上一轮模型接着训"），
    过松则会让"我明明改了宽度"变成静默忽略——所以判据落在"解析之后比一比"。
    """
    from r2048.nn.model import architecture_conflict

    metadata = build(SHARED_ARCHITECTURE).metadata("m")
    # 不指定、完全相同、只写一部分（其余靠默认补齐）都不算冲突。
    assert architecture_conflict("", {}, metadata) is None
    assert architecture_conflict(SHARED_ARCHITECTURE, {}, metadata) is None
    assert architecture_conflict("", {"channels": 64}, metadata) is None
    assert architecture_conflict(SHARED_ARCHITECTURE, {"channels": 64, "blocks": 6}, metadata) is None

    # 真的不同必须指名道姓（"哪一项、原为多少、请求多少"）。
    changed = architecture_conflict("", {"channels": 128}, metadata)
    assert changed and "channels" in changed and "64" in changed and "128" in changed
    swapped = architecture_conflict("dual_tower_v1", {}, metadata)
    assert swapped and SHARED_ARCHITECTURE in swapped and "dual_tower_v1" in swapped

    # 旧 checkpoint 的 architecture_params 只有一部分：补默认值之后再比，不能误报。
    legacy = dict(metadata)
    legacy["architecture_params"] = {"channels": 64, "blocks": 6}
    assert architecture_conflict("", {"channels": 64, "blocks": 6}, legacy) is None
    assert architecture_conflict("", {"head_width": 128}, legacy) is not None


def test_architecture_params_reach_the_model_from_config(tmp_path: Path) -> None:
    """`配置 → 模型`的通路：YAML 里写的结构与超参必须真的落到 TrainConfig 上。

    这条以前是断的（`create_model` 只拿得到一个名字、永远用默认尺寸），
    而"改通道数/改层数"正是接下来一定会做的事。
    """
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "datasets:\n"
        "  - path: data/batch\n"
        "train:\n"
        "  architecture: dual_tower_v1\n"
        "  architecture_params:\n"
        "    channels: 96\n"
        "    shared_blocks: 2\n"
        "    tower_blocks: 4\n"
    )
    args = argparse.Namespace(command="train", config=str(config_path), data="", output="run")
    config = build_train_config(args)
    assert config.architecture == "dual_tower_v1"
    assert config.architecture_params == {"channels": 96, "shared_blocks": 2, "tower_blocks": 4}

    model = create_model(
        {"architecture": config.architecture, "architecture_params": config.architecture_params}
    )
    assert model.channels == 96 and model.shared_blocks == 2 and model.tower_blocks == 4
    # 参数表里没提的字段回落到签名默认值，而不是报错或变成 0。
    assert model.head_width == 256 and model.groups == 8
