"""训练配置的 YAML 层：声明"用哪些数据、怎么训"。

训练配方沿用 ADR 0021；WebUI 固定测评配置独立位于 configs/assessment/，
由 workflows.assessment 读取（ADR 0052），不混入训练配方目录列表。

优先级 CLI > YAML > 默认值。合并靠 argparse 的 `SUPPRESS` 默认值实现：只有**显式给出**的
参数才会出现在 namespace 里，所以"用户到底指定了什么"是精确可知的，不必拿值去和默认值
比对（那样无法区分"用户就是要这个默认值"和"用户没提"）。
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from pathlib import Path

import yaml

from r2048.nn.training import DatasetSpec, TrainConfig

TRAIN_SECTION = "train"
DATASETS_SECTION = "datasets"
_SECTIONS = (DATASETS_SECTION, TRAIN_SECTION)
_DATASET_KEYS = ("path", "weight", "limit")

# 需要取反的布尔开关：argparse 用 `--no-xxx`，配置字段是正向的 `xxx`。
NEGATED_FLAGS = {"no_augment": "augment", "no_channels_last": "channels_last"}
# 不映射到配置字段的参数：`--data` 是 datasets 的简写，另两个不是训练参数。
NON_CONFIG_ARGS = ("command", "config", "data", "performance_profile")
# 回调是运行期接进来的（见 training.config_payload），配置文件与 CLI 都不该设它。
UNSETTABLE_FIELDS = ("on_progress",)


def load_config(path: str | Path) -> dict:
    """读并校验配置文件的结构，返回原始 dict。

    只在这里做"结构对不对"的判断（有哪些小节、键名认不认识），字段值本身留给
    `TrainConfig`：配置层不该重复一遍每个字段的类型规则。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到训练配置文件 {path}")
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} 的顶层必须是映射（datasets / train），实际是 {type(data).__name__}")
    unknown = [key for key in data if key not in _SECTIONS]
    if unknown:
        # 拼错小节名（例如 dataset:）如果不报错，就会安静地变成"没有数据来源"，
        # 然后在别处报一个与之无关的错。
        raise ValueError(f"{path} 顶层出现未知小节 {unknown}，只支持 {list(_SECTIONS)}")
    return data


def parse_datasets(section) -> list[DatasetSpec]:
    """把 `datasets:` 小节变成 DatasetSpec 列表。"""
    if not isinstance(section, list) or not section:
        raise ValueError(f"{DATASETS_SECTION} 必须是非空列表，每项形如 {{path, weight, limit}}")
    specs = []
    for index, entry in enumerate(section):
        if not isinstance(entry, dict):
            raise TypeError(f"{DATASETS_SECTION}[{index}] 必须是映射，实际是 {type(entry).__name__}")
        unknown = [key for key in entry if key not in _DATASET_KEYS]
        if unknown:
            raise ValueError(
                f"{DATASETS_SECTION}[{index}] 出现未知键 {unknown}，只支持 {list(_DATASET_KEYS)}"
            )
        if not entry.get("path"):
            raise ValueError(f"{DATASETS_SECTION}[{index}] 缺少 path")
        specs.append(
            DatasetSpec(
                path=str(entry["path"]),
                weight=float(entry.get("weight", 1.0)),
                limit=int(entry.get("limit", 0)),
            )
        )
    return specs


def check_train_section(path: str | Path, section: dict) -> None:
    """`train:` 里的键必须是真实的配置字段：拼错的键被静默忽略等于"改了半天没生效"。"""
    if not isinstance(section, dict):
        raise TypeError(f"{path} 的 {TRAIN_SECTION} 小节必须是映射")
    allowed = {field.name for field in fields(TrainConfig)} - set(UNSETTABLE_FIELDS)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{path} 的 {TRAIN_SECTION} 小节出现未知字段 {unknown}")


# 字段类型的期望值：由 TrainConfig 的默认值推出来，不再手写一份清单。
_TYPE_NAMES = {bool: "布尔", int: "整数", float: "浮点", str: "字符串"}
# YAML 1.1 的坑：`1e-3` 不是浮点数，而是**字符串**（必须写成 `1.0e-3`）。
# 这类错误读取时不会报错，会一路走到 torch 的优化器构造里才炸，报的是
# "'<=' not supported between instances of 'float' and 'str'"——与配置毫无关系。
FLOAT_HINT = "（YAML 1.1 里 `1e-3` 是字符串，要写成 `1.0e-3`）"


def train_section_payload(path: str | Path, section: dict) -> dict:
    """校验并规范化 `train:` 小节，返回可直接构造 TrainConfig 的 dict。

    做两件事，都属于"配置层该管的"：
    * **拒绝拼错的键**（未知字段）：被静默忽略的键等于"改了半天没生效"；
    * **拒绝类型不对的值**：YAML 不会替你检查，而错误的类型会在很深的地方以
      看不懂的形式炸出来（见 FLOAT_HINT）。
    合法的整数在需要浮点的字段上会被接受并转换（`lr: 3` 是一个自然的写法）。
    """
    check_train_section(path, section)
    payload: dict = {}
    for field in fields(TrainConfig):
        if field.name not in section:
            continue
        value = section[field.name]
        if field.name == "datasets":
            # 数据集有专门的小节（要写权重与上限），写在 train 里只会被当成原始 dict 传下去。
            raise ValueError(f"{path}: 数据集请写在顶层 {DATASETS_SECTION} 小节，不要放在 {TRAIN_SECTION} 里")
        default = field.default
        if default is MISSING:
            # 有 default_factory 的字段（即 dict/list：architecture_params）。容器类型在这里
            # 挡一下，内容交给构造函数——TrainConfig 会逐个关键字校验，未知键会直接报错。
            if not isinstance(value, (dict, list)):
                raise TypeError(
                    f"{path} 的 {field.name} 应当是映射或列表，收到 {type(value).__name__} {value!r}"
                )
            payload[field.name] = value
            continue
        if default is None:  # 运行期字段（回调），已由 check_train_section 拦下
            continue
        if isinstance(value, bool) and type(default) is not bool:
            raise ValueError(
                f"{path} 的 {field.name} 应当是{_TYPE_NAMES[type(default)]}，收到布尔值 {value!r}"
            )
        if type(default) is float and isinstance(value, int):
            payload[field.name] = float(value)
            continue
        if not isinstance(value, type(default)):
            hint = FLOAT_HINT if type(default) is float and isinstance(value, str) else ""
            raise TypeError(
                f"{path} 的 {field.name} 应当是{_TYPE_NAMES[type(default)]}，"
                f"收到 {type(value).__name__} {value!r}{hint}"
            )
        payload[field.name] = value
    return payload


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    """把 `--config` / `--data` / 命令行合并成一个 TrainConfig。

    `--data` 与文件里的 `datasets` 同时给出时以 `--data` 为准（CLI > YAML）：编排里
    常见"用同一条训练线的参数，但换成刚跑出来的数据"。
    """
    file_config: dict = {}
    if getattr(args, "config", ""):
        config_path = Path(args.config)
        file_config = load_config(config_path)
        payload: dict = train_section_payload(config_path, file_config.get(TRAIN_SECTION, {}))
    else:
        payload = {}

    cli_data = getattr(args, "data", "")
    if cli_data:
        payload["datasets"] = [DatasetSpec(path=cli_data)]
    elif DATASETS_SECTION in file_config:
        payload["datasets"] = parse_datasets(file_config[DATASETS_SECTION])

    for name, value in vars(args).items():
        if name in NON_CONFIG_ARGS:
            continue
        target = NEGATED_FLAGS.get(name, name)
        payload[target] = (not value) if name in NEGATED_FLAGS else value

    if not payload.get("datasets"):
        raise SystemExit("必须给出训练数据：--data <数据批>，或在 --config 里写 datasets")
    from r2048.performance import resolve

    defaults = TrainConfig()
    resume_path = Path(payload.get("output", "")) / "last.pt"
    restored = None
    if payload.get("resume") and resume_path.exists():
        from r2048.nn.model import read_checkpoint_payload

        restored = read_checkpoint_payload(str(resume_path)).get("config", {})
        if "batch_size" not in payload:
            payload["batch_size"] = restored["batch_size"]
    effective, performance = resolve(
        "train",
        {
            "device": payload.get("device", defaults.device),
            "batch_size": payload.get("batch_size"),
            "mode": payload.get("mode", defaults.mode),
            "freeze_value": payload.get("freeze_value", False),
        },
        payload.get("init_checkpoint", ""),
        recipe=payload,
        mode=getattr(args, "performance_profile", "auto"),
    )
    payload["batch_size"] = effective["batch_size"]
    payload["performance"] = restored.get("performance", performance) if restored else performance
    return TrainConfig(**payload)
