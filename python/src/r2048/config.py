"""训练配置的 YAML 层：声明"用哪些数据、怎么训"，别的环节仍然只有 CLI。

为什么只有训练有配置文件（见 ADR 0021）：

* 跑谱/评测/评分的参数是**这一次动作**的属性（跑多少局、什么协议、评哪条 track），
  每次都不同；写进文件只会让"这次到底跑了什么"多一层间接，而它们也不需要被长期引用。
* 训练的参数是**一条训练线**的属性：它要被反复引用（迭代编排、续跑闸门、checkpoint 里
  的 `config_digest`），值得有一个可命名、可进版本控制的实体。

优先级 CLI > YAML > 默认值。合并靠 argparse 的 `SUPPRESS` 默认值实现：只有**显式给出**的
参数才会出现在 namespace 里，所以"用户到底指定了什么"是精确可知的，不必拿值去和默认值
比对（那样无法区分"用户就是要这个默认值"和"用户没提"）。
"""

from __future__ import annotations

import argparse
from dataclasses import fields
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
NON_CONFIG_ARGS = ("command", "config", "data")
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


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    """把 `--config` / `--data` / 命令行合并成一个 TrainConfig。

    `--data` 与文件里的 `datasets` 同时给出时以 `--data` 为准（CLI > YAML）：编排里
    常见"用同一条训练线的参数，但换成刚跑出来的数据"。
    """
    file_config: dict = {}
    if getattr(args, "config", ""):
        config_path = Path(args.config)
        file_config = load_config(config_path)
        check_train_section(config_path, file_config.get(TRAIN_SECTION, {}))

    payload: dict = dict(file_config.get(TRAIN_SECTION, {}))
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
    return TrainConfig(**payload)
