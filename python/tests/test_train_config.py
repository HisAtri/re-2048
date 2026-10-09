"""训练配置层的契约：CLI > YAML > 默认值，以及配置摘要覆盖了这份配置。

配置合并最容易出的错不是"算错"，而是"没生效"或"生效了但没人知道"：YAML 里拼错的键、
CLI 覆盖没接上、摘要没覆盖新字段——三者都不会让训练报错，只会让实验在错误的配置下
跑完并留下一堆看似正常的产物。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from r2048.cli import _build_parser
from r2048.config import build_train_config
from r2048.nn.training import DatasetSpec, TrainConfig, config_digest


def args_of(**overrides) -> argparse.Namespace:
    """模拟 `--config` / `--data` / 其余参数未给出时的 namespace（SUPPRESS 默认值）。"""
    payload = {"command": "train", "config": "", "data": "", "output": "run/out"}
    payload.update(overrides)
    return argparse.Namespace(**payload)


def write_config(tmp_path: Path, text: str, name: str = "train.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_cli_overrides_yaml_and_yaml_overrides_defaults(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        """
datasets:
  - path: data/one
    weight: 0.25
  - path: data/two
train:
  mode: dual
  epochs: 40
  batch_size: 512
""",
    )
    # CLI 只给了 epochs：另外两项必须来自 YAML，没提到的字段回落到 TrainConfig 的默认值。
    config = build_train_config(args_of(config=config_path, epochs=3))
    assert config.epochs == 3, "CLI 必须覆盖 YAML"
    assert config.mode == "dual" and config.batch_size == 512, "YAML 必须覆盖默认值"
    assert config.log_every == TrainConfig().log_every, "没提到的字段应当是 TrainConfig 的默认值"
    assert config.datasets == [
        DatasetSpec(path="data/one", weight=0.25),
        DatasetSpec(path="data/two"),
    ], "datasets 必须逐项读出来（含权重与默认 limit）"


def test_data_flag_overrides_yaml_datasets(tmp_path: Path) -> None:
    """编排要用同一条训练线的参数去训新跑出来的批次，所以 --data 优先于 YAML 清单。"""
    config_path = write_config(tmp_path, "datasets:\n  - path: data/old\ntrain:\n  epochs: 4\n")
    config = build_train_config(args_of(config=config_path, data="data/fresh"))
    assert config.datasets == [DatasetSpec(path="data/fresh")]
    assert config.epochs == 4


def test_single_batch_without_config(tmp_path: Path) -> None:
    """`--data` 单批用法保持原样：不要求写配置文件。"""
    config = build_train_config(args_of(data="data/batch", device="cpu", no_augment=True))
    assert config.datasets == [DatasetSpec(path="data/batch")]
    assert config.device == "cpu"
    assert config.augment is False, "取反开关必须落到正向字段上"


def test_tree_label_source_can_be_switched_from_cli() -> None:
    """同批 A/B 必须能从训练入口切换标签，不能只改 Python 内部默认值。"""
    args = _build_parser().parse_args(
        ["train", "--data", "data/batch", "--output", "run/out", "--tree-label-source", "qmean"]
    )
    assert build_train_config(args).tree_label_source == "qmean"


def test_missing_data_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="datasets"):
        build_train_config(args_of())
    config_path = write_config(tmp_path, "datasets:\n  - path: data/one\n")
    # 有 config 但 datasets 小节为空同样是"没有数据"。
    empty = write_config(tmp_path, "train:\n  epochs: 2\n", name="empty.yaml")
    assert build_train_config(args_of(config=config_path)).datasets
    with pytest.raises(SystemExit, match="datasets"):
        build_train_config(args_of(config=empty))


def test_unknown_keys_are_refused(tmp_path: Path) -> None:
    """拼错的键必须报错：静默忽略等于"改了半天配置没生效"。"""
    typo_section = write_config(tmp_path, "dataset:\n  - path: data/one\n", name="section.yaml")
    with pytest.raises(ValueError, match="未知小节"):
        build_train_config(args_of(config=typo_section))

    typo_field = write_config(
        tmp_path, "datasets:\n  - path: data/one\ntrain:\n  epoch: 4\n", name="field.yaml"
    )
    with pytest.raises(ValueError, match="未知字段"):
        build_train_config(args_of(config=typo_field))

    typo_dataset = write_config(
        tmp_path, "datasets:\n  - path: data/one\n    weitgh: 2\n", name="dataset.yaml"
    )
    with pytest.raises(ValueError, match="未知键"):
        build_train_config(args_of(config=typo_dataset))

    # 回调是运行期接进来的，配置文件不该能设它（设了也没用，只说明写的人误解了它）。
    callback = write_config(
        tmp_path, "datasets:\n  - path: data/one\ntrain:\n  on_progress: 1\n", name="callback.yaml"
    )
    with pytest.raises(ValueError, match="未知字段"):
        build_train_config(args_of(config=callback))


def test_config_digest_covers_the_config_and_ignores_callbacks(tmp_path: Path) -> None:
    """摘要要覆盖数据来源与超参，且不受运行期回调影响。

    回调混进摘要会让"带回调跑（走 CLI 的每一次）"与"库调用"算出不同摘要；
    回调进不了 JSON，真混进去就是直接崩在写 checkpoint 上。
    """
    base = TrainConfig(datasets=[DatasetSpec(path="data/one")], output="run", lr=1e-3)
    other_lr = TrainConfig(datasets=[DatasetSpec(path="data/one")], output="run", lr=2e-3)
    other_weight = TrainConfig(datasets=[DatasetSpec(path="data/one", weight=2.0)], output="run", lr=1e-3)
    assert config_digest(base) != config_digest(other_lr)
    assert config_digest(base) != config_digest(other_weight), "采样权重变了必须换摘要"

    base.on_progress = lambda payload: None
    assert config_digest(base) == config_digest(
        TrainConfig(datasets=[DatasetSpec(path="data/one")], output="run", lr=1e-3)
    )


def test_yaml_sample_from_docs_is_loadable(tmp_path: Path) -> None:
    """文档里那份样例配置必须真的能被读进来（文档与代码分叉时，先坏的是文档）。"""
    config_path = write_config(
        tmp_path,
        yaml.safe_dump(
            {
                "datasets": [
                    {"path": "batches/teacher-s256", "weight": 1.0, "limit": 0},
                    {"path": "batches/teacher-s1024", "weight": 0.5},
                ],
                "train": {
                    "mode": "value",
                    "epochs": 8,
                    "batch_size": 4096,
                    "lr": 3.0e-4,
                    "seed": 11,
                },
            },
            sort_keys=False,
        ),
    )
    config = build_train_config(args_of(config=config_path))
    assert config.mode == "value" and config.epochs == 8
    assert config.lr == pytest.approx(3.0e-4)
    assert config.seed == 11
    assert [spec.weight for spec in config.datasets] == [1.0, 0.5]
