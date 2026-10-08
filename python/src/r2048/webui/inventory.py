"""清单层：扫描输出根，把"能选的东西"列出来。

为什么要有这一层：界面里**不该让人手输路径**。跑谱批次、模型、棋谱的名字是机器生成的
（批次号带时间戳、模型在 run 目录下），手输既容易错、又要把输出根的布局知识搬到用户脑子里。
所以界面只提供下拉选择，而"有哪些可选"由这里回答。

三件事都是只读扫描，不改任何东西。模型元数据走 `torch.load(map_location="meta")`：
权重留在磁盘上（meta 张量不占内存），只把 pickle 里的 metadata 读出来——列出几十个模型
也不需要把几十份权重读进内存。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from r2048 import paths


@dataclass(frozen=True)
class ModelEntry:
    path: Path
    model_id: str = ""
    architecture: str = ""
    training_target: str = ""

    def label(self) -> str:
        """下拉里显示的一行：路径 + 训练目标。

        模型标识（`model_id`）刻意不放在这里——它常常是"数据批 + 配方"拼出来的长串，塞进下拉
        会把每行都撑爆、反而谁也认不出来。它会在连接之后的状态行里显示。
        """
        suffix = f"  [{self.training_target}]" if self.training_target else ""
        return f"{self.path.parent.name}/{self.path.name}{suffix}"


@dataclass(frozen=True)
class BatchEntry:
    path: Path
    batch_id: str = ""
    policy: str = ""
    simulations: int = 0
    samples: int = 0
    episodes: int = 0
    state: str = ""
    model_id: str = ""

    def label(self) -> str:
        return (
            f"{self.batch_id}  [{self.state}]  {self.policy} 模拟{self.simulations}  "
            f"{self.samples} 样本 / {self.episodes} 局  模型 {self.model_id or '-'}"
        )


@dataclass(frozen=True)
class RecordEntry:
    path: Path
    moves: int = 0

    def label(self) -> str:
        return f"{self.path.stem}  （{self.moves} 步）"


def _read_checkpoint_metadata(path: Path) -> dict:
    import torch

    try:
        payload = torch.load(str(path), map_location="meta", weights_only=False)
    except Exception:  # noqa: BLE001 - 列清单不该因为一个坏文件就整体失败
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def list_models(root: Path | None = None) -> list[ModelEntry]:
    """扫输出根下所有 `*.pt`。

    不按固定子目录扫（`runs/`、`models/`）而是整个根递归：模型的实际落点一直在长
    （训练 run 下、`models/` 的发布位、以及实验目录），写死子目录会让"新放出来的模型
    在下拉里看不见"，而看不见的原因又完全不可见。代价可以忽略——元数据读取用 meta 张量，
    实测 20 个文件 0.05s（权重留在磁盘上）。
    """
    base = root or paths.output_root()
    if not base.is_dir():
        return []
    entries: list[ModelEntry] = []
    for path in sorted(base.rglob("*.pt")):
        metadata = _read_checkpoint_metadata(path)
        entries.append(
            ModelEntry(
                path=path,
                model_id=str(metadata.get("model_id", "")),
                architecture=str(metadata.get("architecture", "")),
                training_target=str(metadata.get("training_target", "")),
            )
        )
    return entries


def find_model(label: str, root: Path | None = None) -> ModelEntry | None:
    """按下拉里显示的那一行找回条目（界面拿到的只有 label）。"""
    for entry in list_models(root):
        if entry.label() == label:
            return entry
    return None


def list_batches(root: Path | None = None) -> list[BatchEntry]:
    """扫 `data/*/manifest.json`。清单是这批数据"已提交"的标记（ADR 0007），所以有它就够。"""
    base = root or paths.output_root()
    entries: list[BatchEntry] = []
    directory = base / "data"
    if not directory.is_dir():
        return entries
    for manifest in sorted(directory.glob("*/manifest.json")):
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            BatchEntry(
                path=manifest.parent,
                batch_id=str(data.get("batch_id", manifest.parent.name)),
                policy=str(data.get("policy", "")),
                simulations=int(data.get("simulations", 0)),
                samples=int(data.get("samples", 0)),
                episodes=int(data.get("episodes", 0)),
                state=str(data.get("state", "complete")),
                model_id=str(data.get("model_id", "")),
            )
        )
    return entries


def list_records(root: Path | None = None) -> list[RecordEntry]:
    """扫 `records/*.txt`。步数只在能被解析出来时填——格式坏掉的棋谱仍然要能被看到。"""
    from r2048 import records

    base = root or paths.output_root()
    entries: list[RecordEntry] = []
    directory = base / "records"
    if not directory.is_dir():
        return entries
    for path in sorted(directory.glob("*.txt")):
        try:
            commands = records.parse(path.read_text(encoding="utf-8"))
        except (OSError, records.RecordError):
            entries.append(RecordEntry(path=path, moves=-1))  # -1 = 读不出来，界面上要看得见
            continue
        entries.append(RecordEntry(path=path, moves=sum(1 for c in commands if isinstance(c, records.Move))))
    return entries
