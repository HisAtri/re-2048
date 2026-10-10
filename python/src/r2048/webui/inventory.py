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
    root: Path | None = None
    model_id: str = ""
    architecture: str = ""
    training_target: str = ""

    def label(self) -> str:
        """下拉里显示的一行：**相对输出根的路径** + 训练目标。

        必须是相对路径而不是文件名。输出根里同名 checkpoint 到处都是（每个 run 下都有
        `value/model.pt`、`policy-head/model.pt`），只写 `父目录/文件名` 会得到一列谁也认不出
        的 `value/model.pt`；相对路径把 run 身份带回来了，而且相邻行共享前缀，读起来成组。

        `model_id` 不进来：它常是"数据批 + 配方"拼出来的长串，塞进去每行都撑爆——它在连接
        之后的状态行里显示。
        """
        shown: object = self.path
        if self.root is not None:
            try:
                shown = self.path.relative_to(self.root)
            except ValueError:  # 不在输出根下（不该发生）：退回绝对路径，但不要抛
                shown = self.path
        suffix = f"  [{self.training_target}]" if self.training_target else ""
        return f"{shown}{suffix}"


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
        if path.suffix == ".safetensor":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return json.loads((handle.metadata() or {})["metadata"])
        payload = torch.load(str(path), map_location="meta", weights_only=False)
    except Exception:  # noqa: BLE001 - 列清单不该因为一个坏文件就整体失败
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def list_models(root: Path | None = None, include_resume: bool = False) -> list[ModelEntry]:
    """扫输出根下所有 `*.pt` 与已发布的 `*.safetensor`。

    整个根递归而不是写死 `runs/`、`models/`：模型的实际落点一直在长（训练 run 下、`models/`
    的发布位、实验目录），写死子目录会让"新放出来的模型在下拉里看不见"，而看不见的原因完全
    不可见。代价可以忽略——元数据只读 pickle（`map_location="meta"`，权重留在磁盘上），
    实测 20 个 checkpoint 0.05s。

    `last.pt` 默认不列：它是**断点续跑的落点**，不是"训好的模型"，而每个 run 里都有一份，
    混进来会让列表翻倍、却几乎没人要选它。要的时候由调用方显式打开。
    """
    base = root or paths.output_root()
    if not base.is_dir():
        return []
    entries: list[ModelEntry] = []
    candidates = [*base.rglob("*.pt"), *base.rglob("*.safetensor")]
    for path in sorted(candidates):
        if any(part.startswith(".publish-") for part in path.relative_to(base).parts):
            continue
        if not include_resume and path.name == "last.pt":
            continue
        metadata = _read_checkpoint_metadata(path)
        entries.append(
            ModelEntry(
                path=path,
                root=base,
                model_id=str(metadata.get("model_id", "")),
                architecture=str(metadata.get("architecture", "")),
                training_target=str(metadata.get("training_target", "")),
            )
        )
    return entries


def find_model(label: str, root: Path | None = None, include_resume: bool = False) -> ModelEntry | None:
    """按下拉里显示的那一行找回条目（界面拿到的只有 label）。"""
    for entry in list_models(root, include_resume=include_resume):
        if entry.label() == label:
            return entry
    return None


# 完整的模型：两个头都是这一轮训出来的。迭代目录里另有只训了价值或只训了策略的**半模型**，
# 它们不是"拿来对局的模型"。
COMPLETE_TARGETS = ("dual", "joint")


def default_model(entries: list[ModelEntry]) -> str:
    """默认选中哪个：优先完整模型，而不是"列表里的第一个"。

    列表按路径排（那样才读得出来），于是第一个很可能是 `.../runs/value/model.pt`——半个模型。
    默认值往往就是用户会直接点的那一个，所以它得是有意义的那个。
    """
    for entry in entries:
        if entry.training_target in COMPLETE_TARGETS:
            return entry.label()
    return entries[0].label() if entries else ""


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
