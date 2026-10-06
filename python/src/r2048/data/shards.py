"""读取 C++ 侧写出的数据批：manifest 校验 + 分片 mmap + 向量化解码。

数据格式的唯一权威定义在 `cpp/dataio/sample.h`（偏移表）。这里用同一张偏移表构造一个
numpy 结构化 dtype，然后直接对 mmap 出来的字节做视图——**不做逐条 Python 解码**：
千万级样本上，逐条解码会立刻成为训练管线的瓶颈，而结构化视图是零拷贝的。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 与 cpp/core/version.h、cpp/dataio/sample.h 对齐的契约常量。
RECORD_BYTES = 80
SCHEMA_VERSION = 1
RULES_VERSION = 1
# 分数归一化尺度来自规则契约（cpp/core/version.h 的 kScoreScale）。记录里不存它，
# 因为它是语义常量而不是每条样本的属性；测试里会与 C++ 侧暴露的值做一致性校验。
SCORE_SCALE = 1024.0

# 偏移表与 cpp/dataio/sample.h 的注释逐项对应；末尾保留字节以 _padX 命名，
# 便于将来加字段时只改这里和 C++ 侧（记录长度不变）。
RECORD_DTYPE = np.dtype(
    [
        ("decision", "u1", (16,)),      # 0
        ("afterstate", "u1", (16,)),    # 16
        ("action", "u1"),               # 32
        ("legal_mask", "u1"),           # 33
        ("step", "<u2"),                # 34
        ("episode", "<u4"),             # 36
        ("simulations", "<u2"),         # 40
        ("move_score", "<u4"),          # 42
        ("_pad0", "u1", (2,)),          # 46
        ("mc_return", "<i8"),           # 48
        ("_pad1", "u1", (8,)),          # 56
        ("visits", "<u2", (4,)),        # 64
        ("_pad2", "u1", (8,)),          # 72
    ],
    align=False,
)

if RECORD_DTYPE.itemsize != RECORD_BYTES:  # pragma: no cover - 契约自检
    raise RuntimeError(f"记录布局与 C++ 契约不一致：{RECORD_DTYPE.itemsize} != {RECORD_BYTES}")


@dataclass(frozen=True)
class BatchManifest:
    path: Path
    data: dict

    @property
    def samples(self) -> int:
        return int(self.data["samples"])

    @property
    def episodes(self) -> int:
        return int(self.data["episodes"])

    @property
    def policy(self) -> str:
        return str(self.data.get("policy", ""))

    @property
    def simulations(self) -> int:
        return int(self.data.get("simulations", 0))

    @property
    def model_id(self) -> str:
        return str(self.data.get("model_id", ""))

    @property
    def build_id(self) -> str:
        return str(self.data.get("build_id", ""))

    @property
    def seed(self) -> int:
        return int(self.data.get("seed", 0))

    @property
    def params(self) -> dict:
        return dict(self.data.get("params", {}))

    def describe(self) -> str:
        return (
            f"{self.path.parent.name}: 策略 {self.policy}，模拟 {self.simulations}，"
            f"模型 {self.model_id or '-'}，样本 {self.samples}，局数 {self.episodes}，"
            f"引擎 {self.build_id}"
        )


def read_manifest(data_dir: str | Path, check_versions: bool = True) -> BatchManifest:
    """读取并校验 manifest。

    版本不匹配直接失败：用错的语义解释一份数据比读不出来危险得多——它会安静地污染训练。
    """
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。数据批以 manifest.json 作为“已写完”的提交标记，"
            "只有它出现才说明这一批是完整的。"
        )
    data = json.loads(path.read_text())
    if check_versions:
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"数据批的记录格式版本 {data['schema_version']} 与本实现 {SCHEMA_VERSION} 不一致"
            )
        if int(data["rules_version"]) != RULES_VERSION:
            raise ValueError(
                f"数据批的规则版本 {data['rules_version']} 与本实现 {RULES_VERSION} 不一致"
            )
        if int(data["record_bytes"]) != RECORD_BYTES:
            raise ValueError(
                f"数据批的记录长度 {data['record_bytes']} 与本地布局 {RECORD_BYTES} 不一致"
            )
    return BatchManifest(path=path, data=data)


def verify_shards(manifest: BatchManifest, progress: bool = False) -> None:
    """逐分片校验 sha256。默认不跑（训练前的常规检查只做结构与版本），
    但“这批数据是否被改过”这个问题只有它能回答，因此保留成显式动作。"""
    for index, shard in enumerate(manifest.data["shards"]):
        path = manifest.path.parent / shard["path"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != shard["sha256"]:
            raise ValueError(f"分片摘要不匹配：{shard['path']}")
        if progress:
            print(f"\r校验分片 {index + 1}/{len(manifest.data['shards'])}", end="", flush=True)
    if progress:
        print()


class BatchReader:
    """一份数据批的只读视图。

    分片用 mmap 打开、不做拷贝：训练按批随机取样本，若把整批读进内存，
    千万级样本会直接吃掉几 GB；mmap 让内核按需换页，且多个 worker 共享同一份只读映射。
    """

    def __init__(self, data_dir: str | Path, check_versions: bool = True):
        self.data_dir = Path(data_dir)
        self.manifest = read_manifest(self.data_dir, check_versions=check_versions)
        self._arrays: list[np.ndarray] = []
        self._cumulative: np.ndarray | None = None
        self._episodes: np.ndarray | None = None

        counts = []
        cumulative = [0]
        for shard in self.manifest.data["shards"]:
            path = self.data_dir / shard["path"]
            array = np.memmap(path, dtype=RECORD_DTYPE, mode="r")
            expected = int(shard["samples"])
            if array.shape[0] != expected:
                raise ValueError(
                    f"{shard['path']} 的记录数 {array.shape[0]} 与清单 {expected} 不一致"
                )
            self._arrays.append(array)
            counts.append(array.shape[0])
            cumulative.append(cumulative[-1] + array.shape[0])
        self._cumulative = np.asarray(cumulative, dtype=np.int64)

        if int(self._cumulative[-1]) != self.manifest.samples:
            raise ValueError("分片样本总数与清单不一致")

    # ---- 基本属性 ----

    @property
    def samples(self) -> int:
        return int(self._cumulative[-1])

    @property
    def episodes(self) -> int:
        return self.manifest.episodes

    def episode_column(self) -> np.ndarray:
        """每条样本的局号（用于按局切分训练/验证）。读一次后缓存。"""
        if self._episodes is None:
            self._episodes = np.concatenate([array["episode"] for array in self._arrays])
        return self._episodes

    def simulations_column(self) -> np.ndarray:
        return np.concatenate([array["simulations"] for array in self._arrays])

    # ---- 取批 ----

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """按全局样本下标取记录。跨分片时按分片分组后再取，避免逐条 Python 循环。"""
        out = np.empty(len(indices), dtype=RECORD_DTYPE)
        if len(indices) == 0:
            return out
        shard_of = np.searchsorted(self._cumulative[1:], indices, side="right")
        for shard in np.unique(shard_of):
            mask = shard_of == shard
            local = indices[mask] - self._cumulative[shard]
            out[mask] = self._arrays[shard][local]
        return out

    def close(self) -> None:
        self._arrays.clear()
        self._episodes = None
