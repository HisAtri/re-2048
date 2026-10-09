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
# v3 起记录长度是 **基础块 + K × 树内节点块**，K 由清单的 tree_nodes 声明（0 = 没有这块）。
# 因此这里没有单一的 RECORD_BYTES，只有一个函数——两份定义迟早会分叉。
BASE_RECORD_BYTES = 192
TREE_NODE_BYTES = 124
TREE_NODES_MAX = 4
SCHEMA_VERSION = 4


def record_dtype(tree_nodes: int) -> np.dtype:
    """按清单声明的 K 构造记录 dtype。K 不符时记录的字节数就错了（读出来的会是垃圾）。"""
    if not 0 <= tree_nodes <= TREE_NODES_MAX:
        raise ValueError(f"tree_nodes 超出范围 0..{TREE_NODES_MAX}：{tree_nodes}")
    fields = _BASE_FIELDS + [
        (f"tree{slot}_{spec[0]}", *spec[1:]) for slot in range(tree_nodes) for spec in _TREE_FIELDS
    ]
    return np.dtype(fields, align=False)


RULES_VERSION = 1
# 分数归一化尺度来自规则契约（cpp/core/version.h 的 kScoreScale）。记录里不存它，
# 因为它是语义常量而不是每条样本的属性；测试里会与 C++ 侧暴露的值做一致性校验。
SCORE_SCALE = 1024.0

# 候选级监督的标签定义（与 cpp/dataio/sample.h 的注释一致）：
#   candidate_value[a] = Q̄(a) − r(s,a) = A(u_a) 的估计（原始分数单位）
# 之所以强调：搜索在每个决策节点查询**全部合法动作**的 afterstate，而主轨迹只记录实际
# 走过的那个；只报告全局 MAE 会掩盖“在自己被查询的地方没有监督”这件事。
CANDIDATE_TARGET_DEFINITION = "q_mean_minus_reward"

# 偏移表与 cpp/dataio/sample.h 的注释逐项对应；末尾保留字节以 _padX 命名，
# 便于将来加字段时只改这里和 C++ 侧（记录长度不变）。
_BASE_FIELDS = [
    ("decision", "u1", (16,)),  # 0
    ("afterstate", "u1", (16,)),  # 16
    ("action", "u1"),  # 32
    ("legal_mask", "u1"),  # 33
    ("step", "<u2"),  # 34
    ("episode", "<u4"),  # 36
    ("simulations", "<u2"),  # 40
    ("move_score", "<u4"),  # 42
    ("_pad0", "u1", (2,)),  # 46
    ("mc_return", "<i8"),  # 48
    ("_pad1", "u1", (8,)),  # 56
    ("visits", "<u2", (4,)),  # 64
    ("_pad2", "u1", (8,)),  # 72
    # --- v2 候选块：按方向下标对齐，槽位 a 恒对应方向 a ---
    ("candidate_afterstate", "u1", (4, 16)),  # 80
    ("candidate_value", "<i4", (4,)),  # 144：A(u_a) = Q̄(a) − r(s,a)，原始分数单位
    ("candidate_reward", "<u4", (4,)),  # 160：r(s,a)
    ("candidate_visits", "<u2", (4,)),  # 176：该候选边的根访问数
    ("candidate_count", "u1"),  # 184：0 = 无候选监督
    ("tree_count", "u1"),  # 185：实际写出的树内节点数（0..K）
    ("_pad3", "u1", (6,)),  # 186
]

# v3 树内节点块：每个节点 108 字节，按方向下标对齐（与 cpp/dataio/sample.h 的偏移表逐项对应）。
# 它是"价值头真正说了算"的位置上的监督——价值头在搜索里被直接使用的地方是叶节点与未被访问的
# 边，而候选块只覆盖轨迹根（那一点搜索自己就把排序算出来了，见 ADR 0035/0041）。
_TREE_FIELDS = [
    ("legal_mask", "u1"),  # +0
    ("count", "u1"),  # +1
    ("depth", "u1"),  # +2
    ("pad", "u1"),  # +3（4 字节对齐）
    ("value", "<i4", (4,)),  # +4：Q̄(a) − r(s,a)，原始分数单位
    ("reward", "<u4", (4,)),  # +20：r(s,a)
    ("visits", "<u2", (4,)),  # +36：该节点上这条边的访问数
    ("afterstate", "u1", (4, 16)),  # +44
    # +108：**独立** playout 对 A(u_a) 的估计（原始分数单位）。与 value（搜索自己的 Q̄ − r）并存：
    # 低频节点的 value 约等于头自己的输出（one_ply），拿它当监督等于让学生复制自己；rollout 由
    # 规则驱动的续弈策略给出、与头无关（ADR 0043）。
    ("rollout", "<i4", (4,)),
]

if record_dtype(0).itemsize != BASE_RECORD_BYTES:  # pragma: no cover - 契约自检
    raise RuntimeError(f"基础布局与 C++ 契约不一致：{record_dtype(0).itemsize} != {BASE_RECORD_BYTES}")
if record_dtype(1).itemsize != BASE_RECORD_BYTES + TREE_NODE_BYTES:  # pragma: no cover
    raise RuntimeError("树内节点块的字节数与 C++ 契约不一致")


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

    @property
    def state(self) -> str:
        """这一批的生命周期状态（running/paused/stopped/complete）。

        跑谱可以随时停车并就地续跑，所以 manifest 是增量提交日志而不是结尾标记：
        任何时刻读到它，都要知道这一批是「还在长」还是「已经定稿」。取不到状态的老批
        按 complete 算——旧协议只在跑完时才写 manifest，这个默认值与事实一致。
        """
        return str(self.data.get("state", "complete"))

    @property
    def tree_label_plays(self) -> int:
        """树内节点每个候选跑了几次独立 playout（0 = 这批数据没有独立标签）。"""
        return int(self.data.get("tree_label_plays", 0))

    @property
    def tree_nodes(self) -> int:
        """这批数据每条记录里放了几个树内节点（0 = 没有这一块）。"""
        return tree_nodes_of(self.data)

    @property
    def shard_layout(self) -> int:
        """分片切分规则版本：1 = 按容量精确切（旧），2 = 按局边界切（当前）。

        续跑只有在规则相同时才可能逐字节复现，所以这个值必须能被读到并参与比对。
        """
        return int(self.data.get("shard_layout", 1))

    def source_key(self) -> str:
        """这批数据的**来源身份**：由"第 k 局是哪一局"的字段算出，不看目录与批次号。

        为什么需要它：`episode` 是**批内**编号（0..N−1），而切分是按局哈希做的。
        两个来源共用同一 seed 时（自对弈的常规情形——`teacher-s256` 与 `teacher-s1024`
        都是 seed 20261020、局号都从 0 起），合并后同一个局号指向两局不同的棋：
        "同一局不跨训练/验证集"这条保护随之失效，按局统计也会串线。
        把来源折进局的身份是唯一的修法。

        为什么是这些字段而不是路径或批次号：来源是"谁、用什么协议产出的"，
        与它放在哪个目录、叫什么批次号无关（搬目录、就地续跑都不该换身份）。
        也**不含** samples / episodes / state——续跑会一直改它们，
        而续跑前后的数据按定义仍是同一个来源。
        """
        return source_identity(self.data)[0]

    def source_code(self) -> int:
        """来源身份的 64 位数值形式，供向量化的按局哈希使用（见 dataset.make_split）。"""
        return source_identity(self.data)[1]

    def describe(self) -> str:
        # 状态写在最前面：拿一份 paused/stopped 的批次去训练，是「数据还在长」而不是
        # 「数据已定稿」，这个区别必须在第一眼就看见。
        return (
            f"{self.path.parent.name} [{self.state}]: 策略 {self.policy}，模拟 {self.simulations}，"
            f"模型 {self.model_id or '-'}，样本 {self.samples}，局数 {self.episodes}，"
            f"引擎 {self.build_id}，来源 {self.source_key()}"
        )


# 决定"第 k 局是哪一局"的字段：协议 + 种子 + 产出者。
# 少一项就等于把两个不同的来源当成同一个：例如漏掉 simulations，"s256" 与 "s1024"
# 的自对弈数据会被认为第 0 局是同一局，而同种子下它们下的是完全不同的棋。
SOURCE_KEY_FIELDS = ("policy", "simulations", "temperature_moves", "seed", "model_id", "build_id")
# params 里属于协议的键（与 C++ 侧 cpp/dataio/shard.cpp 的 kProtocolParams 同源：
# 那边用它做续跑闸门，这边用它做来源身份——两处都只关心"会改变逐局结果"的参数）。
SOURCE_KEY_PARAMS = (
    "c_puct",
    "lanes",
    "in_flight_per_lane",
    "temperature",
    "cache_entries",
    "max_batch",
    "device",
    "search_threads",
    # rollout 的续弈策略（uniform / greedy）：它决定候选价值标签的信息量（ADR 0030），
    # 也改变逐局结果，所以必须进来源身份——否则"64 模拟 + 均匀续弈"与"64 模拟 + 启发式续弈"
    # 两份数据会被当成同一个来源，同一个局号会指向两盘不同的棋。
    "rollout_policy",
)


def source_identity(data: dict) -> tuple[str, int]:
    """算出来源身份：返回（十六进制短标识, 64 位数值）。

    两个形式来自同一个摘要：短标识给人看与写进日志/元数据，
    数值形式让"按局哈希"可以整数组算，不必逐样本走 Python。
    """
    params = data.get("params", {}) or {}
    canonical = json.dumps(
        {
            "fields": {name: data.get(name) for name in SOURCE_KEY_FIELDS},
            "params": {name: params.get(name) for name in SOURCE_KEY_PARAMS if name in params},
            "schema_version": data.get("schema_version"),
            "rules_version": data.get("rules_version"),
            "protocol_version": data.get("protocol_version"),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    digest = hashlib.blake2b(canonical, digest_size=8).digest()
    return digest.hex(), int.from_bytes(digest, "little")


def record_bytes(tree_nodes: int) -> int:
    """一条记录的字节数（基础块 + K 个树内节点块）。"""
    return BASE_RECORD_BYTES + int(tree_nodes) * TREE_NODE_BYTES


def tree_nodes_of(data: dict) -> int:
    """清单里声明的树内节点数 K。缺失或越界都直接失败：它决定记录长度，猜不得。"""
    if "tree_nodes" not in data:
        raise ValueError(f"清单缺少 tree_nodes：这不是 v{SCHEMA_VERSION} 格式的数据批（旧格式请见归档说明）")
    value = int(data["tree_nodes"])
    if not 0 <= value <= TREE_NODES_MAX:
        raise ValueError(f"清单的 tree_nodes={value} 超出本实现支持的范围 0..{TREE_NODES_MAX}")
    return value


def read_manifest(data_dir: str | Path, check_versions: bool = True) -> BatchManifest:
    """读取并校验 manifest。

    版本不匹配直接失败：用错的语义解释一份数据比读不出来危险得多——它会安静地污染训练。
    """
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。数据批以 manifest.json 作为“已写完”的提交标记，只有它出现才说明这一批是完整的。"
        )
    data = json.loads(path.read_text())
    if check_versions:
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"数据批的记录格式版本 {data['schema_version']} 与本实现 {SCHEMA_VERSION} 不一致"
            )
        if int(data["rules_version"]) != RULES_VERSION:
            raise ValueError(f"数据批的规则版本 {data['rules_version']} 与本实现 {RULES_VERSION} 不一致")
        # 记录长度由清单声明的 tree_nodes 决定（v3）：这里是"声明与推导必须一致"的自检，
        # 而不是把某个常量再抄一遍。缺 tree_nodes 的批次属于 v2 或更早，已在版本闸门处失败。
        expected_bytes = record_bytes(tree_nodes_of(data))
        if int(data["record_bytes"]) != expected_bytes:
            raise ValueError(
                f"数据批的记录长度 {data['record_bytes']} 与 tree_nodes={tree_nodes_of(data)} "
                f"推出的 {expected_bytes} 不一致"
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
        # 记录布局由清单声明的 K 决定：所有读路径都从这里取 dtype，不再有全局的 RECORD_DTYPE。
        self.dtype = record_dtype(self.manifest.tree_nodes)
        self.tree_nodes = self.manifest.tree_nodes
        self._arrays: list[np.ndarray] = []
        self._cumulative: np.ndarray | None = None
        self._episodes: np.ndarray | None = None

        counts = []
        cumulative = [0]
        for shard in self.manifest.data["shards"]:
            path = self.data_dir / shard["path"]
            array = np.memmap(path, dtype=self.dtype, mode="r")
            expected = int(shard["samples"])
            if array.shape[0] != expected:
                raise ValueError(f"{shard['path']} 的记录数 {array.shape[0]} 与清单 {expected} 不一致")
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
        out = np.empty(len(indices), dtype=self.dtype)
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

    def describe(self) -> str:
        """训练日志的一行来源说明。与 `MultiSourceReader.describe()` 同形，
        训练侧不必区分拿到的是一批还是多批。"""
        return self.manifest.describe()
