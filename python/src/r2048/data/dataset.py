"""按批取样本：mmap 记录 → 张量，外加按局号的训练/验证划分。

两个口径直接沿用旧版的结论：

* **按整局划分**：同一局里的相邻局面高度相似，按样本随机划分会让验证集里出现“训练集
  同一局的邻近步”，验证指标因此虚高（旧版明确记录过这个坑）。
* **策略样本只在有搜索的情况下才有意义**：`visits` 全为 0 的记录（随机策略或贪心模式产出的）
  没有策略目标，必须被排除而不是当成“均匀分布”来训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from r2048.data.shards import SCORE_SCALE, BatchReader, BatchManifest


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    train_episodes: int
    val_episodes: int

    def describe(self) -> str:
        return (
            f"训练 {len(self.train)} 条 / {self.train_episodes} 局，"
            f"验证 {len(self.val)} 条 / {self.val_episodes} 局"
        )


def make_split(reader: BatchReader, val_fraction: float = 0.05, seed: int = 0) -> Split:
    """按局号确定性地划分。用哈希而不是“取后 5% 的局”，这样打乱数据顺序、
    追加新批次都不会改变已有局的归属。"""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction 必须落在 [0, 1)")
    episodes = reader.episode_column()
    if val_fraction == 0.0:
        return Split(np.arange(reader.samples), np.zeros(0, dtype=np.int64), reader.episodes, 0)

    # 用 splitmix64 的收尾混合：种子以乘法折进状态，而不是简单加一个常数。
    # 简单加常数只让哈希整体平移一个小量，在局数少时选出的验证集可能完全不变
    # （这个坑被 test_split_keeps_episodes_whole 抓到过）。
    with np.errstate(over="ignore"):  # 这里就是要 64 位整数回绕，溢出是设计的一部分
        state = episodes.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed = state ^ (state >> np.uint64(31))
    threshold = np.uint64(int(val_fraction * float(1 << 64)) & ((1 << 64) - 1))
    is_val = hashed < threshold
    train_indices = np.flatnonzero(~is_val)
    val_indices = np.flatnonzero(is_val)
    val_episodes = int(np.unique(episodes[is_val]).size) if val_indices.size else 0
    return Split(train_indices, val_indices, reader.episodes - val_episodes, val_episodes)


@dataclass
class Batch:
    decision: torch.Tensor      # (B, 16) uint8：决策局面
    afterstate: torch.Tensor    # (B, 16) uint8：移动后、生成块前
    action: torch.Tensor        # (B,) int64：实际执行的方向
    legal_mask: torch.Tensor    # (B,) uint8：bit a = 1 表示方向 a 合法
    visits: torch.Tensor        # (B, 4) float32：根访问数
    policy_target: torch.Tensor  # (B, 4) float32：归一化访问分布（无搜索时为全 0）
    value_target: torch.Tensor  # (B,) float32：归一化回报（分数 / score_scale）
    move_score: torch.Tensor    # (B,) float32：归一化立即得分
    episode: torch.Tensor       # (B,) int64
    # --- 候选级监督（schema v2）：每个合法方向一条，按方向下标对齐 ---
    candidate_afterstate: torch.Tensor  # (B, 4, 16) uint8
    candidate_value: torch.Tensor       # (B, 4) float32：A(u_a) = Q̄(a) − r(s,a)（已归一化）
    candidate_visits: torch.Tensor      # (B, 4) float32：该候选边的根访问数
    candidate_weight: torch.Tensor      # (B, 4) float32：sqrt(访问数)，非法/未访达/无候选处为 0

    def __len__(self) -> int:
        return int(self.decision.shape[0])

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            decision=self.decision.to(device),
            afterstate=self.afterstate.to(device),
            action=self.action.to(device),
            legal_mask=self.legal_mask.to(device),
            visits=self.visits.to(device),
            policy_target=self.policy_target.to(device),
            value_target=self.value_target.to(device),
            move_score=self.move_score.to(device),
            episode=self.episode.to(device),
            candidate_afterstate=self.candidate_afterstate.to(device),
            candidate_value=self.candidate_value.to(device),
            candidate_visits=self.candidate_visits.to(device),
            candidate_weight=self.candidate_weight.to(device),
        )

    @property
    def has_candidates(self) -> bool:
        """这批数据是否带候选级监督（随机策略产出的批次没有）。"""
        return bool(self.candidate_weight.sum() > 0)


class BatchStream:
    """按批产出张量。

    刻意不用 `DataLoader` + 逐样本 `Dataset.__getitem__`：那会为每条样本走一次 Python
    调用与一次张量构造，在本项目的样本规模上会直接把 GPU 喂不饱。这里的解码完全向量化
    （结构化视图 + 批量转张量），单进程即可；实测若显示 GPU 挨饿，再引入多 worker
    分片索引（见 docs/BUILD.md 的性能一节）。
    """

    def __init__(
        self,
        reader: BatchReader,
        indices: np.ndarray,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        score_scale: float | None = None,
    ):
        self.reader = reader
        self.indices = indices
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.score_scale = float(score_scale if score_scale is not None else SCORE_SCALE)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = self.indices
        if self.shuffle:
            # 每轮用 (seed, epoch) 决定顺序：可复现，又不会每轮顺序相同。
            rng = np.random.default_rng((self.seed, self.epoch))
            order = rng.permutation(order)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            yield self.decode(chunk)

    def decode(self, chunk: np.ndarray) -> Batch:
        records = self.reader.gather(np.asarray(chunk, dtype=np.int64))
        visits = records["visits"].astype(np.float32)
        # 策略目标按**访问数之和**归一化，而不是按模拟次数：根节点的求值本身会消耗一次
        # 模拟但不落到任何边上（实测根访问数之和 = 模拟次数 − 1；修复 in-flight 浪费前
        # in_flight=8 时只有 56/64）。用模拟次数做分母会让目标权重只有 0.96，等于把引擎的
        # 记账细节写进了训练标签；按访问和归一化后目标恒为概率分布，且与 in_flight 无关。
        visit_sum = visits.sum(axis=1, keepdims=True)
        policy_target = np.divide(visits, visit_sum, out=np.zeros_like(visits), where=visit_sum > 0)
        # 位域展开成 (B,4)：D4 变换要按方向重排掩码与访问分布，位域形式没法直接变换。
        raw_mask = records["legal_mask"].astype(np.uint8)[:, None]
        legal_mask = ((raw_mask >> np.arange(4, dtype=np.uint8)[None, :]) & 1).astype(np.float32)
        return Batch(
            decision=torch.from_numpy(np.ascontiguousarray(records["decision"])),
            afterstate=torch.from_numpy(np.ascontiguousarray(records["afterstate"])),
            action=torch.from_numpy(records["action"].astype(np.int64)),
            legal_mask=torch.from_numpy(np.ascontiguousarray(legal_mask)),
            visits=torch.from_numpy(np.ascontiguousarray(visits)),
            policy_target=torch.from_numpy(np.ascontiguousarray(policy_target)),
            value_target=torch.from_numpy(
                (records["mc_return"].astype(np.float32) / self.score_scale).copy()
            ),
            move_score=torch.from_numpy(
                (records["move_score"].astype(np.float32) / self.score_scale).copy()
            ),
            episode=torch.from_numpy(records["episode"].astype(np.int64)),
            candidate_afterstate=torch.from_numpy(
                np.ascontiguousarray(records["candidate_afterstate"])
            ),
            candidate_value=torch.from_numpy(
                (records["candidate_value"].astype(np.float32) / self.score_scale).copy()
            ),
            candidate_visits=torch.from_numpy(records["candidate_visits"].astype(np.float32).copy()),
            # 权重 = sqrt(访问数) × 合法性（见 ADR 0015）。未访问过的候选权重为 0，
            # 因此“搜索没有估计过它”与“估计为 0”不会被混为一谈。
            candidate_weight=torch.from_numpy(
                (
                    np.sqrt(records["candidate_visits"].astype(np.float32))
                    * legal_mask
                    * (records["candidate_count"] > 0)[:, None]
                ).astype(np.float32)
            ),
        )


def open_batch(data_dir: str | Path, val_fraction: float = 0.05, seed: int = 0) -> tuple[BatchReader, Split]:
    reader = BatchReader(data_dir)
    return reader, make_split(reader, val_fraction=val_fraction, seed=seed)


def batch_manifest_summary(manifest: BatchManifest) -> dict:
    """给日志与产物用的简要描述（不含分片哈希）。"""
    return {
        "batch_id": manifest.data.get("batch_id", ""),
        "policy": manifest.policy,
        "simulations": manifest.simulations,
        "model_id": manifest.model_id,
        "seed": manifest.seed,
        "samples": manifest.samples,
        "episodes": manifest.episodes,
        "build_id": manifest.build_id,
        "params": manifest.params,
    }
