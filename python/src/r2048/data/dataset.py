"""按批取样本：mmap 记录 → 张量，外加按局号的训练/验证划分。

三个口径直接沿用旧版的结论：

* **按整局划分**：同一局里的相邻局面高度相似，按样本随机划分会让验证集里出现"训练集
  同一局的邻近步"，验证指标因此虚高（旧版明确记录过这个坑）。
* **策略样本只在有搜索的情况下才有意义**：`visits` 全为 0 的记录（随机策略或贪心模式产出的）
  没有策略目标，必须被排除而不是当成"均匀分布"来训练。
* **局号只在批内有意义**：`episode` 是 0..N−1 的批内编号，所以"第 7 局"这个身份必须带上
  来源（`source_key`）。两个来源共用同一 seed 时，裸局号会把两局完全不同的棋当成同一局——
  划分会串线、按局统计会重复计数（见 ADR 0021）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch

from r2048.data.shards import (
    SCORE_SCALE,
    BatchManifest,
    BatchReader,
    record_dtype,
)


@dataclass(frozen=True)
class SourceSplit:
    """一个来源在划分里的结果：局数必须逐来源报出来。

    只报"总共多少局"会掩盖一件重要的事：某个来源可能有 0 条进了验证集。
    多来源训练时这是最常见的配置错误（小来源被权重放大的同时整批进了训练集）。
    """

    key: str
    path: str
    samples: int
    train_episodes: int
    val_episodes: int

    def describe(self) -> str:
        return (
            f"来源 {self.key}（{Path(self.path).name}）："
            f"训练 {self.train_episodes} 局，验证 {self.val_episodes} 局，{self.samples} 条"
        )


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    train_episodes: int
    val_episodes: int
    sources: tuple[SourceSplit, ...] = ()

    def describe(self) -> str:
        head = (
            f"训练 {len(self.train)} 条 / {self.train_episodes} 局，"
            f"验证 {len(self.val)} 条 / {self.val_episodes} 局"
        )
        if len(self.sources) <= 1:
            return head
        return "\n".join([head, *(f"  {source.describe()}" for source in self.sources)])


def _episode_hash(episodes: np.ndarray, seed: int, source_code: int) -> np.ndarray:
    """把（来源, 局号）哈希成 64 位整数，供按局划分使用。

    种子以乘法折进状态，而不是简单加一个常数：简单加常数只让哈希整体平移一个小量，
    在局数少时选出的验证集可能完全不变（这个坑被 test_split_keeps_episodes_whole 抓到过）。
    来源身份同样要真正参与混合，否则"同一个局号、不同来源"仍会算出同一个哈希——
    那正是多来源合并时切分串线的形态。
    """
    with np.errstate(over="ignore"):  # 这里就是要 64 位整数回绕，溢出是设计的一部分
        state = episodes.astype(np.uint64) + np.uint64(source_code)
        state = state + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return state ^ (state >> np.uint64(31))


def make_split(reader: MultiSourceReader | BatchReader, val_fraction: float = 0.05, seed: int = 0) -> Split:
    """按（来源, 局号）确定性地划分训练/验证。

    用哈希而不是"取后 5% 的局"，这样打乱数据顺序、给某个来源追加数据（跑到一半续跑）
    都不会改变已有局的归属。

    与多来源一起工作的关键在**身份**：划分判定的是 `hash(source_key, episode)`，
    而不是裸的局号。同一个来源第 7 局永远落在同一侧（哪怕它出现在另一批数据里），
    而"同种子同局号但协议不同"的两个来源不会再被判成同一局（见 ADR 0021）。
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction 必须落在 [0, 1)")
    sources = reader.sources if isinstance(reader, MultiSourceReader) else _single_source(reader)
    episodes = reader.episode_column()
    threshold = np.uint64(int(val_fraction * float(1 << 64)) & ((1 << 64) - 1))

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    reports: list[SourceSplit] = []
    for index, source in enumerate(sources):
        start = source.offset
        stop = source.offset + source.samples
        local = episodes[start:stop]
        if val_fraction == 0.0:
            is_val = np.zeros(local.shape, dtype=bool)
        else:
            is_val = _episode_hash(local, seed, source.code) < threshold
        # 逐来源原地划分再拼接：全局下标空间是各来源的连续区间，所以偏移量相加即可，
        # 不需要再维护一份"每条样本属于谁"的数组（千万级样本上那是一份白占的内存）。
        offsets = np.arange(start, stop, dtype=np.int64)
        train_parts.append(offsets[~is_val])
        val_parts.append(offsets[is_val])
        val_episodes = int(np.unique(local[is_val]).size) if is_val.any() else 0
        local_episodes = source.episodes
        reports.append(
            SourceSplit(
                key=source.key,
                path=source.path,
                samples=source.samples,
                train_episodes=local_episodes - val_episodes,
                val_episodes=val_episodes,
            )
        )

    train = np.concatenate(train_parts) if train_parts else np.zeros(0, dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.int64)
    return Split(
        train=train,
        val=val,
        train_episodes=sum(report.train_episodes for report in reports),
        val_episodes=sum(report.val_episodes for report in reports),
        sources=tuple(reports),
    )


@dataclass(frozen=True)
class SourceView:
    """一个来源在合并视图里的位置与身份。

    `key`/`code` 是同一个摘要的两种形式：人读的短标识与供向量化哈希用的 64 位数值。
    `offset` 是它在全局下标空间的起点，`samples` 是它贡献的样本数（`limit` 之后）。
    """

    key: str
    code: int
    path: str
    manifest: BatchManifest
    samples: int
    episodes: int
    offset: int

    @property
    def name(self) -> str:
        return Path(self.path).name

    def describe(self) -> str:
        return f"{self.key}（{self.name}）：{self.samples} 条 / {self.episodes} 局"


class MultiSourceReader:
    """把多个数据批当成一个数据集：索引是首尾相接的全局下标。

    这样 `BatchStream` / `Split` / 采样器都不必知道来源存在，只有三处需要按来源切一刀，
    而那只需要各来源的 `(offset, samples)`：划分身份（`make_split`）、逐来源统计、配额采样。
    刻意**不**维护"每条样本属于谁"的数组——千万级样本上那是一份白占的内存，
    而来源是连续区间，`searchsorted` 一下就够。

    `limit` 按**条**截断（0 = 不限）。它可能切在一局中间，这是允许的：按局划分保护的是
    "整局都在池子里"的情形，被截断的半局不会让同一局的样本两侧都出现。
    """

    def __init__(self, data_dirs, limits=None, weights=None):
        dirs = [Path(path) for path in data_dirs]
        if not dirs:
            raise ValueError("至少要有一个数据来源")
        limits = [0] * len(dirs) if limits is None else [int(limit) for limit in limits]
        if len(limits) != len(dirs):
            raise ValueError(f"limit 个数 {len(limits)} 与来源个数 {len(dirs)} 不一致")
        weights = [1.0] * len(dirs) if weights is None else [float(weight) for weight in weights]
        if len(weights) != len(dirs):
            raise ValueError(f"weight 个数 {len(weights)} 与来源个数 {len(dirs)} 不一致")
        if any(weight < 0 for weight in weights):
            raise ValueError("数据来源的权重不能为负（0 表示不用它）")

        self._readers: list[BatchReader] = []
        sources: list[SourceView] = []
        kept_weights: list[float] = []
        duplicates: list[tuple[str, str]] = []
        seen: dict[str, str] = {}
        offset = 0
        for path, limit, weight in zip(dirs, limits, weights):
            reader = BatchReader(path)
            manifest = reader.manifest
            key, code = manifest.source_key(), manifest.source_code()
            if key in seen:
                # 同一来源列两次不是"两倍权重"，而是把同一批数据的切分与配额都算两遍；
                # 它一定是配置写错了（重复了一行、把续跑前后的目录都列上）。这里按身份去重，
                # 并把重复项留给调用方报出来——静默接受配置错误比报错贵得多。
                duplicates.append((key, path.name))
                reader.close()
                continue
            samples = reader.samples if limit <= 0 else min(limit, reader.samples)
            episodes = manifest.episodes if samples == reader.samples else _count_episodes(reader, samples)
            sources.append(SourceView(key, code, str(path), manifest, samples, episodes, offset))
            seen[key] = str(path)
            kept_weights.append(weight)
            offset += samples
            self._readers.append(reader)

        # 记录布局必须一致：gather 会把各来源的样本拼进同一个数组，而数组只有一个 dtype。
        # 混了 K=0 与 K=2 的批，读出来就会错位——那是最难查的一类错误（看起来像棋盘）。
        layouts = {reader.manifest.tree_nodes for reader in self._readers}
        if len(layouts) > 1:
            detail = "，".join(
                f"{reader.data_dir.name}（K={reader.manifest.tree_nodes}）" for reader in self._readers
            )
            raise ValueError(
                f"训练来源的记录布局不一致：{detail}。同一次训练只能用 tree_nodes 相同的批次"
                "（布局由批次的 tree_nodes 声明，见 ADR 0041）。"
            )
        self.dtype = self._readers[0].dtype if self._readers else record_dtype(0)
        self.tree_nodes = self._readers[0].manifest.tree_nodes if self._readers else 0
        self.tree_label_plays = self._readers[0].manifest.tree_label_plays if self._readers else 0
        self.sources: tuple[SourceView, ...] = tuple(sources)
        # 权重与来源一一对应，并且已经按去重结果对齐：调用方拿到的就是"实际会用到的那些来源"。
        self.weights: tuple[float, ...] = tuple(kept_weights)
        self.duplicates: tuple[tuple[str, str], ...] = tuple(duplicates)
        self._total = offset

    @property
    def samples(self) -> int:
        return self._total

    @property
    def episodes(self) -> int:
        """各来源局数之和。**不是**"不同局的个数"：局号是批内的，跨来源会重复。"""
        return sum(source.episodes for source in self.sources)

    def gather(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty(len(indices), dtype=self.dtype)
        if len(indices) == 0:
            return out
        bounds = np.asarray([source.offset for source in self.sources] + [self._total], dtype=np.int64)
        which = np.searchsorted(bounds, indices, side="right") - 1
        for index, reader in enumerate(self._readers):
            selected = np.flatnonzero(which == index)
            if selected.size:
                out[selected] = reader.gather(indices[selected] - self.sources[index].offset)
        return out

    def episode_column(self) -> np.ndarray:
        """每条样本的局号。跨来源拼接，所以只有配合 `sources[i].code` 才有身份意义。"""
        return np.concatenate(
            [reader.episode_column()[: source.samples] for reader, source in zip(self._readers, self.sources)]
        )

    def close(self) -> None:
        for reader in self._readers:
            reader.close()

    def describe(self) -> str:
        head = f"合并 {len(self.sources)} 个来源：{self.samples} 条 / {self.episodes} 局"
        return "\n".join([head, *(f"  {source.describe()}" for source in self.sources)])

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _count_episodes(reader: BatchReader, samples: int) -> int:
    """`limit` 截断后还有多少局（可能含一个不满的末局）。"""
    if samples <= 0:
        return 0
    return int(np.unique(reader.episode_column()[:samples]).size)


class SourceMix:
    """按权重在多个来源之间分配每轮样本数。

    语义是"每轮从各来源按权重抽这么多条"，而不是"每个来源跑几遍"：课程学习里常见的是
    把新数据小比例混进大池子，配额语义能直接表达它。

    **权重 0 等价于没列这个来源**：`samples_per_epoch` 只累加权重 > 0 的来源，
    权重 0 的来源也从不消耗随机数，所以把权重改成 0 不会让任何一条采样结果变化。
    （`test_weight_zero_source_is_invisible` 就钉住这条。）

    单一有效来源时退化成今天的行为：直接对整池做 `permutation`，一个随机数都不多用。
    """

    def __init__(self, pools, weights, samples_per_epoch: int | None = None, seed: int = 0):
        active = [(pool, float(weight)) for pool, weight in zip(pools, weights) if weight > 0 and len(pool)]
        if not active:
            raise ValueError("没有任何权重 > 0 且有样本的数据来源")
        self.pools = tuple(pool for pool, _ in active)
        self.weights = tuple(weight for _, weight in active)
        self.seed = int(seed)
        total = sum(len(pool) for pool in self.pools)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else total
        self._quotas = _largest_remainder(self.weights, self.samples_per_epoch)

    @classmethod
    def for_split(
        cls,
        reader: MultiSourceReader,
        indices: np.ndarray,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ) -> SourceMix:
        """从合并视图 + 训练下标构造采样器：各来源的池子就是它自己那段下标。

        权重取自 `reader.weights`（已按去重对齐），所以"配置里写了什么"与"实际用了什么"
        不会分叉。注意池子是**训练集**下标——验证集永远不参与采样，配额也就只按训练池算。
        """
        pools = []
        for source in reader.sources:
            start = source.offset
            stop = source.offset + source.samples
            # 来源在全局下标里是连续区间，用两次比较切出自己那段，不需要掩码数组。
            pools.append(indices[(indices >= start) & (indices < stop)])
        return cls(pools, reader.weights, samples_per_epoch=samples_per_epoch, seed=seed)

    def epoch_indices(self, epoch: int) -> np.ndarray:
        """这一轮的采样顺序（长度 = `samples_per_epoch`）。由 (seed, epoch) 完全决定。"""
        rng = np.random.default_rng((self.seed, epoch))
        if len(self.pools) == 1 and self._quotas[0] == len(self.pools[0]):
            return rng.permutation(self.pools[0])
        parts = []
        for pool, quota in zip(self.pools, self._quotas):
            if quota <= 0:
                continue
            # 配额超过池子时**允许有放回**：小池子被指定了高权重时，"每轮抽满配额"
            # 比"悄悄缩水"更符合配置的字面意思，也更容易被看见（同一局会在一轮里出现多次）。
            parts.append(rng.choice(pool, size=quota, replace=quota > len(pool)))
        order = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return rng.permutation(order)

    def describe(self) -> str:
        parts = [
            f"{index}（{len(pool)} 条，权重 {weight:g}，配额 {quota}）"
            for index, (pool, weight, quota) in enumerate(zip(self.pools, self.weights, self._quotas))
        ]
        return f"每轮 {self.samples_per_epoch} 条：" + "，".join(parts)


def _largest_remainder(weights, total: int) -> tuple[int, ...]:
    """按权重把 total 分成整数配额：先取整，余数按小数部分大的优先补。

    不用四舍五入再修差额：那样在来源多、权重相近时会系统性地少给排在前面的来源，
    而最大余数法给出的配额和恒等于 total。并列时按顺序决定，保证可复现。
    """
    weight_sum = float(sum(weights))
    exact = [total * weight / weight_sum for weight in weights]
    quotas = [int(value) for value in exact]
    remainder = total - sum(quotas)
    if remainder > 0:
        order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - quotas[index]), index))
        for index in order[:remainder]:
            quotas[index] += 1
    return tuple(quotas)


@dataclass
class Batch:
    decision: torch.Tensor  # (B, 16) uint8：决策局面
    afterstate: torch.Tensor  # (B, 16) uint8：移动后、生成块前
    action: torch.Tensor  # (B,) int64：实际执行的方向
    legal_mask: torch.Tensor  # (B,) uint8：bit a = 1 表示方向 a 合法
    visits: torch.Tensor  # (B, 4) float32：根访问数
    policy_target: torch.Tensor  # (B, 4) float32：归一化访问分布（无搜索时为全 0）
    value_target: torch.Tensor  # (B,) float32：归一化回报（分数 / score_scale）
    move_score: torch.Tensor  # (B,) float32：归一化立即得分
    episode: torch.Tensor  # (B,) int64
    # --- 候选级监督（schema v2）：每个合法方向一条，按方向下标对齐 ---
    candidate_afterstate: torch.Tensor  # (B, 4, 16) uint8
    candidate_value: torch.Tensor  # (B, 4) float32：A(u_a) = Q̄(a) − r(s,a)（已归一化）
    candidate_reward: torch.Tensor  # (B, 4) float32：r(s,a)，与 candidate_value 同量纲
    candidate_visits: torch.Tensor  # (B, 4) float32：该候选边的根访问数
    candidate_weight: torch.Tensor  # (B, 4) float32：sqrt(访问数)，非法/未访达/无候选处为 0
    # --- 树内节点块（schema v3；批次声明的 K=0 时这些维度为 0）---
    # 价值头在搜索里被**直接使用**的位置上的监督：叶节点与未被访问的边（那里的 Q 就是 r + A）。
    # 候选块只覆盖轨迹根，而搜索在根上有几百次模拟，自己就把排序算出来了（ADR 0035/0041）。
    tree_afterstate: torch.Tensor  # (B, K, 4, 16) uint8：该节点各候选的 afterstate
    tree_value: torch.Tensor  # (B, K, 4) float32：Q̄ − r（已归一化，与 candidate_value 同定义）
    tree_reward: torch.Tensor  # (B, K, 4) float32：r
    tree_visits: torch.Tensor  # (B, K, 4) float32：该节点上这条边的访问数
    tree_depth: torch.Tensor  # (B, K) float32：该节点到根经过的步数（诊断用）
    tree_rollout_value: torch.Tensor  # (B, K, 4) float32：**独立** playout 对 A(u_a) 的估计（已归一化）
    tree_legal: torch.Tensor  # (B, K, 4) float32：该槽位是否有效（缺席槽位与非法方向都是 0）
    tree_count: torch.Tensor  # (B,) int64：这条记录实际写了几个节点（0 = 没采到）

    def __len__(self) -> int:
        return int(self.decision.shape[0])

    def to(self, device: torch.device) -> Batch:
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
            candidate_reward=self.candidate_reward.to(device),
            candidate_visits=self.candidate_visits.to(device),
            candidate_weight=self.candidate_weight.to(device),
            tree_afterstate=self.tree_afterstate.to(device),
            tree_value=self.tree_value.to(device),
            tree_reward=self.tree_reward.to(device),
            tree_visits=self.tree_visits.to(device),
            tree_depth=self.tree_depth.to(device),
            tree_rollout_value=self.tree_rollout_value.to(device),
            tree_legal=self.tree_legal.to(device),
            tree_count=self.tree_count.to(device),
        )

    def candidate_q(self) -> torch.Tensor:
        """Q(s,a) = r(s,a) + A(u_a)：每个合法候选一条，非法/无候选处为 0。

        候选块存的正是这两项，所以每个根的**完整动作值向量**是重建的、不是新信息——
        策略改进算子（π̃ ∝ π̂·exp(Â/β)）因此能在训练侧就地实现，不必重新跑谱。
        """
        return self.candidate_reward + self.candidate_value

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
        reader: BatchReader | MultiSourceReader,
        indices: np.ndarray,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        score_scale: float | None = None,
        mix: SourceMix | None = None,
    ):
        self.reader = reader
        self.indices = indices
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.score_scale = float(score_scale if score_scale is not None else SCORE_SCALE)
        self.mix = mix
        # 每轮的样本数由采样器决定（多来源时配额之和），而不是由索引数组长度决定：
        # 否则 `len(stream)` 会与实际产出条数不符，epoch 数与进度都会对不上。
        self.samples_per_epoch = len(indices) if mix is None else mix.samples_per_epoch
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.mix is not None:
            # 采样器自己决定顺序（含打乱），不再叠加一次 shuffle：
            # 顺序由 (seed, epoch) 唯一确定，多来源与单来源走的是同一条可复现路径。
            order = self.mix.epoch_indices(self.epoch)
        else:
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
        tree_nodes = int(self.reader.tree_nodes)
        if tree_nodes > 0:
            # 每个节点块按 slot 拼成 (B, K, ...)：与 C++ 侧的偏移表逐项对应，不做逐条 Python 解码。
            tree_afterstate = np.stack(
                [records[f"tree{slot}_afterstate"] for slot in range(tree_nodes)], axis=1
            )
            tree_value = np.stack(
                [
                    (records[f"tree{slot}_value"].astype(np.float32) / self.score_scale)
                    for slot in range(tree_nodes)
                ],
                axis=1,
            )
            tree_reward = np.stack(
                [
                    (records[f"tree{slot}_reward"].astype(np.float32) / self.score_scale)
                    for slot in range(tree_nodes)
                ],
                axis=1,
            )
            tree_visits = np.stack(
                [records[f"tree{slot}_visits"].astype(np.float32) for slot in range(tree_nodes)], axis=1
            )
            tree_legal = np.stack(
                [
                    ((records[f"tree{slot}_legal_mask"][:, None] >> np.arange(4, dtype=np.uint8)) & 1)
                    for slot in range(tree_nodes)
                ],
                axis=1,
            ).astype(np.float32)
            tree_depth = np.stack(
                [records[f"tree{slot}_depth"].astype(np.float32) for slot in range(tree_nodes)], axis=1
            )
            tree_legal = tree_legal.astype(np.float32)
            tree_rollout_value = np.stack(
                [
                    records[f"tree{slot}_rollout"].astype(np.float32) / self.score_scale * tree_legal
                    for slot in range(tree_nodes)
                ],
                axis=1,
            )
            # 非法槽位清零：引擎写的是合法方向的真值，非法方向写 0；这里再乘一次掩码，
            # 让"非法方向的槽位恒为 0"成为下游可以直接依赖的性质（与候选块同一口径）。
            tree_value = tree_value * tree_legal
            tree_reward = tree_reward * tree_legal
            tree_visits = tree_visits * tree_legal
        else:
            tree_afterstate = np.zeros((len(records), 0, 4, 16), dtype=np.uint8)
            tree_value = np.zeros((len(records), 0, 4), dtype=np.float32)
            tree_reward = np.zeros((len(records), 0, 4), dtype=np.float32)
            tree_visits = np.zeros((len(records), 0, 4), dtype=np.float32)
            tree_depth = np.zeros((len(records), 0), dtype=np.float32)
            tree_rollout_value = np.zeros((len(records), 0, 4), dtype=np.float32)
            tree_legal = np.zeros((len(records), 0, 4), dtype=np.float32)
        return Batch(
            tree_afterstate=torch.from_numpy(np.ascontiguousarray(tree_afterstate)),
            tree_value=torch.from_numpy(np.ascontiguousarray(tree_value)),
            tree_reward=torch.from_numpy(np.ascontiguousarray(tree_reward)),
            tree_visits=torch.from_numpy(np.ascontiguousarray(tree_visits)),
            tree_depth=torch.from_numpy(np.ascontiguousarray(tree_depth)),
            tree_rollout_value=torch.from_numpy(np.ascontiguousarray(tree_rollout_value)),
            tree_legal=torch.from_numpy(np.ascontiguousarray(tree_legal)),
            tree_count=torch.from_numpy(records["tree_count"].astype(np.int64)),
            decision=torch.from_numpy(np.ascontiguousarray(records["decision"])),
            afterstate=torch.from_numpy(np.ascontiguousarray(records["afterstate"])),
            action=torch.from_numpy(records["action"].astype(np.int64)),
            legal_mask=torch.from_numpy(np.ascontiguousarray(legal_mask)),
            visits=torch.from_numpy(np.ascontiguousarray(visits)),
            policy_target=torch.from_numpy(np.ascontiguousarray(policy_target)),
            value_target=torch.from_numpy(
                (records["mc_return"].astype(np.float32) / self.score_scale).copy()
            ),
            move_score=torch.from_numpy((records["move_score"].astype(np.float32) / self.score_scale).copy()),
            episode=torch.from_numpy(records["episode"].astype(np.int64)),
            candidate_afterstate=torch.from_numpy(np.ascontiguousarray(records["candidate_afterstate"])),
            candidate_value=torch.from_numpy(
                (records["candidate_value"].astype(np.float32) / self.score_scale).copy()
            ),
            candidate_reward=torch.from_numpy(
                (records["candidate_reward"].astype(np.float32) / self.score_scale).copy()
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


def _single_source(reader: BatchReader) -> tuple[SourceView, ...]:
    """把裸的 BatchReader 看成"只有一个来源"的视图。

    单批用法（`BatchReader` + `make_split`，既有测试与临时脚本都这么用）不必被多来源的
    仪式感拖累，所以这条适配留在这里，而不是要求调用方先包一层 `MultiSourceReader`。
    """
    manifest = reader.manifest
    key, code = manifest.source_key(), manifest.source_code()
    return (SourceView(key, code, str(reader.data_dir), manifest, reader.samples, reader.episodes, 0),)


def open_sources(specs, val_fraction: float = 0.05, seed: int = 0) -> tuple[MultiSourceReader, Split]:
    """打开一个或多个来源并划分（训练的入口）：`specs` 是 `[(path, weight, limit)]`。

    权重不参与划分——它只决定每轮从各来源抽多少条（见 `SourceMix`）。
    """
    reader = MultiSourceReader(
        [path for path, _, _ in specs],
        [limit for _, _, limit in specs],
        [weight for _, weight, _ in specs],
    )
    split = make_split(reader, val_fraction=val_fraction, seed=seed)
    return reader, split


def source_summary(reader: BatchReader | MultiSourceReader) -> dict:
    """训练产物里的数据来源摘要：谁产出的、什么协议、多少条。

    逐来源记全字段（而不是只记一个总数）是复现的必要条件：一批权重是用哪些来源、按什么
    比例训出来的，只有这里说得清；`state` 也在里面——拿一个 paused 的批次训练过，
    与拿它跑完后的版本训练过，是两件事。
    """
    if isinstance(reader, MultiSourceReader):
        sources, weights = reader.sources, reader.weights
    else:
        sources, weights = _single_source(reader), (1.0,)
    return {
        "samples": reader.samples,
        "episodes": reader.episodes,
        "sources": [
            {
                "key": source.key,
                "name": source.name,
                "path": source.path,
                "weight": float(weight),
                "samples": source.samples,
                "episodes": source.episodes,
                "policy": source.manifest.policy,
                "simulations": source.manifest.simulations,
                "model_id": source.manifest.model_id,
                "build_id": source.manifest.build_id,
                "seed": source.manifest.seed,
                # state/shard_layout 进训练产物：日后回看一份权重是用什么数据训出来的，
                # 「这批数据当时跑完了没有」和「分片规则是哪一版」都在决定它能不能被复现。
                "state": source.manifest.state,
                "shard_layout": source.manifest.shard_layout,
            }
            for source, weight in zip(sources, weights)
        ],
        "manifests": {source.key: str(source.manifest.path) for source in sources},
    }
