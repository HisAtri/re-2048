"""训练器：价值头 / 策略头 / 联合 / 双塔联合，四种配方共用同一个循环。

口径全部沿用旧版被实测过的结论，它们各自都对应一次真实的失败：

* **标签**：价值回归的标签是 afterstate 的 MC 回报（`mc_return`，不含产生该 afterstate 的
  那次合并得分）；策略目标是搜索根访问分布 `visits / simulations`。两者语义不同，不能互相派生。
* **策略目标默认是"访问分布"本身**（纯蒸馏）：不动点就是教师。要越过教师，用
  `--policy-target improved`——一步策略改进 π̃ ∝ π̂·exp(Â/(β·同根 std))，见
  `improved_policy_target()` 与 ADR 0032。
* **按整局划分训练/验证**：同一局的相邻局面高度相似，按样本随机划分会让验证指标虚高。
* **策略损失排除无搜索样本**：`visits` 全 0 的记录没有策略目标，把它们当成“均匀分布”
  训练等于教网络输出均匀先验（旧版实测这条路与随机打平）。
* **策略头默认在冻结价值上训练**：共享主干联合优化会把 value MAE 从约 352 恶化到
  13,406 分，进而让搜索弱于纯 MCTS（旧版 ADR 0008）。`joint` / `dual` 保留为对照臂。
* **架构**：`dual` 用双塔（浅层共享 3 块 + 每塔 3 块），旧版同数据同配方对照里
  以 +5,367 分（95% CI [+3,560, +7,186]）胜出。
* **离线指标不是验收标准**：ADR 0011 里两种架构的 MAE/CE/同根排序几乎完全重合，
  差别只在同 seed 对局里显现。这里记录离线指标用于排查，验收放在评测阶段做。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import r2048_core
import torch
from torch.nn import functional as F

from r2048.data.dataset import (
    Batch,
    BatchStream,
    MultiSourceReader,
    SourceMix,
    open_sources,
    source_summary,
)
from r2048.data.shards import CANDIDATE_TARGET_DEFINITION
from r2048.fileio import save_torch_atomic
from r2048.jobs import finish_console, progress_payload
from r2048.nn.augment import augment_joint, augment_policy, augment_value
from r2048.nn.encode import BOARD_CELLS, encode_boards
from r2048.nn.model import (
    DUAL_TOWER_ARCHITECTURE,
    SHARED_ARCHITECTURE,
    architecture_conflict,
    create_model,
    freeze_value_path,
)

VALUE_MODES = ("value",)
POLICY_MODES = ("policy",)
JOINT_MODES = ("joint", "dual")
ALL_MODES = VALUE_MODES + POLICY_MODES + JOINT_MODES
# 策略目标的两种口径：visits = 搜索根访问分布（纯蒸馏，不动点是教师）；
# improved = 一步策略改进（π̃ ∝ π̂·exp(Â/β)），见 improved_policy_target()。
POLICY_TARGETS = ("visits", "improved")
# 价值头 checkpoint 的挑选口径：best_hit = A 口径最优候选命中率（历史口径，默认）；
# q_best_hit = r+A 口径（与引擎的选择值同口径）。见 ADR 0034/0035。
VALUE_SELECTIONS = ("best_hit", "q_best_hit")


def default_architecture(mode: str) -> str:
    """双塔是联合训练里被实测验证过的那一臂（旧版 ADR 0011）；单头训练用共享结构即可。"""
    return DUAL_TOWER_ARCHITECTURE if mode == "dual" else SHARED_ARCHITECTURE


def trains_policy(mode: str) -> bool:
    """这个配方训不训策略头（决定 checkpoint 里该不该记 `policy_target`）。"""
    return mode in POLICY_MODES or mode in JOINT_MODES


@dataclass(frozen=True)
class DatasetSpec:
    """一个训练数据来源：路径 + 采样权重 + 可选条数上限。

    权重是"每轮按它分配采样配额"的相对量，不是"这个来源跑几遍"：课程学习里要表达的是
    "把新数据小比例混进大池子"，配额语义能直接写出它。**权重 0 等于不列这个来源**
    （样本数不参与每轮长度，也不消耗随机数），所以关掉一个来源不会改变别的来源的采样。
    """

    path: str
    weight: float = 1.0
    limit: int = 0  # 只用该来源的前 N 条（0 = 不限）


@dataclass
class TrainConfig:
    datasets: list[DatasetSpec] = field(default_factory=list)
    output: str = ""
    mode: str = "value"
    init_checkpoint: str = ""
    # 版本 ID（= 用哪个网络定义）。超参走 architecture_params，**不由 ID 反推**（ADR 0024）：
    # 于是"改宽度/改深度"是改这里的数字，ID 只在出现新拓扑形状时才新增。
    architecture: str = ""
    architecture_params: dict = field(default_factory=dict)
    freeze_value: bool = False
    epochs: int = 20
    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.0
    lr_schedule: str = "cosine"
    warmup_steps: int = 200
    grad_clip: float = 1.0
    value_loss: str = "huber"
    value_weight: float = 1.0
    # auto：有候选监督就用候选目标（推荐），否则退回主轨迹 MC 回报；
    # candidate：强制候选；trajectory：强制主轨迹（用于复现旧结果）。
    value_target: str = "auto"
    policy_weight: float = 1.0
    # 策略目标口径（见 improved_policy_target）：visits = 纯蒸馏（默认，历史口径）；
    # improved = 一步策略改进 π̃ ∝ π̂·exp(Â/(β·scale))。
    policy_target: str = "visits"
    # β 是**无量纲**的（乘在同根 Â 的标准差上）。离线审计（scripts/advantage_probe.py）
    # 在 greedy 教师批上的量级：β=1 改变 6% 的决策、KL 0.10；β=0.25 改变 9%、KL 0.30。
    improve_beta: float = 1.0
    # π̂ 上的均匀伪访问数：让搜索未访问过的候选也能拿到质量。0 = 严格乘法重加权。
    improve_floor: float = 0.0
    val_fraction: float = 0.05
    # Q 空间的同根成对排序损失权重（0 = 关）。Q = r + A 是引擎的选择值：把 r 加在
    # 预测与目标两边**不改变预测误差**（所以 Huber 在两种口径下逐值相同），但会改变
    # **成对的标签**——在 r6 的批上 12.9% 的根给出不同的 argmax（ADR 0034）。因此
    # "在 Q 空间做排序监督"与"在 A 空间做"不是同一件事，而 Q 空间的排序目标恰好是
    # 教师自己的边均值 Q：对搜索输出最忠实的蒸馏。见 ADR 0035。
    value_ranking_weight: float = 0.0
    # 价值头 checkpoint 的挑选依据：best_hit = A 口径的最优候选命中率（历史口径）；
    # q_best_hit = r+A 口径（与引擎的选择值同口径，见 ADR 0034）。只影响"留哪个 epoch 的
    # 权重"，不影响损失——因为价值头原理上只能学 A（r 不是 afterstate 的函数）。
    value_selection: str = "best_hit"
    split_seed: int = 0
    # 每轮采样多少条（0 = 各来源训练池之和）。只在多来源或显式想缩短轮长时才有意义。
    samples_per_epoch: int = 0
    augment: bool = True
    max_samples: int = 0
    device: str = "cuda"
    seed: int = 0
    amp: str = "off"  # off | bf16
    channels_last: bool = True
    compile: bool = False
    log_every: int = 50
    resume: bool = False
    # 每多少步写一次可续跑的 checkpoint（0 = 只用时间兜底，见 SAVE_INTERVAL_SECONDS）。
    # 按步数保存能让"崩了之后丢多少"可预期（评测/小实验用），长跑则靠时间兜底。
    save_every_steps: int = 0
    # 结构化进度回调（见 r2048.jobs）：训练动辄几十分钟到几小时，外部观察者
    # （脚本、界面）不该去解析那一行覆盖式打印。给了回调就不再打印。
    # 回调粒度与 log_every 一致——每步回调一次是没必要的开销（会按步写状态文件）。
    on_progress: Callable[[dict], None] | None = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_value_mae: float
    train_policy_ce: float
    val_loss: float
    val_value_mae: float
    val_policy_ce: float
    val_policy_agreement: float
    val_policy_records: int
    val_candidate_mae: float
    val_candidate_pair_agreement: float
    val_candidate_best_hit: float
    val_candidate_regret: float
    samples: int
    seconds: float
    # 同一套指标在 r+A 口径下的值（搜索的选择值就是 r + A，见 ADR 0034）。
    # 为什么两个口径都要报：A 口径量"价值头预测得准不准"，r+A 口径量"引擎会不会选对"，
    # 而加 r 不改变预测误差、只改变排序——两者差多少必须能从产物里读出来。
    val_candidate_q_pair_agreement: float = float("nan")
    val_candidate_q_best_hit: float = float("nan")
    val_candidate_q_regret: float = float("nan")
    # 排序项的加权均值（分数单位）：只在 value_ranking_weight > 0 时有值。
    ranking_loss: float = float("nan")
    # 这一轮价值头实际回归的是哪个量（"candidate" / "trajectory"；未训价值头时为空）。
    # 为什么必须显式记录：同一个 mode + 同一批数据在两种价值目标下会训出**不同**的权重，
    # 而 model_id 只由 (mode, 数据标签) 组成——实测两个不同的模型因为沿用旧写法而同名，
    # 评分池按 model_id 去重时会把它们当成同一个。标识与元数据都必须能区分。
    value_target: str = ""
    # 这一轮策略头学的是哪个目标（"visits" / "improved"；未训策略头时为空）。
    # 与 value_target 同一条纪律：不同目标训出的是**不同权重**，必须能从产物本身读出来。
    policy_target: str = ""

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else 0.0


@dataclass
class TrainResult:
    run_dir: str
    best_checkpoint: str
    last_checkpoint: str
    metrics: list[EpochMetrics] = field(default_factory=list)
    frozen_parameters: int = 0
    train_json: str = ""


def config_payload(config: TrainConfig) -> dict:
    """可序列化的配置快照：进度回调是运行期接进来的，不属于配置。

    它必须与配置摘要、落盘共用同一份口径——回调进不了 JSON，一旦混进去，
    带回调跑训练（也就是走 CLI 的每一次）会在第一次写 checkpoint 时崩掉。
    """
    payload = asdict(config)
    payload.pop("on_progress", None)
    return payload


def config_digest(config: TrainConfig) -> str:
    return hashlib.sha256(json.dumps(config_payload(config), sort_keys=True).encode()).hexdigest()[:16]


def data_label(config: TrainConfig) -> str:
    """写进 model_id 的数据标签：单来源用目录名（与既有 checkpoint 的命名习惯一致），
    多来源只报个数——把三个来源的目录名拼进 model_id 只会让标识长到没人读。"""
    if len(config.datasets) == 1:
        return Path(config.datasets[0].path).name
    return f"{len(config.datasets)}src"


def model_id_for(config: TrainConfig, value_target: str) -> str:
    """写进 checkpoint 的 model_id。

    价值目标不是默认值时进名字：`value-<数据标签>` 这个旧写法让"候选目标"和"主轨迹回报"
    训出的两份**不同权重**同名，而评分池按 model_id 去重、按 model_id 分组——同名就会被
    当成同一个模型（实测已发生，见 ADR 0030）。名字只给人看，但它必须是**可区分**的。
    策略目标同理：`improved` 与 `visits` 是同一条线上两份不同权重，名字里必须看得出 β。
    """
    if value_target == "trajectory" and config.mode in ("value", "joint", "dual"):
        base = f"{config.mode}-trajectory-{data_label(config)}"
    else:
        base = f"{config.mode}-{data_label(config)}"
    if config.policy_target != "visits":
        # β 进名字：它能改变 6%~9% 的决策，是一个会被反复扫的旋钮，不能只靠 metadata 区分。
        base = f"{base}+{config.policy_target}-b{config.improve_beta:g}"
    if config.value_selection != "best_hit":
        # 同一批数据、同一个种子、只换挑选口径 ⇒ 两份不同权重。名字不分就无法事后区分。
        base = f"{base}+{config.value_selection}"
    if config.value_ranking_weight > 0:
        base = f"{base}+rank{config.value_ranking_weight:g}"
    return base


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """torch.compile 会把原模块包一层 _orig_mod，取元数据与 state_dict 时要还原。"""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# 断点续跑的写入间隔上限（秒）。按步数保存是可选开关，但**默认必须有时间兜底**：
# 一个 epoch 可能跑几小时，只在轮末落盘等于"崩一次白干一轮"（这正是 P3b 之前的行为）。
SAVE_INTERVAL_SECONDS = 600.0
# 续跑时"这一份 checkpoint 是什么时候写的"以外的字段：它们不是训练语义，
# 变化不该触发警报（`resume` 就是用户在按的那个开关，保存间隔与停止在哪一步无关）。
RESUME_VOLATILE_FIELDS = ("resume", "save_every_steps")


def rng_state(rng: torch.Generator, device: torch.device) -> dict:
    """把所有会消耗随机数的发生器状态抓下来。

    缺任何一项，续跑出来的就不是同一条轨迹：增强的目标下标来自 `rng`，
    模型里将来若加入 dropout/噪声则来自全局 torch 状态，numpy 侧同理。
    """
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "augment": rng.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict, rng: torch.Generator, device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    rng.set_state(state["augment"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def data_identity(reader: MultiSourceReader, config: TrainConfig) -> dict:
    """续跑闸门里的**数据身份**：来源 + 权重 + 划分口径。

    刻意**不含**样本数与路径：就地续跑会让批次一直增长（ADR 0020），换个目录挂同一份数据
    也不该被当成换了数据。而权重与 split 口径变了就是真的换了池子（配额与划分都会不同），
    必须拦住——"数据换了却接着训"是一类不会报错的静默错误。
    """
    return {
        "sources": [[source.key, float(weight)] for source, weight in zip(reader.sources, reader.weights)],
        "val_fraction": float(config.val_fraction),
        "split_seed": int(config.split_seed),
    }


def config_differences(stored: dict, current: dict) -> list[str]:
    """逐字段列出配置差异（"哪一项变了"比"摘要不同"可行动，见 ADR 0020 决策 7）。"""
    differences = []
    for key in sorted(set(stored) | set(current)):
        if key in RESUME_VOLATILE_FIELDS:
            continue
        before, after = stored.get(key, "<缺>"), current.get(key, "<缺>")
        if before != after:
            differences.append(f"{key}: {before} → {after}")
    return differences


def _make_lr_lambda(config: TrainConfig, total_steps: int):
    warmup = max(1, config.warmup_steps)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if config.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


@dataclass
class _PolicyStats:
    loss: torch.Tensor
    records: int
    agreement: float


def improved_policy_target(batch: Batch, beta: float, floor: float) -> torch.Tensor:
    """一步策略改进的目标：π̃_a ∝ π̂_a · exp(Â_a / (β · scale))。

    * Â_a = Q_a − V，V = Σ_a π̂_a Q_a（on-policy 基线）；Q_a = r_a + A(u_a) 由候选块重建。
    * `scale` 是**同根** Â 的无权重标准差。用无权重而不是 π̂ 加权：加权 std 会把"访问数怎么
      分配"混进尺度，而尺度只该描述这个根的候选之间差多少；且无权重 std 与 ADR 0030 的
      "同根标签 std"同定义，离线审计的数字可以直接对上。
    * **β 无量纲是硬约束**：Â 的绝对量级由标签来源决定（ADR 0030 §4.1 实测同一族价值头在
      不同教师下量级差 10 倍）。固定 β 等于让"这一轮的标签量级"顺带决定"目标有多尖"，
      两个本该独立的旋钮被焊在一起——ADR 0013 与 §4.1 的两次退化都是这个形态。
    * `floor` 是 π̂ 上的均匀伪访问数：纯乘法重加权只能把质量在**搜索访问过的**边之间重分配，
      未访问边永远拿不到质量（ADR 0030 §6.6 实测 NN 自举批有 0.8%~7.7% 的合法边访问数为 0）。
    * β→∞ 时逐样本等于 π̂（回归测试钉住这条）；没有候选监督的记录原样返回 π̂，因此
      混入随机策略批次时不会把垃圾写进目标。

    这一步改进的保证（E_{π̃}[Q] ≥ E_{π̂}[Q]）是在**同一个 Q 下**成立的，用生成 Â 的那个 Q
    度量它没有信息量——算子的有效性只能由训练轮 + 评测臂回答（ADR 0032）。
    """
    has_candidates = batch.candidate_weight.sum(dim=-1) > 0
    if not bool(has_candidates.any()):
        return batch.policy_target
    legal = batch.legal_mask.bool()
    visits = batch.visits + floor
    pi = visits / visits.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    q = batch.candidate_q()
    value = (pi * q * legal).sum(dim=-1, keepdim=True)
    advantage = (q - value) * legal
    count = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = advantage.sum(dim=-1, keepdim=True) / count
    var = ((advantage - mean) ** 2 * legal).sum(dim=-1, keepdim=True) / count
    scale = var.clamp_min(0.0).sqrt().clamp_min(1e-9)
    # 减去同根最大权重再取指数：溢出防护，且不影响归一化后的分布（同根之差被抵消）。
    exponent = advantage / (beta * scale)
    neg_inf = torch.full_like(exponent, float("-inf"))
    exponent = exponent - torch.where(legal, exponent, neg_inf).max(dim=-1, keepdim=True).values
    weights = torch.where(legal, exponent, neg_inf).exp()
    improved = pi * weights
    improved = improved / improved.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.where(has_candidates[:, None], improved, batch.policy_target)


def policy_loss(logits: torch.Tensor, batch: Batch, target: torch.Tensor) -> _PolicyStats | None:
    """软标签交叉熵（只统计有搜索记录的样本）+ 与搜索实际选招的一致率。

    `target` 由调用方给出（`visits` 或改进后的 π̃），不再从 batch 里现取：改进目标必须在
    D4 增强**之前**算好，否则候选块的方向下标与增强后的棋盘就不再对齐（ADR 0010 决策 4）。
    """
    has_search = batch.visits.sum(dim=-1) > 0
    if not bool(has_search.any()):
        return None
    legal = batch.legal_mask.bool()
    kept = logits[has_search]
    legal_kept = legal[has_search]
    masked = kept.masked_fill(~legal_kept, float("-inf"))
    target = target[has_search]
    log_probs = F.log_softmax(masked, dim=-1)
    # 非法方向上的 log 概率是 -inf，而目标在那些位置是 0：0 × (-inf) 直接得到 nan。
    # 先把非法项替换成 0，非法项对加权和就没有贡献了，合法项不受影响。
    log_probs = torch.where(legal_kept, log_probs, torch.zeros_like(log_probs))
    loss = (-(target * log_probs).sum(dim=-1)).mean()
    agreement = (target.argmax(dim=-1) == masked.argmax(dim=-1)).float().mean()
    return _PolicyStats(loss=loss, records=int(kept.shape[0]), agreement=float(agreement))


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    total = float(denominator)
    return float(numerator) / total if total > 0 else float("nan")


def _mean_score(total: torch.Tensor, count: torch.Tensor, score_scale: float) -> float:
    value = float(count)
    return float(total) / value * score_scale if value > 0 else float("nan")


def _ranking_totals(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """一次排序统计：(成对一致、成对不一致、最优命中、参与命中统计的根数、后悔值之和)。

    抽成函数是因为它要跑两遍（A 口径与 r+A 口径）——两遍的差别只是整体平移了 r，
    所以只有"谁排第一"会变，代码必须完全相同，否则两个口径就不是同一个度量了。
    """
    concordant = torch.zeros((), dtype=torch.float64, device=device)
    discordant = torch.zeros((), dtype=torch.float64, device=device)
    best_hit = torch.zeros((), dtype=torch.float64, device=device)
    best_total = torch.zeros((), dtype=torch.float64, device=device)
    regret = torch.zeros((), dtype=torch.float64, device=device)
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pair_valid = valid[:, first] & valid[:, second] & (target[:, first] != target[:, second])
        if not bool(pair_valid.any()):
            continue
        agree = (prediction[:, first] - prediction[:, second]).sign() == (
            target[:, first] - target[:, second]
        ).sign()
        concordant += (agree & pair_valid).double().sum()
        discordant += (~agree & pair_valid).double().sum()

    has_two = valid.sum(dim=1) >= 2
    if bool(has_two.any()):
        masked_prediction = prediction.masked_fill(~valid, float("-inf"))
        masked_target = target.masked_fill(~valid, float("-inf"))
        picked = masked_prediction.argmax(dim=1)
        best = masked_target.argmax(dim=1)
        best_hit += (picked[has_two] == best[has_two]).double().sum()
        best_total += has_two.double().sum()
        picked_value = target.gather(1, picked[:, None]).squeeze(1)
        regret += (masked_target.max(dim=1).values[has_two] - picked_value[has_two]).double().sum()
    return concordant, discordant, best_hit, best_total, regret


class _Metrics:
    """在设备上累计，只在写日志时同步一次：每步 `.item()` 会把 GPU 拖成串行。"""

    def __init__(self, device: torch.device, score_scale: float):
        self.device = device
        self.score_scale = score_scale
        self.value_abs = torch.zeros((), dtype=torch.float64, device=device)
        self.value_count = 0
        self._policy_ce_sum = 0.0
        self._policy_agreement_sum = 0.0
        self.policy_count = 0
        # 候选级监督的指标：加权误差 + 同根成对排序一致率 + 最优候选命中 + 后悔值
        self.candidate_weight = torch.zeros((), dtype=torch.float64, device=device)
        self.candidate_abs_error = torch.zeros((), dtype=torch.float64, device=device)
        self.concordant = torch.zeros((), dtype=torch.float64, device=device)
        self.discordant = torch.zeros((), dtype=torch.float64, device=device)
        self.best_hit = torch.zeros((), dtype=torch.float64, device=device)
        self.best_total = torch.zeros((), dtype=torch.float64, device=device)
        self.regret = torch.zeros((), dtype=torch.float64, device=device)
        # 同一套排序指标的 r+A 口径（q_ 前缀）：搜索的选择值是 r + A，所以"引擎会不会选对"
        # 只能在这一套上量。两套口径共用 valid 掩码，因此根集合与权重完全相同、可直接对比。
        self.q_concordant = torch.zeros((), dtype=torch.float64, device=device)
        self.q_discordant = torch.zeros((), dtype=torch.float64, device=device)
        self.q_best_hit = torch.zeros((), dtype=torch.float64, device=device)
        self.q_best_total = torch.zeros((), dtype=torch.float64, device=device)
        self.q_regret = torch.zeros((), dtype=torch.float64, device=device)
        # 排序项自己的量级（加权均值，分数单位）：权重取值合理不合理只能从这里看出来。
        self._ranking_sum = 0.0
        self._ranking_weight = 0.0
        # 价值头这一轮回归的量（空集 = 没训价值头）。由真正做决策的那一行写入，
        # 不在别处重新推导一遍；用集合是因为多来源混训时不同批可能落到不同的量上，
        # 那种情况必须在记录里看得见，而不是被最后一个批覆盖掉。
        self.value_targets_used: set[str] = set()

    def add_ranking(self, ranking: torch.Tensor, batch: Batch) -> None:
        """累计排序项（按这一批的候选质量加权，与损失里的口径一致）。"""
        weight = float(batch.candidate_weight.sum())
        self._ranking_sum += float(ranking) * weight
        self._ranking_weight += weight

    def ranking_loss(self) -> float:
        """排序项的加权均值（**分数单位**，与候选 MAE 同一量纲，便于比较两者量级）。"""
        if self._ranking_weight <= 0:
            return float("nan")
        return self._ranking_sum / self._ranking_weight * self.score_scale

    def add_value(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.value_abs += (prediction.double() - target.double()).abs().sum()
        self.value_count += int(target.numel())

    def add_policy(self, stats: _PolicyStats) -> None:
        self._policy_ce_sum += float(stats.loss.detach()) * stats.records
        self._policy_agreement_sum += stats.agreement * stats.records
        self.policy_count += stats.records

    def add_candidates(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        reward: torch.Tensor | None = None,
    ) -> None:
        """候选预测与搜索给出的候选目标对比。

        排序指标用“同根成对一致率”而不是 Spearman：每个根只有 2~4 个候选，
        成对一致率等价于 Kendall tau 的分子，且在候选数少时更稳定、更容易解释。

        `reward` 是每个候选的立即合并得分 r(s,a)：给了它，就**同时**累计一套 r+A 口径的
        排序指标（成对一致率 / 最优命中 / 后悔值）。两套口径回答两个不同的问题：

        * **A 口径**（不加 r）＝"价值头把 afterstate 价值预测得准不准"。这是价值头该学的量
          ——它的输入是 afterstate `u = T(s,a)`，而 `r(s,a)` **不是 u 的函数**（同一个
          afterstate 既能由"合并出 4"产生、也能由"本来就有 4 只是平移"产生），所以网络
          原理上无法预测 r+A，r 必须在查询时由规则加上。
        * **r+A 口径**＝"引擎会不会选对"。引擎的选择值是 `one_ply = r + A`
          （`cpp/search/batched_engine.cpp`），所以只有加上 r 之后排序才与搜索同一口径。

        两个口径在真数据上的差别不小（r 能改写 3%~13% 的 argmax，全是价值几乎并列的根），
        而**加 r 不改变预测误差**——`(pred+r) − (target+r) = pred − target`，所以 MAE/后悔值
        的量级不变，变的只有"谁排第一"。见 ADR 0034。
        """
        valid = weight > 0
        self.candidate_weight += weight.double().sum()
        self.candidate_abs_error += ((prediction - target).double().abs() * weight.double()).sum()

        ranking = _ranking_totals(prediction, target, valid, self.device)
        self.concordant += ranking[0]
        self.discordant += ranking[1]
        self.best_hit += ranking[2]
        self.best_total += ranking[3]
        self.regret += ranking[4]
        if reward is not None:
            q_ranking = _ranking_totals(prediction + reward, target + reward, valid, self.device)
            self.q_concordant += q_ranking[0]
            self.q_discordant += q_ranking[1]
            self.q_best_hit += q_ranking[2]
            self.q_best_total += q_ranking[3]
            self.q_regret += q_ranking[4]

    def candidate_mae(self) -> float:
        if float(self.candidate_weight) <= 0:
            return float("nan")
        return float((self.candidate_abs_error / self.candidate_weight).item() * self.score_scale)

    def pair_agreement(self) -> float:
        return _ratio(self.concordant, self.concordant + self.discordant)

    def best_candidate_hit(self) -> float:
        return _ratio(self.best_hit, self.best_total)

    def candidate_regret(self) -> float:
        return _mean_score(self.regret, self.best_total, self.score_scale)

    def q_pair_agreement(self) -> float:
        """r+A 口径的同根成对一致率（与搜索实际排序同一口径）。"""
        return _ratio(self.q_concordant, self.q_concordant + self.q_discordant)

    def q_best_candidate_hit(self) -> float:
        """r+A 口径的最优候选命中率：预测的 argmax(r+A) 是否等于目标的 argmax(r+A)。"""
        return _ratio(self.q_best_hit, self.q_best_total)

    def q_candidate_regret(self) -> float:
        """r+A 口径的后悔值：按预测的 argmax(r+A) 走，比最优候选少拿多少 Q（分数单位）。"""
        return _mean_score(self.q_regret, self.q_best_total, self.score_scale)

    def value_mae(self) -> float:
        return (
            float(self.value_abs.item() / self.value_count * self.score_scale)
            if self.value_count
            else float("nan")
        )

    def policy_ce(self) -> float:
        return self._policy_ce_sum / self.policy_count if self.policy_count else float("nan")

    def agreement(self) -> float:
        return self._policy_agreement_sum / self.policy_count if self.policy_count else float("nan")


def _forward(
    model: torch.nn.Module,
    boards: torch.Tensor,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = encode_boards(boards.to(dtype=torch.long))
    if channels_last:
        # 卷积网络在通道最后的内存布局下通常更快；这里只对编码后的 4D 特征做转换，
        # 棋盘本身是 (N,16) 的二维索引张量，没有“通道最后”的概念。
        encoded = encoded.contiguous(memory_format=torch.channels_last)
    if amp_dtype is not None:
        with torch.autocast(device_type=encoded.device.type, dtype=amp_dtype):
            return model(encoded)
    return model(encoded)


def _value_loss(
    predictions: torch.Tensor,
    batch: Batch,
    use_candidates: bool,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """价值损失。

    候选模式：在**全部合法候选**上回归搜索给出的 A(u_a)，按 sqrt(访问数) 加权——
    搜索在每个决策节点都会查询这些候选，而主轨迹只记录实际走过的那个。
    主轨迹模式：退回 afterstate 的 MC 回报（无候选监督的数据，例如随机策略批次）。
    返回 (loss, 用于指标的预测张量)；候选模式下后者已 reshape 成 (B, 4)。
    """
    if not use_candidates:
        prediction = predictions.reshape(-1)
        target = batch.value_target
        loss = (
            F.huber_loss(prediction, target)
            if config.value_loss == "huber"
            else F.mse_loss(prediction, target)
        )
        return loss, prediction.detach()

    count = batch.decision.shape[0]
    prediction = predictions.reshape(count, int(r2048_core.NUM_ACTIONS))
    weight = batch.candidate_weight
    per_candidate = (
        F.huber_loss(prediction, batch.candidate_value, reduction="none")
        if config.value_loss == "huber"
        else (prediction - batch.candidate_value) ** 2
    )
    total_weight = weight.sum().clamp_min(1e-9)
    return (per_candidate * weight).sum() / total_weight, prediction.detach()


# 排序项的"让步界限"（分数单位）：预测的 Q 间隔只要超过它，这一对就算学对了。
# 取 20 分 ≈ 这条线上候选 MAE 的一半（r6 的验证候选 MAE 约 40 分）——比它更小的间隔本来就
# 埋在价值头自己的误差里，不值得去追。它只影响近并列的那些对，不影响"符号错"的惩罚。
_RANKING_MARGIN_POINTS = 20.0


def _ranking_loss(prediction: torch.Tensor, batch: Batch, score_scale: float) -> torch.Tensor:
    """Q 空间的同根成对排序项：按 Q 间隔加权的**软成对后悔**。

    为什么在 Q = r + A 空间做：引擎的选择值就是 `one_ply = r + A`（ADR 0034）。把 r 加在
    预测与目标两边**不改变预测误差**——`(pred+r) − (target+r) = pred − target`，所以 Huber
    在两种口径下逐值相同，这也是"价值头只能训 A"的由来（r 不是 afterstate 的函数，网络学
    不到它）。**但成对的标签会变**：A 口径比 `target_a vs target_b`，Q 口径比
    `target_a + r_a vs target_b + r_b`，在 r6 的批上 12.9% 的根给出不同的 argmax。所以这一项
    不是"换个度量再训一遍"，它喂给网络的监督本身不同。

    形状：对每根的每对合法候选，令 g = 目标 Q 间隔、h = 预测 Q 间隔，取

        |g| · τ · softplus(−h · sign(g) / τ)

    * `τ·softplus(x/τ)` 是 `relu(x)` 的平滑版（τ→0 就是它），所以这一项**在极限上等于成对
      后悔**：符号错就罚 |g|·|h|，符号对（且间隔超过 τ）就趋 0。用 |g| 当权重 = "按这一对值
      多少分罚"，于是它的量纲与 Huber 的 MAE 可比，`value_ranking_weight` 有直接含义
      （1.0 = 一分软后悔与一分 MAE 同权）。
    * 只看**符号**而不要求间隔匹配：决策只取决于 argmax，间隔差多少不影响走哪一步。
    * 目标间隔为 0 的对不计入（没有可学的排序）。
    """
    margin = _RANKING_MARGIN_POINTS / score_scale
    count = batch.decision.shape[0]
    predicted = prediction.reshape(count, int(r2048_core.NUM_ACTIONS))
    valid = batch.candidate_weight > 0
    q_predicted = predicted + batch.candidate_reward
    q_target = batch.candidate_value + batch.candidate_reward
    total = torch.zeros((), dtype=torch.float32, device=prediction.device)
    used = torch.zeros(count, dtype=torch.bool, device=prediction.device)
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        pair_valid = valid[:, first] & valid[:, second]
        if not bool(pair_valid.any()):
            continue
        gap_target = (q_target[:, first] - q_target[:, second]) * pair_valid
        pair_weight = gap_target.abs()
        usable = pair_valid & (pair_weight > 0)
        if not bool(usable.any()):
            continue
        gap_predicted = (q_predicted[:, first] - q_predicted[:, second]) * pair_valid
        pair_loss = margin * F.softplus(-gap_predicted * gap_target.sign() / margin)
        total = total + (pair_loss * pair_weight)[usable].sum()
        used = used | usable
    # 按**根**归一化：这一项于是是"每一步决策的软后悔（分数）"，与候选 MAE（每个候选的误差，
    # 分数）同一量纲、同一量级，`value_ranking_weight` 才有可比的含义。按间隔和归一化会被
    # "间隔大、本来就容易对"的那些对稀释，同一个错误报出来的数会小一个量级。
    return total / used.double().sum().clamp_min(1.0)


def _add_ranking(
    metrics: _Metrics, config: TrainConfig, prediction: torch.Tensor, batch: Batch
) -> torch.Tensor:
    """算排序项并把它加进损失；顺带把"排序项自己有多大"记进指标。

    为什么必须单独记账：这一项与 Huber 的量纲都是"分数"，但两者的实际量级差几倍到几十倍，
    权重取值是否合理只能从这两个数上读出来——否则"加了排序项"会变成一个看不见的配方改动。
    """
    if config.value_ranking_weight <= 0:
        return torch.zeros((), dtype=torch.float32, device=prediction.device)
    ranking = _ranking_loss(prediction, batch, metrics.score_scale)
    metrics.add_ranking(ranking.detach(), batch)
    return config.value_ranking_weight * ranking


def _run_epoch(
    model: torch.nn.Module,
    stream: BatchStream,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    config: TrainConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    score_scale: float,
    rng: torch.Generator,
    epoch: int,
    label: str,
    skip: int = 0,
    on_step: Callable[[int, int, float], None] | None = None,
) -> tuple[float, _Metrics, float]:
    """跑一轮。

    `skip` 是"这一轮已经训过的批数"（断点续跑用）：这些批**只能被跳过，不能被重训**，
    否则参数会被更新两次，最终权重与一次跑完不再相同。跳过时**不增强**——
    增强会消耗随机数，而 checkpoint 里的随机数状态正是"第 skip 批之前"的状态。
    顺序由 `(seed, epoch)` 完全决定（`BatchStream`），所以"跳过前 skip 批"接下来看到的
    就是当时那一批。

    `on_step(epoch, done, loss)` 在每次优化器更新之后调用，供外层落盘使用。
    """
    training = optimizer is not None
    model.train(training)
    metrics = _Metrics(device, score_scale)
    loss_total = torch.zeros((), dtype=torch.float64, device=device)
    batches = 0
    want_value = config.mode in VALUE_MODES or config.mode in JOINT_MODES
    want_policy = config.mode in POLICY_MODES or config.mode in JOINT_MODES
    # 验证集绝不能做增强：那会让指标在“变换后的分布”上度量，既噪声大又不可比。
    augment = config.augment and training

    if training:
        stream.set_epoch(epoch)
    started = time.monotonic()
    for step, batch in enumerate(stream):
        if step < skip:
            continue
        batch = batch.to(device)
        # 改进目标必须在增强**之前**从原始候选块算出来：增强会按方向重排棋盘、合法掩码与
        # 目标，而候选块（candidate_value/reward/visits）是"按方向下标对齐"的，先增强棋盘
        # 再从候选块算目标，方向就对不上了。算好之后把目标交给增强函数一起变换即可。
        target = (
            improved_policy_target(batch, config.improve_beta, config.improve_floor)
            if config.policy_target == "improved"
            else batch.policy_target
        )
        if augment:
            if config.mode == "value":
                batch.afterstate = augment_value(batch.afterstate, rng)
            elif config.mode == "policy":
                batch.decision, batch.legal_mask, target = augment_policy(
                    batch.decision, batch.legal_mask, target, rng
                )
            else:
                (batch.decision, batch.afterstate, batch.legal_mask, target) = augment_joint(
                    batch.decision, batch.afterstate, batch.legal_mask, target, rng
                )

        with torch.set_grad_enabled(training):
            loss = torch.zeros((), dtype=torch.float32, device=device)
            # 候选级监督：价值头训练在**搜索实际查询过的全部合法候选**上，而不是只训练实际
            # 走过的那个动作。权重 sqrt(访问数) 让搜索更看重的候选获得更高权重，
            # 未访问过的候选权重为 0（它们没有估计，不能当成“估计为 0”）。
            use_candidates = want_value and config.value_target != "trajectory" and batch.has_candidates
            if config.value_target == "candidate" and want_value and not batch.has_candidates:
                raise ValueError("要求候选目标，但这一批数据没有候选监督（检查数据是否由搜索产出）")
            if config.policy_target == "improved" and want_policy and not batch.has_candidates:
                # 没有候选块就无法重建 Q，"改进"会安静地退化成原访问分布——那等于
                # "改了参数却没生效"，必须变成显式失败（同 value_target=candidate 的口径）。
                raise ValueError("要求改进的策略目标，但这一批数据没有候选监督（检查数据是否由搜索产出）")
            if want_value:
                metrics.value_targets_used.add("candidate" if use_candidates else "trajectory")
            count = batch.decision.shape[0]

            if config.mode in JOINT_MODES:
                # 一次前向覆盖两种行：决策局面喂策略头、候选 afterstate 喂价值头。
                if use_candidates:
                    candidate_rows = batch.candidate_afterstate.reshape(-1, BOARD_CELLS)
                    rows = torch.cat([batch.decision, candidate_rows], dim=0)
                else:
                    rows = torch.cat([batch.decision, batch.afterstate], dim=0)
                values, logits = _forward(model, rows, amp_dtype, config.channels_last)
                values = values.float()
                logits = logits.float()
                if want_value:
                    value_loss, value_pred = _value_loss(values[count:], batch, use_candidates, config)
                    loss = loss + config.value_weight * value_loss
                    if use_candidates:
                        metrics.add_candidates(
                            value_pred, batch.candidate_value, batch.candidate_weight, batch.candidate_reward
                        )
                        loss = loss + _add_ranking(metrics, config, values[count:], batch)
                    else:
                        metrics.add_value(value_pred, batch.value_target)
                if want_policy:
                    stats = policy_loss(logits[:count], batch, target)
                    if stats is not None:
                        loss = loss + config.policy_weight * stats.loss
                        metrics.add_policy(stats)
            else:
                if want_value and use_candidates:
                    boards = batch.candidate_afterstate.reshape(-1, BOARD_CELLS)
                else:
                    boards = batch.afterstate if want_value else batch.decision
                values, logits = _forward(model, boards, amp_dtype, config.channels_last)
                values = values.float()
                logits = logits.float()
                if want_value:
                    value_loss, value_pred = _value_loss(values, batch, use_candidates, config)
                    loss = loss + config.value_weight * value_loss
                    if use_candidates:
                        metrics.add_candidates(
                            value_pred, batch.candidate_value, batch.candidate_weight, batch.candidate_reward
                        )
                        loss = loss + _add_ranking(metrics, config, values, batch)
                    else:
                        metrics.add_value(value_pred, batch.value_target)
                if want_policy:
                    stats = policy_loss(logits, batch, target)
                    if stats is not None:
                        loss = loss + config.policy_weight * stats.loss
                        metrics.add_policy(stats)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], config.grad_clip
                    )
                optimizer.step()
                scheduler.step()

        loss_total += loss.detach().double()
        batches += 1
        if training and on_step is not None:
            # 落盘检查在统计之后、进度回调之前：回调可能抛异常（测试里的"杀掉"就是这样），
            # 那时 checkpoint 必须已经反映刚刚完成的这一步。
            on_step(epoch, step + 1, float(loss.detach()))
        if training and config.log_every and step % config.log_every == 0:
            elapsed = time.monotonic() - started
            seen = (step + 1) * config.batch_size
            gpu = ""
            if device.type == "cuda":
                gpu = f" 显存 {torch.cuda.memory_allocated(device) / 2**20:.0f}MiB"
            payload = progress_payload(
                "train",
                step + 1,
                len(stream),
                "steps",
                started,
                line=(
                    f"{label} epoch {epoch + 1} step {step + 1}/{len(stream)} loss {float(loss.detach()):.4f} "
                    f"{seen / max(1e-9, elapsed):.0f} 样本/秒{gpu}"
                ),
                extra={
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "mode": config.mode,
                    "samples_per_second": round(seen / max(1e-9, elapsed), 1),
                },
            )
            if config.on_progress is None:
                print("\r" + payload["line"] + "   ", end="", flush=True)
            else:
                config.on_progress(payload)

    seconds = time.monotonic() - started
    if config.log_every and config.on_progress is None:
        finish_console()
    return float(loss_total.item() / max(1, batches)), metrics, seconds


def run_training(config: TrainConfig) -> TrainResult:
    if config.mode not in ALL_MODES:
        raise ValueError(f"未知训练模式: {config.mode}")
    if config.policy_target not in POLICY_TARGETS:
        raise ValueError(f"未知策略目标口径: {config.policy_target}，可选 {list(POLICY_TARGETS)}")
    if config.value_ranking_weight < 0:
        raise ValueError(f"value_ranking_weight 必须 ≥ 0（0 = 关），收到 {config.value_ranking_weight}")
    if config.value_selection not in VALUE_SELECTIONS:
        raise ValueError(f"未知 checkpoint 挑选口径: {config.value_selection}，可选 {list(VALUE_SELECTIONS)}")
    if config.policy_target == "improved" and config.improve_beta <= 0:
        # β 是"用多少倍同根 std 当温度"，0 或负数会把目标直接退化成 argmax（且负 β 会
        # 反过来奖励差的候选），这是配置错误而不是一种配方。
        raise ValueError(f"improve_beta 必须 > 0，收到 {config.improve_beta}")
    if config.mode == "policy" and config.freeze_value and not config.init_checkpoint:
        raise ValueError(
            "冻结价值路径时必须给出 --init-checkpoint：在一个随机初始化的价值头上只训策略头没有意义"
        )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed & 0xFFFFFFFF)

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("请求了 cuda 但不可用，回退到 cpu")
        device = torch.device("cpu")
    amp_dtype = torch.bfloat16 if config.amp == "bf16" and device.type == "cuda" else None
    # 增强用的随机数发生器必须与张量同设备：在 CUDA 张量上生成随机下标时，
    # 传一个 CPU 发生器会直接报错（生成器按设备区分）。
    rng = torch.Generator(device=device.type).manual_seed(config.seed)
    score_scale = float(r2048_core.SCORE_SCALE)

    if not config.datasets:
        raise ValueError(
            "必须给出训练数据：TrainConfig.datasets 至少要有一个来源（CLI 是 --data 或 --config）"
        )
    reader, split = open_sources(
        [(spec.path, spec.weight, spec.limit) for spec in config.datasets],
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )
    if reader.duplicates:
        repeated = "，".join(f"{name}（{key}）" for key, name in reader.duplicates)
        print(f"警告：以下来源在配置里重复出现，已按来源身份去重：{repeated}")
    train_indices, val_indices = split.train, split.val
    if config.max_samples > 0:
        # 调试用的全局上限：按全局下标顺序截断（多来源时就是按配置顺序先喂满前面的来源）。
        train_indices = train_indices[: config.max_samples]
        val_indices = val_indices[: max(1, min(config.max_samples // 10, val_indices.size))]
    if train_indices.size == 0:
        raise ValueError("训练集为空：检查 val_fraction 与数据批内容")
    if config.val_fraction > 0 and val_indices.size == 0:
        # 验证集空掉时不会报错，只会让所有验证指标变成 NaN、让"最佳 checkpoint"永远选不出来
        # （实测：5% × 50 局的批次有约 8% 的概率一局都选不进验证集）。这类安静的空转
        # 必须变成显式的失败，或者由调用方显式声明 val_fraction=0。
        raise ValueError(
            f"验证集为空：val_fraction={config.val_fraction} 在 {reader.episodes} 局（逐来源："
            + "，".join(f"{source.episodes} 局" for source in reader.sources)
            + "）里一局都没选到。提高 val_fraction、换 split_seed，或显式设 val_fraction=0 关闭验证。"
        )
    # 单来源不套采样器：走原来的“整池打乱”，一个随机数都不多用。多来源（或显式指定轮长）
    # 才需要按权重分配配额。采样器在截断之后建，池子就一定是训练集的下标子集。
    mix = (
        SourceMix.for_split(
            reader, train_indices, samples_per_epoch=config.samples_per_epoch, seed=config.seed
        )
        if len(reader.sources) > 1 or config.samples_per_epoch > 0
        else None
    )
    data_summary = source_summary(reader)
    # 单来源保留历史上那两个键（评测脚本与旧产物比对在用）；多来源不挑一个来源冒充整体，
    # 身份一律以 sources 列表为准。
    only_manifest = reader.sources[0].manifest if len(reader.sources) == 1 else None
    single_batch_id = str(only_manifest.data.get("batch_id", "")) if only_manifest else ""
    single_build_id = only_manifest.build_id if only_manifest else ""

    source_metadata: dict = {}
    if config.init_checkpoint:
        source = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        source_metadata = dict(source.get("metadata", {}))
        # 从 checkpoint 初始化时，结构由这份权重自己决定：请求里如果写了**不同的**架构或超参，
        # 它们会被静默忽略（"我明明改了宽度"是最难查的一类困惑），所以直接拒绝。
        # 与 checkpoint 完全一致的写法是允许的——同一条训练线的 YAML 会被反复带上。
        conflict = architecture_conflict(config.architecture, config.architecture_params, source_metadata)
        if conflict is not None:
            raise ValueError(
                f"从 checkpoint 初始化时不能改结构：{conflict}。"
                f"这份权重是 {source_metadata.get('architecture')} / "
                f"{source_metadata.get('architecture_params')}；要换结构就新起一条训练线。"
            )
        model = create_model(source_metadata)
        model.load_state_dict(source["model"])
    else:
        architecture = config.architecture or default_architecture(config.mode)
        model = create_model(
            {"architecture": architecture, "architecture_params": config.architecture_params}
        )
    model = model.to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)

    frozen: list[str] = []
    if config.mode == "policy" and config.freeze_value:
        frozen = freeze_value_path(model)

    if config.compile:
        model = torch.compile(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    train_stream = BatchStream(
        reader, train_indices, config.batch_size, seed=config.seed, shuffle=True, mix=mix
    )
    val_stream = (
        BatchStream(reader, val_indices, config.batch_size, seed=config.seed) if val_indices.size else None
    )
    # 轮长取采样器报出的每轮样本数，而不是训练下标数组的长度：多来源时两者不等
    # （配额之和 vs 池子大小），用错了会让学习率日程按错误的总步数铺开。
    steps_per_epoch = max(1, train_stream.samples_per_epoch // config.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(config, steps_per_epoch * config.epochs)
    )

    architecture = source_metadata.get(
        "architecture", config.architecture or default_architecture(config.mode)
    )

    run_dir = Path(config.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    # 每轮的批数：drop_last=False 时最后不足一批也算一步，所以直接用流长度（与调度器的
    # `steps_per_epoch` 相差至多一步，那是既有口径，不在 P3b 里改）。
    batches_per_epoch = len(train_stream)
    step_in_epoch = 0
    start_epoch = 0
    history: list[EpochMetrics] = []
    best_metric: float | None = None
    best_metric_kind = ""
    best_checkpoint = ""
    identity = data_identity(reader, config)
    if config.resume and not (run_dir / "last.pt").exists():
        # 说了要接着跑、却什么也没接着，是必须说出来的一件事：安静地重头开始，
        # 用户会以为自己在续跑（盘上还会多出一份"看起来是延续"的产物）。
        print(f"--resume 指定了，但 {run_dir / 'last.pt'} 不存在：从头开始训练")
    if config.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        # 闸门一：数据身份。换数据接着训是静默错误（划分与标签口径都变了，
        # 而损失照样下降），所以直接拒绝并说清是哪一项不同。
        stored_identity = state.get("data_identity")
        if stored_identity is not None and stored_identity != identity:
            raise ValueError(
                "拒绝续跑：数据身份与 last.pt 不一致。\n"
                f"  上一段：{stored_identity}\n"
                f"  这一次：{identity}\n"
                "要换数据（来源或权重）就换一个 --output 新起一条训练线；"
                "如果只是同一批数据长大了一些，身份不会变，可以直接续跑。"
            )
        # 闸门二：其余配置。允许"把轮数调长"这类改动，但必须让人看见。
        differences = config_differences(dict(state.get("config", {})), config_payload(config))
        if differences:
            print("配置与上一段不同（数据身份未变，继续续跑）：")
            for line in differences:
                print(f"  {line}")

        _unwrap(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        # 恢复调度器会把 `base_lrs` 一起恢复，于是配置里的 lr 会被悄悄忽略（实测：把 lr 从
        # 1e-3 改成 1000 续跑，实际仍按 1e-3 训练，验证指标一模一样）。
        # **状态从 checkpoint 恢复，超参以配置为准**：按配置重算当前步位置上的学习率，
        # 而不是把 lr 直接设成 config.lr——调度器保存下来的 lr 已经是"基准 × 日程"的值，
        # 直接覆盖会让第一步用错（少了日程因子）。同一个 lr 时这是恒等运算，不影响逐张量复现。
        scheduler.base_lrs = [config.lr for _ in scheduler.base_lrs]
        resumed_lambda = _make_lr_lambda(config, steps_per_epoch * config.epochs)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * resumed_lambda(scheduler.last_epoch)
        history = [EpochMetrics(**entry) for entry in state.get("history", [])]
        # 恢复到"哪一轮、这一轮第几批"：同一个 globals 位置续跑，权重才会与一次跑完逐张量相同。
        start_epoch = int(state.get("epoch", len(history) - 1 if history else 0))
        step_in_epoch = int(state.get("step_in_epoch", 0))
        if "global_step" not in state:
            # P3b 之前的 checkpoint 只有轮末状态：没有随机数状态，也没有批内位置，
            # 只能从下一轮的轮首接着跑（等价于以前的行为，不是逐张量可复现）。
            start_epoch += 1
            step_in_epoch = 0
            print(f"续训：这份 checkpoint 没有随机数状态，从第 {start_epoch + 1} 轮轮首继续")
        if step_in_epoch >= batches_per_epoch:
            start_epoch += 1
            step_in_epoch = 0
        best_metric = state.get("best_metric")
        best_metric_kind = str(state.get("best_metric_kind", ""))
        # 续跑是接着这一条线跑，所以"当前的最佳 checkpoint"就是上一段选出来的那个文件。
        # 不恢复它（保持空串）会让"这一段没有哪一轮更好"时的返回值丢掉一个真实存在的路径——
        # 调用方（CLI、迭代编排）拿到的空路径会被当成"这条线没有最佳模型"。
        if (run_dir / "model.pt").exists():
            best_checkpoint = str(run_dir / "model.pt")
        if "rng" in state:
            restore_rng(state["rng"], rng, device)
        resume_position = (
            f"第 {start_epoch + 1} 轮第 {step_in_epoch + 1} 批"
            if step_in_epoch
            else f"第 {start_epoch + 1} 轮轮首"
        )
        print(f"续训：从{resume_position}继续（已有 {len(history)} 轮记录）")
        stored_samples = state.get("data_samples")
        current_samples = {source.key: source.samples for source in reader.sources}
        if stored_samples is not None and stored_samples != current_samples:
            # 批次原地增长（ADR 0020）是常态：来源身份没变，但池子变大会改变划分与每轮内容。
            # 这不是错误，只是"续跑不等于一次跑完"的又一个来源，必须说出来。
            print(
                f"注意：数据来源的样本数与上一段不同 {stored_samples} → {current_samples}；"
                "池子变了，划分与每轮内容可能与上一段不同（来源身份未变，不属于换数据）"
            )

    print(reader.describe())
    print(split.describe())
    if mix is not None:
        print(mix.describe())
    print(
        f"模式 {config.mode}，结构 {architecture}，可训练参数 {sum(p.numel() for p in trainable):,}，"
        f"冻结张量 {len(frozen)}，批 {config.batch_size}，设备 {device}{' bf16' if amp_dtype else ''}"
    )
    if config.policy_target != "visits":
        # 目标口径必须在启动时打印出来：它与默认口径训出的权重不同，日志里看不见就等于
        # 事后无法判断"这一份是怎么训出来的"。
        print(
            f"策略目标 {config.policy_target}：π̃ ∝ π̂·exp(Â/(β·同根 std))，"
            f"β={config.improve_beta:g}，floor={config.improve_floor:g}"
        )

    # 主指标的方向取决于“这一轮用没用候选监督”，只有进入循环才知道（`best_metric` 与
    # `best_metric_kind` 已在续跑块里恢复，不能在这里清零——那等于每次续跑都重启最佳的选择）。
    last_save = time.monotonic()

    def save_last(epoch: int, done: int) -> None:
        """写可续跑的 `last.pt`（原子替换）：轮内的位置 + 随机数状态 + 当前最佳值。

        这是"崩一次只丢一个保存间隔"的全部依据，也是"续跑等于一次跑完"的依据：
        它记录的是**批边界**上的状态，`_run_epoch` 的 `skip` 能精确回到那里。
        """
        nonlocal last_save
        save_torch_atomic(
            run_dir / "last.pt",
            {
                # 版本 2 = 带批内位置与随机数状态（版本 1 只在轮末写、只能从轮首续跑）。
                "checkpoint_version": 2,
                "epoch": epoch,
                "step_in_epoch": done,
                "global_step": epoch * batches_per_epoch + done,
                "model": _unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": [asdict(item) for item in history],
                "config": config_payload(config),
                "data_identity": identity,
                "data_samples": {source.key: source.samples for source in reader.sources},
                "best_metric": best_metric,
                "best_metric_kind": best_metric_kind,
                "rng": rng_state(rng, device),
            },
        )
        last_save = time.monotonic()

    def on_step(epoch: int, done: int, _loss: float) -> None:
        by_steps = config.save_every_steps and done % config.save_every_steps == 0
        if by_steps or time.monotonic() - last_save >= SAVE_INTERVAL_SECONDS:
            save_last(epoch, done)

    for epoch in range(start_epoch, config.epochs):
        # 只有续跑后的第一轮需要跳过：之后每一轮都从轮首开始。
        skip = step_in_epoch if epoch == start_epoch else 0
        train_loss, train_metrics, train_seconds = _run_epoch(
            model,
            train_stream,
            optimizer,
            scheduler,
            config,
            device,
            amp_dtype,
            score_scale,
            rng,
            epoch,
            label="训练",
            skip=skip,
            on_step=on_step,
        )
        nan = float("nan")
        val_loss, val_mae, val_ce, val_agreement, val_records = nan, nan, nan, nan, 0
        candidate_mae, pair_agreement, best_hit, regret = nan, nan, nan, nan
        q_pair_agreement, q_best_hit, q_regret = nan, nan, nan
        ranking = nan
        if val_stream is not None:
            val_loss, val_metrics, _ = _run_epoch(
                model,
                val_stream,
                None,
                scheduler,
                config,
                device,
                amp_dtype,
                score_scale,
                rng,
                epoch,
                label="验证",
            )
            val_mae, val_ce, val_agreement, val_records = (
                val_metrics.value_mae(),
                val_metrics.policy_ce(),
                val_metrics.agreement(),
                val_metrics.policy_count,
            )
            candidate_mae = val_metrics.candidate_mae()
            pair_agreement = val_metrics.pair_agreement()
            best_hit = val_metrics.best_candidate_hit()
            regret = val_metrics.candidate_regret()
            # r+A 口径：与引擎的选择值同口径（ADR 0034）。只在数据带候选时才有值。
            q_pair_agreement = val_metrics.q_pair_agreement()
            q_best_hit = val_metrics.q_best_candidate_hit()
            q_regret = val_metrics.q_candidate_regret()
            ranking = val_metrics.ranking_loss()

        entry = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_value_mae=train_metrics.value_mae(),
            train_policy_ce=train_metrics.policy_ce(),
            val_loss=val_loss,
            val_value_mae=val_mae,
            val_policy_ce=val_ce,
            val_policy_agreement=val_agreement,
            val_policy_records=val_records,
            val_candidate_mae=candidate_mae,
            val_candidate_pair_agreement=pair_agreement,
            val_candidate_best_hit=best_hit,
            val_candidate_regret=regret,
            val_candidate_q_pair_agreement=q_pair_agreement,
            val_candidate_q_best_hit=q_best_hit,
            val_candidate_q_regret=q_regret,
            ranking_loss=ranking,
            samples=int(train_indices.size),
            seconds=train_seconds,
            value_target="+".join(sorted(train_metrics.value_targets_used)),
            policy_target=config.policy_target if trains_policy(config.mode) else "",
        )
        history.append(entry)
        # 主指标：**供应给搜索的价值头必须用同根候选排序来挑，而不是全局 MAE**
        # （旧版的结论：全局 MAE 无法替代同根候选排序，架构差别就藏在后者里）。
        # 有候选监督时用“最优候选命中率”，否则退回价值 MAE；策略模式看交叉熵。
        if config.mode == "policy":
            primary, maximize, kind = val_ce, False, "val_policy_ce"
        elif config.value_selection == "q_best_hit":
            if math.isnan(q_best_hit):
                # 要求按 r+A 口径挑、却没有候选监督 ⇒ 无从计算。静默退回 A 口径（或 MAE）
                # 等于"改了参数却没生效"，必须显式失败。
                raise ValueError(
                    f"要求按 {config.value_selection} 口径挑 checkpoint，但这一轮没有候选监督"
                    "（需要搜索产出的数据批）"
                )
            primary, maximize, kind = q_best_hit, True, "val_candidate_q_best_hit"
        elif not math.isnan(best_hit):  # 非 NaN 说明这一轮确实用了候选监督
            primary, maximize, kind = best_hit, True, "val_candidate_best_hit"
        else:
            primary, maximize, kind = val_mae, False, "val_value_mae"
        if best_metric_kind and best_metric_kind != kind:
            # 主指标换了口径（换了配方，或换了一批没有候选监督的数据）：旧的"最佳"不可比。
            # 继续拿它比大小会安静地锁死最佳 checkpoint 的选择，所以丢弃并说明。
            print(f"上一段的最佳值来自 {best_metric_kind}，本段是 {kind}：不沿用旧的最佳值")
            best_metric, best_metric_kind = None, kind
        best_metric_kind = kind
        ranking_note = f"，排序项 {ranking:.1f} 分/根" if not math.isnan(ranking) else ""
        print(
            f"第 {epoch + 1}/{config.epochs} 轮：训练损失 {train_loss:.4f}，验证损失 {val_loss:.4f} | "
            f"候选 MAE {candidate_mae:.1f} 分，A 口径 一致率 {pair_agreement * 100:.1f}%/命中 {best_hit * 100:.1f}%"
            f"/后悔 {regret:.1f} 分，r+A 口径 一致率 {q_pair_agreement * 100:.1f}%/命中 {q_best_hit * 100:.1f}%"
            f"/后悔 {q_regret:.1f} 分{ranking_note} | "
            f"策略 CE {val_ce:.4f}（一致率 {val_agreement * 100:.1f}%） | "
            f"{train_seconds:.1f}s（{entry.samples_per_second:.0f} 样本/秒）"
        )

        model_state = _unwrap(model).state_dict()
        # `primary` 为 NaN 表示这一轮没算出主指标（例如验证集里没有可用样本），
        # 显式写出来而不是用 `x != x` 的自比较写法。
        improved = not math.isnan(primary) and (
            best_metric is None or (primary > best_metric if maximize else primary < best_metric)
        )
        if improved:
            best_metric = primary
            best_checkpoint = str(run_dir / "model.pt")
            meta = _unwrap(model).metadata(
                model_id=model_id_for(config, entry.value_target),
                parent=str(source_metadata.get("model_id", "")),
            )
            meta.update(
                {
                    "training_target": config.mode,
                    "value_frozen": bool(config.mode == "policy" and config.freeze_value),
                    "rules_version": int(r2048_core.RULES_VERSION),
                    # 数据身份以**逐来源**的形式记：多来源训练时，单独一个 batch_id 说不清
                    # 这份权重吃过什么。单来源时同时保留 batch_id/data_build_id，让既有
                    # 读取方（评测脚本、旧产物比对）继续能拿到熟悉的那两个键。
                    "sources": data_summary["sources"],
                    "batch_id": single_batch_id,
                    "data_build_id": single_build_id,
                    "config_digest": config_digest(config),
                    "value_target": entry.value_target or "none",
                    # 价值头是按哪个口径挑出来的（两份权重只差这个旋钮）。
                    "value_selection": config.value_selection,
                    "value_ranking_weight": float(config.value_ranking_weight),
                    # 策略目标的口径与它的两个旋钮必须落在产物里：它是"这一份策略头是纯蒸馏
                    # 还是被一步改进过"的唯一凭据，而两者训出的是不同权重（同 model_id 陷阱）。
                    "policy_target": entry.policy_target or "none",
                    "improve_beta": float(config.improve_beta),
                    "improve_floor": float(config.improve_floor),
                    "candidate_target_definition": CANDIDATE_TARGET_DEFINITION,
                    "metrics": {
                        "value_mae": entry.val_value_mae,
                        "candidate_mae": entry.val_candidate_mae,
                        "candidate_pair_agreement": entry.val_candidate_pair_agreement,
                        "candidate_best_hit": entry.val_candidate_best_hit,
                        "candidate_regret": entry.val_candidate_regret,
                        # 引擎的选择值是 r + A，所以"会不会选对"看这一组（ADR 0034）。
                        "candidate_q_pair_agreement": entry.val_candidate_q_pair_agreement,
                        "candidate_q_best_hit": entry.val_candidate_q_best_hit,
                        "candidate_q_regret": entry.val_candidate_q_regret,
                        "ranking_loss": entry.ranking_loss,
                        "policy_ce": entry.val_policy_ce,
                        "policy_agreement": entry.val_policy_agreement,
                    },
                }
            )
            if config.init_checkpoint:
                meta["source_checkpoint_sha256"] = file_digest(config.init_checkpoint)
            save_torch_atomic(
                best_checkpoint, {"checkpoint_version": 1, "model": model_state, "metadata": meta}
            )

        # 轮末的 last.pt 写在最佳值更新之后：它是"续跑从这里接着跑"的锚点，
        # 里头的 best_metric 必须是这一轮挑选之后的值（否则续跑会把最佳选择的进度退回去）。
        save_last(epoch, batches_per_epoch)

    train_json = str(run_dir / "train.json")
    Path(train_json).write_text(
        json.dumps(
            {
                "config": config_payload(config),
                "config_digest": config_digest(config),
                "architecture": architecture,
                "frozen_parameters": len(frozen),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "data": data_summary,
                # 逐来源的 manifest 摘要：manifest 是"这批数据有没有被改过"的唯一凭据，
                # 多来源时它必须逐来源记，否则事后无法判断到底是哪一批变了。
                "data_manifest_sha256": {
                    source.key: file_digest(source.manifest.path) for source in reader.sources
                },
                "split": {"train": int(train_indices.size), "val": int(val_indices.size)},
                "history": [asdict(item) for item in history],
                "best_checkpoint": best_checkpoint,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return TrainResult(
        run_dir=str(run_dir),
        best_checkpoint=best_checkpoint,
        last_checkpoint=str(run_dir / "last.pt"),
        metrics=history,
        frozen_parameters=len(frozen),
        train_json=train_json,
    )
