"""训练器：价值头 / 策略头 / 联合 / 双塔联合，四种配方共用同一个循环。

口径全部沿用旧版被实测过的结论，它们各自都对应一次真实的失败：

* **标签**：价值回归的标签是 afterstate 的 MC 回报（`mc_return`，不含产生该 afterstate 的
  那次合并得分）；策略目标是搜索根访问分布 `visits / simulations`。两者语义不同，不能互相派生。
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import r2048_core

from r2048.data.dataset import Batch, BatchStream, batch_manifest_summary, open_batch
from r2048.nn.augment import augment_joint, augment_policy, augment_value
from r2048.nn.encode import encode_boards
from r2048.nn.model import (
    DUAL_TOWER_ARCHITECTURE,
    SHARED_ARCHITECTURE,
    create_model,
    freeze_value_path,
)

VALUE_MODES = ("value",)
POLICY_MODES = ("policy",)
JOINT_MODES = ("joint", "dual")
ALL_MODES = VALUE_MODES + POLICY_MODES + JOINT_MODES


def default_architecture(mode: str) -> str:
    """双塔是联合训练里被实测验证过的那一臂（旧版 ADR 0011）；单头训练用共享结构即可。"""
    return DUAL_TOWER_ARCHITECTURE if mode == "dual" else SHARED_ARCHITECTURE


@dataclass
class TrainConfig:
    data: str
    output: str
    mode: str = "value"
    init_checkpoint: str = ""
    architecture: str = ""
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
    policy_weight: float = 1.0
    val_fraction: float = 0.05
    split_seed: int = 0
    augment: bool = True
    max_samples: int = 0
    device: str = "cuda"
    seed: int = 0
    amp: str = "off"          # off | bf16
    channels_last: bool = True
    compile: bool = False
    log_every: int = 50
    resume: bool = False


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
    samples: int
    seconds: float

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


def config_digest(config: TrainConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:16]


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """torch.compile 会把原模块包一层 _orig_mod，取元数据与 state_dict 时要还原。"""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


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


def policy_loss(logits: torch.Tensor, batch: Batch) -> _PolicyStats | None:
    """软标签交叉熵（只统计有搜索记录的样本）+ 与搜索实际选招的一致率。"""
    has_search = batch.visits.sum(dim=-1) > 0
    if not bool(has_search.any()):
        return None
    legal = batch.legal_mask.bool()
    kept = logits[has_search]
    legal_kept = legal[has_search]
    masked = kept.masked_fill(~legal_kept, float("-inf"))
    target = batch.policy_target[has_search]
    log_probs = F.log_softmax(masked, dim=-1)
    # 非法方向上的 log 概率是 -inf，而目标在那些位置是 0：0 × (-inf) 直接得到 nan。
    # 先把非法项替换成 0，非法项对加权和就没有贡献了，合法项不受影响。
    log_probs = torch.where(legal_kept, log_probs, torch.zeros_like(log_probs))
    loss = (-(target * log_probs).sum(dim=-1)).mean()
    agreement = (target.argmax(dim=-1) == masked.argmax(dim=-1)).float().mean()
    return _PolicyStats(loss=loss, records=int(kept.shape[0]), agreement=float(agreement))


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

    def add_value(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.value_abs += (prediction.double() - target.double()).abs().sum()
        self.value_count += int(target.numel())

    def add_policy(self, stats: _PolicyStats) -> None:
        self._policy_ce_sum += float(stats.loss.detach()) * stats.records
        self._policy_agreement_sum += stats.agreement * stats.records
        self.policy_count += stats.records

    def value_mae(self) -> float:
        return float(self.value_abs.item() / self.value_count * self.score_scale) if self.value_count else float("nan")

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
) -> tuple[float, _Metrics, float]:
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
        batch = batch.to(device)
        if augment:
            if config.mode == "value":
                batch.afterstate = augment_value(batch.afterstate, rng)
            elif config.mode == "policy":
                batch.decision, batch.legal_mask, batch.policy_target = augment_policy(
                    batch.decision, batch.legal_mask, batch.policy_target, rng
                )
            else:
                (batch.decision, batch.afterstate, batch.legal_mask, batch.policy_target) = augment_joint(
                    batch.decision, batch.afterstate, batch.legal_mask, batch.policy_target, rng
                )

        with torch.set_grad_enabled(training):
            loss = torch.zeros((), dtype=torch.float32, device=device)
            if config.mode in JOINT_MODES:
                # 一次前向覆盖两种行：决策局面喂策略头、afterstate 喂价值头。
                split = batch.decision.shape[0]
                rows = torch.cat([batch.decision, batch.afterstate], dim=0)
                values, logits = _forward(model, rows, amp_dtype, config.channels_last)
                values = values.float()
                logits = logits.float()
                if want_value:
                    value_pred = values[split:]
                    value_loss = (
                        F.huber_loss(value_pred, batch.value_target)
                        if config.value_loss == "huber"
                        else F.mse_loss(value_pred, batch.value_target)
                    )
                    loss = loss + config.value_weight * value_loss
                    metrics.add_value(value_pred.detach(), batch.value_target)
                if want_policy:
                    stats = policy_loss(logits[:split], batch)
                    if stats is not None:
                        loss = loss + config.policy_weight * stats.loss
                        metrics.add_policy(stats)
            else:
                boards = batch.afterstate if want_value else batch.decision
                values, logits = _forward(model, boards, amp_dtype, config.channels_last)
                values = values.float()
                logits = logits.float()
                if want_value:
                    value_loss = (
                        F.huber_loss(values, batch.value_target)
                        if config.value_loss == "huber"
                        else F.mse_loss(values, batch.value_target)
                    )
                    loss = loss + config.value_weight * value_loss
                    metrics.add_value(values.detach(), batch.value_target)
                if want_policy:
                    stats = policy_loss(logits, batch)
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
        if training and config.log_every and step % config.log_every == 0:
            elapsed = time.monotonic() - started
            seen = (step + 1) * config.batch_size
            gpu = ""
            if device.type == "cuda":
                gpu = f" 显存 {torch.cuda.memory_allocated(device) / 2**20:.0f}MiB"
            print(
                f"\r{label} epoch {epoch + 1} step {step + 1}/{len(stream)} loss {float(loss):.4f} "
                f"{seen / max(1e-9, elapsed):.0f} 样本/秒{gpu}   ",
                end="",
                flush=True,
            )

    seconds = time.monotonic() - started
    if config.log_every:
        print()
    return float(loss_total.item() / max(1, batches)), metrics, seconds


def run_training(config: TrainConfig) -> TrainResult:
    if config.mode not in ALL_MODES:
        raise ValueError(f"未知训练模式: {config.mode}")
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

    reader, split = open_batch(config.data, val_fraction=config.val_fraction, seed=config.split_seed)
    train_indices, val_indices = split.train, split.val
    if config.max_samples > 0:
        train_indices = train_indices[: config.max_samples]
        val_indices = val_indices[: max(1, min(config.max_samples // 10, val_indices.size))]
    if train_indices.size == 0:
        raise ValueError("训练集为空：检查 val_fraction 与数据批内容")

    source_metadata: dict = {}
    if config.init_checkpoint:
        source = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        source_metadata = dict(source.get("metadata", {}))
        model = create_model(source_metadata)
        model.load_state_dict(source["model"])
    else:
        architecture = config.architecture or default_architecture(config.mode)
        model = create_model({"architecture": architecture})
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
    steps_per_epoch = max(1, len(train_indices) // config.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(config, steps_per_epoch * config.epochs)
    )

    train_stream = BatchStream(reader, train_indices, config.batch_size, seed=config.seed, shuffle=True)
    val_stream = (
        BatchStream(reader, val_indices, config.batch_size, seed=config.seed) if val_indices.size else None
    )
    architecture = source_metadata.get("architecture", config.architecture or default_architecture(config.mode))

    run_dir = Path(config.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    history: list[EpochMetrics] = []
    if config.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        history = [EpochMetrics(**entry) for entry in state.get("history", [])]
        print(f"续训：从第 {start_epoch + 1} 轮继续（已有 {len(history)} 轮记录）")

    print(reader.manifest.describe())
    print(split.describe())
    print(
        f"模式 {config.mode}，结构 {architecture}，可训练参数 {sum(p.numel() for p in trainable):,}，"
        f"冻结张量 {len(frozen)}，批 {config.batch_size}，设备 {device}{' bf16' if amp_dtype else ''}"
    )

    best_metric = float("inf")
    best_checkpoint = ""
    for epoch in range(start_epoch, config.epochs):
        train_loss, train_metrics, train_seconds = _run_epoch(
            model, train_stream, optimizer, scheduler, config, device, amp_dtype, score_scale, rng, epoch,
            label="训练",
        )
        val_loss, val_mae, val_ce, val_agreement, val_records = float("nan"), float("nan"), float("nan"), float("nan"), 0
        if val_stream is not None:
            val_loss, val_metrics, _ = _run_epoch(
                model, val_stream, None, scheduler, config, device, amp_dtype, score_scale, rng, epoch,
                label="验证",
            )
            val_mae, val_ce, val_agreement, val_records = (
                val_metrics.value_mae(),
                val_metrics.policy_ce(),
                val_metrics.agreement(),
                val_metrics.policy_count,
            )

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
            samples=int(train_indices.size),
            seconds=train_seconds,
        )
        history.append(entry)
        # 主指标按模式选：价值看 MAE，策略/联合看交叉熵。它只用于挑 checkpoint，
        # 真正的验收是同 seed 对局（离线指标测不出架构差别，见 ADR 0011）。
        primary = val_mae if config.mode in VALUE_MODES else val_ce
        print(
            f"第 {epoch + 1}/{config.epochs} 轮：训练损失 {train_loss:.4f}，验证损失 {val_loss:.4f} | "
            f"value MAE {val_mae:.1f} 分 | policy CE {val_ce:.4f}（一致率 {val_agreement * 100:.1f}%，"
            f"{val_records} 条） | {train_seconds:.1f}s（{entry.samples_per_second:.0f} 样本/秒）"
        )

        model_state = _unwrap(model).state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": [asdict(item) for item in history],
                "config": asdict(config),
            },
            run_dir / "last.pt",
        )
        if primary == primary and primary < best_metric:  # 非 NaN 且更优
            best_metric = primary
            best_checkpoint = str(run_dir / "model.pt")
            meta = _unwrap(model).metadata(
                model_id=f"{config.mode}-{Path(config.data).name}",
                parent=str(source_metadata.get("model_id", "")),
            )
            meta.update(
                {
                    "training_target": config.mode,
                    "value_frozen": bool(config.mode == "policy" and config.freeze_value),
                    "rules_version": int(r2048_core.RULES_VERSION),
                    "batch_id": reader.manifest.data.get("batch_id", ""),
                    "data_build_id": reader.manifest.build_id,
                    "config_digest": config_digest(config),
                    "metrics": {
                        "value_mae": entry.val_value_mae,
                        "policy_ce": entry.val_policy_ce,
                        "policy_agreement": entry.val_policy_agreement,
                    },
                }
            )
            if config.init_checkpoint:
                meta["source_checkpoint_sha256"] = file_digest(config.init_checkpoint)
            torch.save({"checkpoint_version": 1, "model": model_state, "metadata": meta}, best_checkpoint)

    train_json = str(run_dir / "train.json")
    Path(train_json).write_text(
        json.dumps(
            {
                "config": asdict(config),
                "config_digest": config_digest(config),
                "architecture": architecture,
                "frozen_parameters": len(frozen),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "data": batch_manifest_summary(reader.manifest),
                "data_manifest_sha256": file_digest(reader.manifest.path),
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
