"""训练流程。

一次 ``handwash train`` 会完整落盘以下产物（报告与复现全靠它们）
------------------------------------------------------------------
::

    outputs/<run_name>/
      resolved_config.yaml    最终生效的完整配置 + config_hash
      git_info.json           git commit / 分支 / 是否脏工作区
      env_info.json           torch / cuda / 设备
      run_log.jsonl           逐 epoch 指标（可直接画曲线）
      history.json            训练历史汇总
      models/best.pt          按 val 指标选出的最佳 checkpoint
      models/last.pt          最后一轮（便于中途中断后续训）
      model_summary.json      参数量等模型信息
      manifest_summary.json   数据规模与类别分布

训练循环刻意写得直白（不用 Lightning / Trainer 框架）：
课程项目需要每个人都能读懂并修改每一步，"能看懂"比"少写 20 行"更重要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from handwash.core.config import ResolvedConfig, dump_config
from handwash.core.labels import get_label_space
from handwash.core.metrics import macro_f1
from handwash.core.seeding import seed_everything
from handwash.errors import TrainingError
from handwash.io.utils import append_jsonl, write_json
from handwash.logging import get_logger
from handwash.models.checkpoint import save_checkpoint
from handwash.paths import BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, checkpoint_path, ensure_dir
from handwash.pipelines.common import (
    build_manifest_loader,
    describe_device,
    maybe_enable_tf32,
    resolve_device,
    resolve_split_records,
)

__all__ = ["TrainResult", "train", "EpochMetrics", "evaluate_loader"]

log = get_logger(__name__)


@dataclass
class EpochMetrics:
    """单轮指标。"""

    epoch: int
    phase: str  # train | val
    loss: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "phase": self.phase,
            "loss": round(self.loss, 6),
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
        }


@dataclass
class TrainResult:
    """训练结果：CLI 用它打印摘要，测试用它断言。"""

    run_dir: Path
    best_checkpoint: Path | None
    best_metric: float
    epochs_run: int
    history: list[EpochMetrics] = field(default_factory=list)
    model_summary: dict[str, Any] = field(default_factory=dict)
    used_synthetic_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "best_metric": self.best_metric,
            "epochs_run": self.epochs_run,
            "history": [m.to_dict() for m in self.history],
            "model_summary": self.model_summary,
            "used_synthetic_data": self.used_synthetic_data,
        }


# ============================================================================
# 损失
# ============================================================================
def build_criterion(rc: ResolvedConfig, *, device: torch.device) -> nn.Module:
    """构造损失函数。

    * ``focal_gamma > 0``：改用焦点损失（类别极不平衡时有用）；
    * ``class_weights = balanced``：按类别频率倒数加权（Kaggle 7 类不平衡时常用）；
    * 否则：普通交叉熵（含 label smoothing）。

    注意：类别权重需要先统计训练集分布，因此这里只处理"是否 balanced"，
    权重数值由 ``compute_class_weights`` 从数据算出来。
    """
    weights = None
    if rc.train.class_weights == "balanced":
        weights = compute_class_weights(rc, device=device)

    gamma = float(rc.train.focal_gamma)
    smoothing = float(rc.train.label_smoothing)
    if gamma > 0:
        return FocalLoss(gamma=gamma, weight=weights, label_smoothing=smoothing)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)


def compute_class_weights(rc: ResolvedConfig, *, device: torch.device) -> torch.Tensor:
    """按训练集帧数计算 ``balanced`` 类别权重。"""
    records = resolve_split_records(rc, split="train")
    num_classes = len(get_label_space(rc.label_space))
    counts = np.zeros(num_classes, dtype=np.float64)
    space = get_label_space(rc.label_space)
    for rec in records:
        counts[space.to_index(rec.label)] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    log.info("类别权重（balanced）：%s", np.round(weights, 3).tolist())
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """焦点损失（Lin et al. 2017）：抑制易分样本的主导作用。

    项目里很有用：六步动作的帧数往往极不均衡（第 1 步多、第 6 步少）。
    """

    def __init__(self, *, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), target: (N,)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = nn.functional.nll_loss(
            log_probs,
            target,
            weight=self.weight if self.weight.numel() > 0 else None,
            reduction="none",
        )
        pt = torch.exp(-nll).clamp(min=1e-6, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * nll
        return loss.mean()


# ============================================================================
# 评估一个 loader（训练中每轮都用它算 val 指标）
# ============================================================================
@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
    num_classes: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """跑完一个 loader，返回 ``(loss, accuracy, macro_f1, y_true, y_pred)``。

    逐帧统计（把 ``(B, T)`` 展平）—— 这样帧级与时序模型共用同一套指标代码。
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label_index"].to(device, non_blocking=True)
        logits = model.logits(images)  # (B, T, C)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)

        if criterion is not None:
            loss = criterion(flat_logits, flat_targets)
            total_loss += float(loss.detach()) * flat_targets.numel()
        total_count += flat_targets.numel()

        y_true_parts.append(flat_targets.detach().cpu().numpy())
        y_pred_parts.append(flat_logits.argmax(dim=-1).detach().cpu().numpy())

    if total_count == 0:
        raise TrainingError("评估 loader 为空，无法计算指标")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    mean_loss = total_loss / total_count if criterion is not None else float("nan")
    acc = float((y_true == y_pred).mean())
    f1 = macro_f1(y_true, y_pred, num_classes=num_classes)
    return mean_loss, acc, f1, y_true, y_pred


# ============================================================================
# 主训练入口
# ============================================================================
def train(rc: ResolvedConfig, *, max_epochs: int | None = None) -> TrainResult:
    """执行训练。

    Parameters
    ----------
    max_epochs:
        覆盖配置里的 epoch 数（冒烟测试用：``train(rc, max_epochs=2)``）。
    """
    seed = seed_everything(rc.runtime.seed, deterministic=rc.runtime.deterministic)
    device = resolve_device(rc.runtime.device)
    maybe_enable_tf32(device)

    run_dir = ensure_dir(rc.resolve_out_dir())
    log.info("实验目录：%s（config_hash=%s，seed=%d）", run_dir, rc.config_hash, seed)

    # --- 产物：配置 / 环境 / git ------------------------------------------
    dump_config(rc, run_dir / "resolved_config.yaml")
    write_json(
        run_dir / "env_info.json",
        {
            **describe_device(device),
            "config_hash": rc.config_hash,
            "seed": seed,
            "label_space": rc.label_space,
            "arch": rc.model.arch,
            "mode": rc.train.mode,
            "config_sources": list(rc.sources),
        },
    )
    write_json(run_dir / "git_info.json", _git_info())

    # --- 数据 -------------------------------------------------------------
    clip_mode = rc.train.mode in ("clip", "hybrid")
    train_records = resolve_split_records(rc, split="train")
    val_records = resolve_split_records(rc, split="val")
    used_synthetic = all(r.dataset == "synthetic" for r in train_records)

    train_loader = build_manifest_loader(rc, train_records, split="train", shuffle=True,
                                         for_clip_mode=clip_mode)
    val_loader = build_manifest_loader(rc, val_records, split="val", shuffle=False,
                                       for_clip_mode=clip_mode)

    write_json(
        run_dir / "manifest_summary.json",
        {
            "train_frames": len(train_records),
            "val_frames": len(val_records),
            "train_clips": len({r.clip_id for r in train_records}),
            "val_clips": len({r.clip_id for r in val_records}),
            "datasets": sorted({r.dataset for r in train_records + val_records}),
            "label_distribution_train": _label_distribution(train_records, rc),
        },
    )
    if used_synthetic:
        log.warning("本次训练使用合成数据：只能验证代码链路，指标不可写入报告")

    # --- 模型 / 优化器 ----------------------------------------------------
    from handwash.models.factory import build_model

    model = build_model(rc).to(device)
    num_classes = len(get_label_space(rc.label_space))
    criterion = build_criterion(rc, device=device)
    optimizer = _build_optimizer(rc, model)
    scheduler = _build_scheduler(rc, optimizer, max_epochs or rc.train.epochs)
    scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
        "cuda", enabled=(rc.train.precision == "fp16" and device.type == "cuda")
    )

    model_summary = model.describe() if hasattr(model, "describe") else {}
    write_json(run_dir / "model_summary.json", model_summary)
    log.info(
        "开始训练：epochs=%d，batch=%d，lr=%g，precision=%s，参数=%s",
        max_epochs or rc.train.epochs,
        rc.train.batch_size,
        rc.train.lr,
        rc.train.precision,
        f"{model_summary.get('params_total', 0):,}",
    )

    # --- 训练循环 ---------------------------------------------------------
    epochs = int(max_epochs or rc.train.epochs)
    history: list[EpochMetrics] = []
    best_metric = -1.0
    best_path: Path | None = None
    patience = rc.train.early_stopping_patience
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device=device, epoch=epoch, rc=rc
        )
        val_loss, val_acc, val_f1, _, _ = evaluate_loader(
            model, val_loader, device=device, criterion=criterion, num_classes=num_classes
        )
        val_metrics = EpochMetrics(epoch, "val", val_loss, val_acc, val_f1)
        history.extend([train_metrics, val_metrics])
        append_jsonl(run_dir / "run_log.jsonl", [train_metrics.to_dict(), val_metrics.to_dict()])
        if scheduler is not None:
            scheduler.step()

        log.info(
            "epoch %d/%d | train loss %.4f acc %.4f macroF1 %.4f | val loss %.4f acc %.4f macroF1 %.4f",
            epoch, epochs,
            train_metrics.loss, train_metrics.accuracy, train_metrics.macro_f1,
            val_loss, val_acc, val_f1,
        )

        # 选模标准：macro-F1（类别不平衡时比 accuracy 更能反映真实水平）
        if val_f1 > best_metric:
            best_metric = val_f1
            bad_epochs = 0
            best_path = checkpoint_path(run_dir, BEST_CHECKPOINT_NAME)
            save_checkpoint(
                best_path,
                model,
                arch=rc.model.arch,
                mode=rc.train.mode,
                num_classes=num_classes,
                label_space=rc.label_space,
                image_size=rc.model.image_size,
                normalize=rc.model.normalize,
                temporal=rc.model.temporal.to_dict(),
                metrics={"val_accuracy": val_acc, "val_macro_f1": val_f1, "val_loss": val_loss},
                config_hash=rc.config_hash,
                seed=seed,
                epoch=epoch,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            log.info("验证指标连续 %d 轮未提升，提前停止", patience)
            break

    last_path = checkpoint_path(run_dir, LAST_CHECKPOINT_NAME)
    save_checkpoint(
        last_path,
        model,
        arch=rc.model.arch,
        mode=rc.train.mode,
        num_classes=num_classes,
        label_space=rc.label_space,
        image_size=rc.model.image_size,
        normalize=rc.model.normalize,
        temporal=rc.model.temporal.to_dict(),
        config_hash=rc.config_hash,
        seed=seed,
        epoch=len([m for m in history if m.phase == "val"]),
    )

    result = TrainResult(
        run_dir=run_dir,
        best_checkpoint=best_path,
        best_metric=best_metric,
        epochs_run=len([m for m in history if m.phase == "val"]),
        history=history,
        model_summary=model_summary,
        used_synthetic_data=used_synthetic,
    )
    write_json(run_dir / "history.json", result.to_dict())
    log.info("训练完成：best val macroF1=%.4f，checkpoint=%s", best_metric, best_path)
    return result


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    *,
    device: torch.device,
    epoch: int,
    rc: ResolvedConfig,
) -> EpochMetrics:
    model.train()
    use_amp = rc.train.precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if rc.train.precision == "fp16" else torch.bfloat16

    running_loss = 0.0
    count = 0
    correct = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    accumulated = 0
    accum_steps = max(1, rc.train.accumulate_grad_batches)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label_index"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model.logits(images)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            loss = criterion(flat_logits, flat_targets) / accum_steps

        if use_amp and rc.train.precision == "fp16":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated += 1
        if accumulated >= accum_steps:
            if rc.train.grad_clip_norm > 0:
                if use_amp and rc.train.precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), rc.train.grad_clip_norm)
            if use_amp and rc.train.precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0

        running_loss += float(loss.detach()) * accum_steps * flat_targets.numel()
        count += flat_targets.numel()
        preds = flat_logits.argmax(dim=-1)
        correct += int((preds == flat_targets).sum())
        y_true_parts.append(flat_targets.detach().cpu().numpy())
        y_pred_parts.append(preds.detach().cpu().numpy())

    if accumulated > 0:  # 处理最后一个不完整累积步
        if rc.train.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc.train.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if count == 0:
        raise TrainingError("训练 loader 为空")

    num_classes = len(get_label_space(rc.label_space))
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return EpochMetrics(
        epoch=epoch,
        phase="train",
        loss=running_loss / count,
        accuracy=correct / count,
        macro_f1=macro_f1(y_true, y_pred, num_classes=num_classes),
    )


def _build_optimizer(rc: ResolvedConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise TrainingError("模型没有可训练参数（是不是把整个模型都冻结了？）")
    if rc.train.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=rc.train.lr, momentum=rc.train.momentum, weight_decay=rc.train.weight_decay, nesterov=True
        )
    return torch.optim.AdamW(params, lr=rc.train.lr, weight_decay=rc.train.weight_decay)


def _build_scheduler(rc: ResolvedConfig, optimizer: torch.optim.Optimizer, epochs: int):
    if rc.train.scheduler == "none":
        return None
    total = max(1, epochs)
    if rc.train.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total // 3), gamma=0.1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=rc.train.lr * 0.01)


def _label_distribution(records: Any, rc: ResolvedConfig) -> dict[str, int]:
    space = get_label_space(rc.label_space)
    counts: dict[str, int] = {}
    for rec in records:
        key = space.canonicalize(rec.label).value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _git_info() -> dict[str, Any]:
    """记录 git 状态：报告里要能说明"用的是哪份代码"。

    用 subprocess 直接问 git，比引入 GitPython 轻得多；没有 git 时静默降级。
    """
    import subprocess

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run(["rev-parse", "HEAD"])
    return {
        "commit": commit,
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
        "available": commit is not None,
    }
