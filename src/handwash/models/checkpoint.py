"""Checkpoint 存取：一次保存，必须能原地恢复。

checkpoint 里到底存什么（务必读懂，否则会出现"换了台机器结果全错"）
--------------------------------------------------------------------
::

    {
      "format": "handwash-checkpoint",
      "format_version": 1,
      "handwash_version": "0.1.0",
      "arch": "yolo26n-cls",              # 模型架构名
      "mode": "frame",                    # frame | clip | hybrid
      "num_classes": 6,
      "label_space": "kaggle",            # 类别顺序的权威来源
      "temporal": {...},                  # 时序头配置（仅 clip/hybrid）
      "image_size": 224,
      "normalize": "zero_one",
      "state": {...},                     # 模型参数
      "metrics": {...},                   # 该 checkpoint 的验证指标（写报告要用）
      "config_hash": "a1b2c3d4e5f6a7b8",  # 训练时的完整配置指纹
      "seed": 42,
      "epoch": 7,
      "created_at": "2026-03-01T12:00:00"
    }

``label_space`` 必须随 checkpoint 走：只有它才能保证"下标 -> 步骤"的映射不错位。
加载时会校验 ``num_classes`` 与当前配置一致，不一致直接报错。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from handwash.errors import CheckpointError
from handwash.logging import get_logger
from handwash.paths import ensure_dir

__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "save_checkpoint", "load_checkpoint", "model_state_for_saving"]

log = get_logger(__name__)

FORMAT_NAME = "handwash-checkpoint"
FORMAT_VERSION = 1


def model_state_for_saving(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """决定哪些参数进 checkpoint。

    * ``YoloClassifier``：只存 ``head.*``（骨干由 ultralytics 官方权重提供，
      全存会让文件从 3 MB 涨到 300 MB，且跨版本加载易碎）。
    * 其它模型：全部参数。
    """
    if hasattr(model, "trainable_state_dict"):
        state = model.trainable_state_dict()
        log.info("checkpoint 只保存 %d 个自有参数（骨干权重来自官方文件）", len(state))
        return state
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    arch: str,
    mode: str,
    num_classes: int,
    label_space: str,
    image_size: int,
    normalize: str,
    temporal: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    config_hash: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """保存 checkpoint（原子写：先写 .tmp 再替换）。"""
    target = Path(path)
    ensure_dir(target.parent)
    payload: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "arch": arch,
        "mode": mode,
        "num_classes": int(num_classes),
        "label_space": label_space,
        "image_size": int(image_size),
        "normalize": normalize,
        "temporal": dict(temporal or {}),
        "state": model_state_for_saving(model),
        "metrics": dict(metrics or {}),
        "config_hash": config_hash,
        "seed": seed,
        "epoch": epoch,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        payload["extra"] = dict(extra)

    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    log.info("checkpoint 已保存：%s（epoch=%s）", target, epoch)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expect_num_classes: int | None = None,
    expect_label_space: str | None = None,
) -> dict[str, Any]:
    """读取 checkpoint 并做一致性校验。

    Raises
    ------
    CheckpointError
        文件缺失、格式不对、类别数或标签空间与当前配置不一致。
    """
    target = Path(path)
    if not target.exists():
        raise CheckpointError(
            f"checkpoint 不存在：{target}",
            hint="确认路径；训练产物默认在 outputs/<run_name>/models/ 下。",
        )
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - torch.load 报错类型不稳定
        raise CheckpointError(f"checkpoint 读取失败：{target}（{exc}）") from exc

    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise CheckpointError(
            f"不是本项目的 checkpoint 格式：{target}",
            hint=f"期望 format={FORMAT_NAME!r}；若是 ultralytics 官方 .pt，请放到 models/ 下作为权重使用。",
        )
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint 版本不兼容：{payload.get('format_version')}（本代码支持 {FORMAT_VERSION}）",
            hint="请重新训练，或从 Git 历史里检出旧版本代码来加载。",
        )

    if expect_num_classes is not None and int(payload["num_classes"]) != expect_num_classes:
        raise CheckpointError(
            f"checkpoint 类别数 {payload['num_classes']} 与当前标签空间 {expect_num_classes} 不一致",
            hint="标签空间变了就必须重新训练：旧权重的通道顺序已经没有意义。",
        )
    if expect_label_space is not None and str(payload["label_space"]) != expect_label_space:
        raise CheckpointError(
            f"checkpoint 训练于标签空间 `{payload['label_space']}`，当前是 `{expect_label_space}`",
            hint="跨数据集评估请显式做标签映射（core.labels.LabelSpace.to_space），不要直接加载。",
        )
    log.info(
        "checkpoint 已加载：%s（arch=%s，epoch=%s，config_hash=%s）",
        target,
        payload.get("arch"),
        payload.get("epoch"),
        payload.get("config_hash"),
    )
    return payload
