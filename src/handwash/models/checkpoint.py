"""项目原生 checkpoint 的原子保存与兼容性检查。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from handwash.errors import CheckpointError
from handwash.paths import ensure_dir


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
    temporal: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    config_hash: str,
    seed: int,
    epoch: int,
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    payload = {
        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "arch": arch,
            "mode": mode,
            "num_classes": int(num_classes),
            "label_space": label_space,
            "image_size": int(image_size),
            "normalize": normalize,
            "temporal": dict(temporal),
            "metrics": dict(metrics or {}),
            "config_hash": config_hash,
            "seed": int(seed),
            "epoch": int(epoch),
        },
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expect_num_classes: int | None = None,
    expect_label_space: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise CheckpointError(f"找不到模型 checkpoint：{target}")
    try:
        try:
            payload = torch.load(target, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch 2.2 compatibility
            payload = torch.load(target, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(f"无法读取 checkpoint：{target}", hint=str(exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise CheckpointError(f"checkpoint 格式无效：{target}", hint="需要包含 state 与 metadata 的项目格式")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointError(f"checkpoint 缺少 metadata：{target}")
    if expect_num_classes is not None and int(metadata.get("num_classes", -1)) != expect_num_classes:
        raise CheckpointError(
            f"checkpoint 类别数不匹配：期望 {expect_num_classes}，实际 {metadata.get('num_classes')}"
        )
    if expect_label_space is not None and metadata.get("label_space") != expect_label_space:
        raise CheckpointError(
            f"checkpoint 标签空间不匹配：期望 {expect_label_space!r}，实际 {metadata.get('label_space')!r}"
        )
    return {**payload, **metadata}
