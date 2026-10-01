"""Ultralytics YOLO 分类模型适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from handwash.core.registry import register_model
from handwash.errors import BackendUnavailableError, ModelError
from handwash.models.base import BaseClassifier


def _ultralytics_model(weights: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise BackendUnavailableError("ultralytics", "请安装与权重兼容的 ultralytics 版本。") from exc
    try:
        return YOLO(weights).model
    except Exception as exc:
        raise ModelError(f"无法加载 Ultralytics 模型：{weights}", hint=str(exc)) from exc


def probe_yolo_availability(weights: str = "yolo26n-cls.pt") -> dict[str, Any]:
    """doctor 的只读探测：只加载显式本地权重，不触发网络下载。"""
    try:
        import ultralytics
    except Exception as exc:
        return {
            "ultralytics": False,
            "loadable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "安装 ultralytics 后重试。",
        }
    candidate = Path(weights).expanduser()
    if not candidate.is_file():
        return {
            "ultralytics": True,
            "loadable": False,
            "version": ultralytics.__version__,
            "hint": f"权重不在本地：{weights}；doctor 不会自动下载，请训练时再按需准备预训练权重。",
        }
    try:
        _ultralytics_model(str(candidate))
        return {"ultralytics": True, "loadable": True, "version": ultralytics.__version__}
    except Exception as exc:
        return {
            "ultralytics": True,
            "loadable": False,
            "version": ultralytics.__version__,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "核对模型名称、Ultralytics 版本及网络访问。",
        }


@register_model("yolo26n-cls")
@register_model("yolo26m-cls")
@register_model("yolon-cls")
@register_model("yolov8n-cls")
class YoloClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        num_classes: int,
        temporal: Any,
        arch_name: str = "yolo26n-cls",
        dropout: float = 0.2,
        pretrained: bool | str = "auto",
        **_: Any,
    ) -> None:
        arch = str(arch_name).lower()
        if arch == "yolon-cls":
            arch = "yolo11n-cls"
        use_pretrained = pretrained is not False and str(pretrained).lower() != "false"
        if isinstance(pretrained, str) and pretrained.lower() not in ("auto", "true", "false", "imagenet"):
            weights = pretrained
        else:
            weights = f"{arch}.pt" if use_pretrained else f"{arch}.yaml"

        model = _ultralytics_model(weights)
        layers = getattr(model, "model", None)
        if layers is None or len(layers) < 2:
            raise ModelError(f"Ultralytics 分类模型结构不符合预期：{weights}")
        head = layers[-1]
        linear = getattr(head, "linear", None)
        if not isinstance(linear, nn.Linear):
            raise ModelError(
                f"Ultralytics 分类头缺少 Linear 层：{weights}",
                hint="该权重或 Ultralytics 版本与当前适配器不兼容。",
            )
        feature_dim = int(linear.in_features)
        head.linear = nn.Identity()

        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            temporal=temporal,
            dropout=dropout,
            arch_name=arch,
        )
        self.network = model

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        layers = self.network.model
        features = layers[0](images)
        for layer in list(layers)[1:-1]:
            features = layer(features)
        head = layers[-1]
        features = head.conv(features)
        return head.pool(features).flatten(1)
