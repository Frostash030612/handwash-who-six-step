"""MobileNetV2 图像分类器。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from handwash.core.registry import register_model
from handwash.errors import BackendUnavailableError, ModelError
from handwash.models.base import BaseClassifier


@register_model("mobilenet_v2")
class MobileNetV2Classifier(BaseClassifier):
    def __init__(
        self,
        *,
        num_classes: int,
        temporal: Any,
        dropout: float = 0.2,
        pretrained: bool | str = "auto",
        **_: Any,
    ) -> None:
        try:
            from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        except Exception as exc:  # torchvision may fail on a mismatched torch build
            raise BackendUnavailableError(
                "torchvision",
                "请按 environment.yml 安装与 PyTorch 匹配的 torchvision。",
            ) from exc

        use_pretrained = pretrained is True or str(pretrained).lower() in ("auto", "true", "imagenet")
        try:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if use_pretrained else None)
        except Exception as exc:
            raise ModelError(
                "无法初始化 MobileNetV2",
                hint="预训练权重需可下载；离线冒烟训练请设置 model.pretrained=false。",
            ) from exc
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier = nn.Identity()

        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            temporal=temporal,
            dropout=dropout,
            arch_name="mobilenet_v2",
        )
        self.backbone = backbone

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
