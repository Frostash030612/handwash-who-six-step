"""额外的 torchvision 分类骨干，供配置文档中列出的基线使用。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from handwash.core.registry import register_model
from handwash.errors import BackendUnavailableError, ModelError
from handwash.models.base import BaseClassifier


def _pretrained_weights(pretrained: bool | str, weights_type):
    enabled = pretrained is True or str(pretrained).lower() in ("auto", "true", "imagenet")
    return weights_type.DEFAULT if enabled else None


@register_model("resnet18")
class ResNet18Classifier(BaseClassifier):
    def __init__(self, *, num_classes: int, temporal: Any, dropout: float = 0.2,
                 pretrained: bool | str = "auto", **_: Any) -> None:
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            backbone = resnet18(weights=_pretrained_weights(pretrained, ResNet18_Weights))
        except Exception as exc:
            raise BackendUnavailableError(
                "torchvision", "请安装与 PyTorch 匹配的 torchvision，并检查预训练权重是否可用。"
            ) from exc
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        super().__init__(num_classes=num_classes, feature_dim=feature_dim, temporal=temporal,
                         dropout=dropout, arch_name="resnet18")
        self.backbone = backbone

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


@register_model("efficientnet_b0")
class EfficientNetB0Classifier(BaseClassifier):
    def __init__(self, *, num_classes: int, temporal: Any, dropout: float = 0.2,
                 pretrained: bool | str = "auto", **_: Any) -> None:
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            backbone = efficientnet_b0(weights=_pretrained_weights(pretrained, EfficientNet_B0_Weights))
        except Exception as exc:
            raise BackendUnavailableError(
                "torchvision", "请安装与 PyTorch 匹配的 torchvision，并检查预训练权重是否可用。"
            ) from exc
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier = nn.Identity()
        super().__init__(num_classes=num_classes, feature_dim=feature_dim, temporal=temporal,
                         dropout=dropout, arch_name="efficientnet_b0")
        self.backbone = backbone

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)
