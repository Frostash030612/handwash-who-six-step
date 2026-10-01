"""基线 CNN：MobileNetV2 / ResNet18 / EfficientNet-B0（torchvision 实现）。

为什么要有这个文件
------------------
选题文档把 MobileNetV2 列为基线模型，开源基线（github.com/edi-riga/handwash）
也是 MobileNetV2 + GRU。因此必须能一键切换：

    handwash train model.arch=mobilenet_v2
    handwash train model.arch=yolo26n-cls

**注意**：torchvision 自带预训练权重需要联网下载一次（缓存到 ~/.cache/torch）。
离线环境请把 ``model.pretrained`` 设为 false 或指向本地 .pth 文件。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from handwash.core.registry import register_model
from handwash.errors import ModelError
from handwash.logging import get_logger
from handwash.models.base import BaseClassifier

__all__ = ["FrameCNN", "BACKBONE_SPECS"]

log = get_logger(__name__)


#: name -> (torchvision 构造函数名, 预训练权重枚举名, 特征维度, 需要替换的分类头名)
BACKBONE_SPECS: dict[str, tuple[str, str, int, str]] = {
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights", 1280, "classifier"),
    "resnet18": ("resnet18", "ResNet18_Weights", 512, "fc"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", 1280, "classifier"),
}


def _build_torchvision_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, str]:
    try:
        import torchvision.models as tvm
    except ImportError as exc:  # pragma: no cover - 环境缺 torchvision
        raise ModelError(
            "torchvision 不可用，无法构建 CNN 基线",
            hint="conda install -c pytorch torchvision，或改用 model.arch=yolo26n-cls。",
        ) from exc

    if name not in BACKBONE_SPECS:
        raise ModelError(f"未知的 CNN 骨干：{name!r}", hint=f"可用：{sorted(BACKBONE_SPECS)}")
    ctor_name, weights_enum_name, feat_dim, head_name = BACKBONE_SPECS[name]
    ctor = getattr(tvm, ctor_name)

    weights = None
    if pretrained:
        weights = getattr(tvm, weights_enum_name).DEFAULT
    model = ctor(weights=weights)

    # 把原分类头换成 Identity，forward 直接拿池化后的特征
    if head_name == "classifier":
        model.classifier = nn.Identity()
    elif head_name == "fc":
        model.fc = nn.Identity()
    else:  # pragma: no cover - 目前不会走到
        raise ModelError(f"未支持的分类头名称：{head_name}")
    return model, feat_dim, head_name


@register_model("mobilenet_v2")
@register_model("resnet18")
@register_model("efficientnet_b0")
class FrameCNN(BaseClassifier):
    """单帧分类基线：骨干 + 分类头，输出 ``(B, 1, C)``。

    通过 ``__init__`` 的 ``backbone`` 参数区分具体骨干，
    因此三个注册名共用同一个类（避免复制粘贴三份几乎相同的代码）。
    """

    arch_name = "frame_cnn"

    def __init__(
        self,
        num_classes: int,
        *,
        backbone: str = "mobilenet_v2",
        pretrained: bool = True,
        dropout: float = 0.2,
        **_: Any,
    ) -> None:
        super().__init__(num_classes, dropout=dropout)
        self.backbone_name = backbone
        self.arch_name = f"{backbone}"
        self.backbone, feat_dim, _ = _build_torchvision_backbone(backbone, pretrained)
        self._feature_dim = int(feat_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self._feature_dim, num_classes),
        )
        log.info("构建骨干 %s：特征维度 %d，类别数 %d", backbone, self._feature_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """输入 ``(B, 3, H, W)`` 或 ``(B, T, 3, H, W)``，输出 ``(B, T, D)``。

        单帧模型天然没有时间维，但**必须能接受带时间维的输入**：
        Dataloader 统一产出 (B, T, 3, H, W)（帧模式下 T=1），
        模型这边负责把时间维折叠进 batch，而不是让 pipeline 分两套代码。
        """
        if x.ndim == 5:
            batch, steps = x.shape[0], x.shape[1]
            flat = x.reshape(batch * steps, *x.shape[2:])
        elif x.ndim == 4:
            batch, steps, flat = x.shape[0], 1, x
        else:
            raise ModelError(
                f"单帧模型输入应为 (B, 3, H, W) 或 (B, T, 3, H, W)，实际 {tuple(x.shape)}"
            )

        feats = self.backbone(flat)
        if feats.ndim > 2:  # 某些骨干返回 (B, C, 1, 1)
            feats = torch.flatten(feats, start_dim=1)
        return feats.reshape(batch, steps, self._feature_dim)

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def backbone_modules(self) -> list[nn.Module]:
        return [self.backbone]
