"""图像分类骨干与逐帧时序头的统一接口。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CausalResidualBlock(nn.Module):
    """只读取当前和过去特征的一维时序卷积块。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        # LayerNorm is applied per time step below; BatchNorm1d would aggregate
        # across the temporal axis and leak future-frame statistics into a causal head.
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(nn.functional.pad(x, (self.padding, 0)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(self.activation(y))


class BaseClassifier(nn.Module):
    """接受 ``(B,C,H,W)`` 或 ``(B,T,C,H,W)``，统一输出 ``(B,T,K)``。"""

    def __init__(
        self,
        *,
        num_classes: int,
        feature_dim: int,
        temporal: Any,
        dropout: float = 0.2,
        arch_name: str = "classifier",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.arch_name = arch_name
        self.temporal_kind = str(temporal.kind)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.temporal_projection: nn.Module | None = None
        self.temporal_head: nn.Module | None = None

        if self.temporal_kind == "mean_pool":
            self.temporal_projection = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif self.temporal_kind == "gru":
            self.temporal_head = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=int(temporal.hidden_size),
                num_layers=int(temporal.num_layers),
                batch_first=True,
                bidirectional=bool(temporal.bidirectional),
                dropout=float(temporal.dropout) if int(temporal.num_layers) > 1 else 0.0,
            )
            out_dim = int(temporal.hidden_size) * (2 if temporal.bidirectional else 1)
            self.temporal_classifier = nn.Linear(out_dim, self.num_classes)
        elif self.temporal_kind == "tcn":
            dilations = tuple(int(d) for d in temporal.dilations) or (1,)
            self.temporal_head = nn.Sequential(
                *[
                    CausalResidualBlock(
                        self.feature_dim,
                        int(temporal.kernel_size),
                        dilation,
                        float(temporal.dropout),
                    )
                    for dilation in dilations
                ]
            )
            self.temporal_classifier = nn.Conv1d(self.feature_dim, self.num_classes, kernel_size=1)

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """子类实现：输入 ``(N,3,H,W)``，返回 ``(N,D)`` 特征。"""
        raise NotImplementedError

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """返回 ``(B,T,D)`` 的逐帧视觉特征。"""
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"输入应为 (B,C,H,W) 或 (B,T,C,H,W)，实际 {tuple(images.shape)}")
        batch, steps, channels, height, width = images.shape
        features = self._encode_frames(images.reshape(batch * steps, channels, height, width))
        return features.reshape(batch, steps, -1)

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """计算逐时间点 logits，输出形状固定为 ``(B,T,K)``。"""
        features = self.embed(images)
        if self.temporal_kind == "none":
            output = features
        elif self.temporal_kind == "mean_pool":
            counts = torch.arange(1, features.shape[1] + 1, device=features.device, dtype=features.dtype)
            context = features.cumsum(dim=1) / counts.view(1, -1, 1)
            output = self.temporal_projection(torch.cat((features, context), dim=-1))
        elif self.temporal_kind == "gru":
            output, _ = self.temporal_head(features)
        elif self.temporal_kind == "tcn":
            output = self.temporal_head(features.transpose(1, 2)).transpose(1, 2)
        else:  # 配置层会提前校验；此处防止绕过配置工厂。
            raise ValueError(f"不支持的时序类型：{self.temporal_kind}")

        if self.temporal_kind in ("gru", "tcn"):
            if self.temporal_kind == "tcn":
                return self.temporal_classifier(self.dropout(output).transpose(1, 2)).transpose(1, 2)
            return self.temporal_classifier(self.dropout(output))
        return self.classifier(self.dropout(output))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits(images)

    def describe(self) -> dict[str, int | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "arch": self.arch_name,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "params_total": total,
            "params_trainable": trainable,
        }
