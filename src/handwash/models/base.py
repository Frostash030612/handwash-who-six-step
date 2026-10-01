"""模型基类：统一 ``(B, T, C)`` 输出契约。

新模型只需要继承 ``BaseClassifier`` 并实现 ``_forward_logits``，
其余（形状校验、冻结/解冻、参数量统计、设备迁移）由基类提供。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from handwash.errors import ModelError
from handwash.logging import get_logger

__all__ = ["BaseClassifier", "check_logits_shape"]

log = get_logger(__name__)


def check_logits_shape(logits: torch.Tensor, num_classes: int, *, where: str) -> torch.Tensor:
    """强制校验输出形状为 ``(B, T, C)``。

    宁可在这里报错，也不要在 loss 里因为广播规则悄悄算出错误结果。
    """
    if logits.ndim != 3:
        raise ModelError(
            f"{where} 输出的 logits 必须是 (B, T, C) 三维张量，实际 shape={tuple(logits.shape)}",
            hint="单帧模型请先 unsqueeze(1)，即 (B, C) -> (B, 1, C)。",
        )
    if logits.shape[-1] != num_classes:
        raise ModelError(
            f"{where} 输出的类别数 {logits.shape[-1]} 与标签空间 {num_classes} 不一致",
            hint="检查 model.num_classes 是否由标签空间推导（不要手写数字）。",
        )
    return logits


class BaseClassifier(nn.Module, ABC):
    """所有帧级 / 序列级分类器的基类。"""

    #: 子类应覆盖，用于日志与报告
    arch_name: str = "base"

    def __init__(self, num_classes: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes < 2:
            raise ModelError(f"num_classes 至少为 2，实际 {num_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ModelError(f"dropout 必须在 [0, 1)，实际 {dropout}")
        self._num_classes = int(num_classes)
        self.dropout_p = float(dropout)

    # --- 契约 -------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """``embed`` 输出的特征维度 D。时序头需要它来构造。"""

    @abstractmethod
    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """子类实现：返回 ``(B, T, C)``。"""

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """统一入口：带形状校验的 logits。"""
        out = self._forward_logits(x)
        return check_logits_shape(out, self._num_classes, where=f"{self.arch_name}.logits")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``(B, T, D)`` 特征。默认实现：复用 ``logits`` 之前的特征不可得，
        因此子类**应当**覆写本方法；未覆写时抛错而不是悄悄返回 logits。
        """
        raise ModelError(
            f"{self.arch_name} 未实现 embed()，无法作为时序模型的骨干",
            hint="如需两级训练（帧模型 + GRU/TCN），请在该模型里实现 embed()。",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` 直接返回 logits —— 保证 ``torch.compile`` / 导出友好。"""
        return self.logits(x)

    # --- 训练期工具 -------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """softmax 概率，形状 ``(B, T, C)``。"""
        return torch.softmax(self.logits(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """argmax 类别下标，形状 ``(B, T)``。"""
        return self.logits(x).argmax(dim=-1)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """冻结/解冻骨干（骨干由子类通过 ``backbone_modules()`` 声明）。"""
        for module in self.backbone_modules():
            for param in module.parameters():
                param.requires_grad = not freeze
        log.info("%s 骨干已%s", self.arch_name, "冻结" if freeze else "解冻")

    def backbone_modules(self) -> list[nn.Module]:
        """默认把除分类头之外的全部子模块视为骨干；子类可覆写精确指定。"""
        head_names = {"head", "classifier", "fc"}
        return [m for name, m in self.named_children() if name not in head_names]

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            return int(sum(p.numel() for p in params if p.requires_grad))
        return int(sum(p.numel() for p in params))

    def describe(self) -> dict[str, Any]:
        """写进实验日志的模型摘要（报告里要给出参数量）。"""
        return {
            "arch": self.arch_name,
            "num_classes": self._num_classes,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout_p,
            "params_total": self.num_parameters(),
            "params_trainable": self.num_parameters(trainable_only=True),
        }
