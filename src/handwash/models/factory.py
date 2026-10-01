"""模型工厂：把配置翻译成模型实例。

两条路径，必须区分清楚（CONTRIBUTING.md R18）
----------------------------------------------
1. **单帧模型**（``train.mode=frame``）：``build_model(rc)`` 只造骨干 + 分类头。
2. **时序模型**（``train.mode=clip`` / ``hybrid``）：``build_model(rc)`` 返回
   ``TemporalClassifier``，其内部骨干由 ``model.arch`` 决定，时序头由
   ``model.temporal.kind`` 决定。

``num_classes`` **永远**由标签空间推导，禁止在配置里手写数字 ——
这是"改了标签空间却忘了改模型"这类事故的唯一防线。
"""

from __future__ import annotations

from typing import Any

from handwash.core.config import ModelConfig, ResolvedConfig
from handwash.core.labels import get_label_space
from handwash.core.registry import MODELS
from handwash.errors import ConfigError, ModelError
from handwash.logging import get_logger

__all__ = ["build_model", "build_backbone", "available_model_archs", "register_all", "resolve_num_classes"]

log = get_logger(__name__)

_REGISTERED = False


def register_all() -> list[str]:
    """导入全部模型实现，触发注册。返回已注册的名字列表。

    必须在这里显式 import：注册依赖 import 的副作用，
    而 ``handwash.models`` 采用了惰性导入（见该模块 __getattr__）。
    """
    global _REGISTERED
    if not _REGISTERED:
        from handwash.models import frame_cnn, temporal, yolo26_cls  # noqa: F401

        _REGISTERED = True
    return MODELS.names()


def available_model_archs() -> list[str]:
    return register_all()


def resolve_num_classes(cfg: ModelConfig, *, label_space: str) -> int:
    """类别数：显式配置优先，否则由标签空间推导，并做一致性校验。"""
    space = get_label_space(label_space)
    derived = len(space)
    if cfg.num_classes is None:
        return derived
    if cfg.num_classes != derived:
        raise ConfigError(
            f"model.num_classes={cfg.num_classes} 与标签空间 `{label_space}` 的 {derived} 类不一致",
            hint="把 model.num_classes 设为 null（或不写），让代码自动推导。",
        )
    return cfg.num_classes


def _normalize_pretrained(cfg: ModelConfig) -> bool | str:
    """``pretrained`` 支持 auto / true / false / 本地路径 / 权重名。"""
    value = cfg.pretrained
    if value == "auto":
        # YOLO 系列默认用官方权重（架构要求的），CNN 基线默认用 ImageNet 预训练
        return True
    return value


def build_backbone(
    arch: str,
    *,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool | str = True,
    **kwargs: Any,
):
    """只造骨干（供 ``TemporalClassifier`` 与两级训练复用）。"""
    register_all()
    key = str(arch).strip().lower()
    if key in ("temporal", "gru", "tcn"):
        raise ConfigError(
            f"`{arch}` 是时序组合模型，不能作为骨干使用",
            hint="骨干请用 yolo26n-cls / mobilenet_v2 / resnet18 之一。",
        )
    try:
        cls = MODELS.get(key)
    except ModelError as exc:
        raise ConfigError(
            f"未知的模型架构：{arch!r}",
            hint=f"已注册：{MODELS.names()}；新增模型见 docs/ARCHITECTURE.md 的扩展指南。",
        ) from exc

    # 防止调用方把 dropout/pretrained/num_classes 重复传两遍（会触发 TypeError，
    # 报错信息很难懂）。显式参数优先。
    extra = {k: v for k, v in kwargs.items() if k not in ("dropout", "pretrained", "num_classes", "backbone")}

    if key in ("mobilenet_v2", "resnet18", "efficientnet_b0"):
        return cls(num_classes, backbone=key, pretrained=bool(pretrained), dropout=dropout, **extra)
    return cls(num_classes, pretrained=pretrained, dropout=dropout, **extra)


def build_model(rc: ResolvedConfig):
    """按 ResolvedConfig 造模型。

    返回对象一定满足 ``models.base.BaseClassifier`` 的契约：
    ``logits(x) -> (B, T, C)``，``embed(x) -> (B, T, D)``，``num_classes``，``feature_dim``。
    """
    cfg = rc.model
    num_classes = resolve_num_classes(cfg, label_space=rc.label_space)
    pretrained = _normalize_pretrained(cfg)

    if rc.train.mode == "frame":
        model = build_backbone(
            cfg.arch,
            num_classes=num_classes,
            dropout=cfg.dropout,
            pretrained=pretrained,
        )
    else:
        from handwash.models.temporal import TemporalClassifier

        backbone_kwargs: dict[str, Any] = {"dropout": cfg.dropout, "pretrained": pretrained}
        model = TemporalClassifier(
            num_classes,
            backbone_arch=cfg.arch,
            temporal_cfg=cfg.temporal,
            backbone_kwargs=backbone_kwargs,
            dropout=cfg.dropout,
            backbone_state=rc.weight_path,
        )

    log.info("模型已构建：%s（%s），类别数=%d", model.arch_name, rc.train.mode, num_classes)
    return model
