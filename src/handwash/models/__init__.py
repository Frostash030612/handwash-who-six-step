"""模型层：只负责"图像/特征 -> logits"，不负责训练循环与指标。

分层位置：L3。允许 import torch / torchvision / ultralytics。

核心接口（CONTRIBUTING.md R18）
--------------------------------
所有模型（无论是 YOLO 适配器还是自建 CNN，无论单帧还是时序）都必须实现::

    model.logits(x) -> torch.Tensor

输出形状统一为 ``(B, T, C)``：
    * 单帧模型：``T == 1``，即 ``(B, 1, C)``；
    * 时序模型：``T`` 为窗口长度，逐帧输出分类 logits。

统一成 ``(B, T, C)`` 的好处：
    训练损失（逐帧交叉熵）、评估（展平算指标）、推理（平滑/分段）
    三处代码完全共用，不需要为"帧模型/序列模型"各写一套分支。
    这是本项目最容易失控的地方，因此把它固定成契约。

另外三个必须实现的方法：
    ``embed(x) -> (B, T, D)``   时序头的输入特征（用于两级训练）
    ``num_classes -> int``
    ``feature_dim -> int``

具体实现文件的注册名（``ModelConfig.arch`` 的合法取值）：
    yolo26n-cls      主模型（见 yolo26_cls.py）
    yolon-cls        通用 YOLO 分类适配器（可指定 yolo11n-cls / yolov8n-cls）
    mobilenet_v2     基线（见 frame_cnn.py；论文 Baseline 之一）
    resnet18         基线
    efficientnet_b0  扩展基线
"""

from __future__ import annotations

__all__ = [
    "BaseClassifier",
    "TemporalClassifier",
    "build_model",
    "available_model_archs",
    "load_checkpoint",
    "save_checkpoint",
    "register_all",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BaseClassifier": ("handwash.models.base", "BaseClassifier"),
    "TemporalClassifier": ("handwash.models.temporal", "TemporalClassifier"),
    "build_model": ("handwash.models.factory", "build_model"),
    "available_model_archs": ("handwash.models.factory", "available_model_archs"),
    "load_checkpoint": ("handwash.models.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("handwash.models.checkpoint", "save_checkpoint"),
    "register_all": ("handwash.models.factory", "register_all"),
}


def __getattr__(name: str):
    """惰性转发：避免 `import handwash.models` 时就强制拉起 torch / ultralytics。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0])
    return getattr(module, target[1])
