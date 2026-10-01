"""可选深度学习模型实现。

导入本包只注册轻量 CNN。Ultralytics 仅在构造 YOLO 模型时延迟导入，
因此 MobileNet 冒烟流程不要求安装 Ultralytics。
"""

from handwash.models.factory import build_model, register_all

register_all()

__all__ = ["build_model", "register_all"]
