"""按配置构造并注册分类模型。"""

from __future__ import annotations

from handwash.core.labels import get_label_space
from handwash.core.registry import available_models, get_model_cls
from handwash.errors import ModelError, ModelNotFoundError


def register_all() -> None:
    # import 用于触发装饰器注册；YOLO 的第三方依赖仍在模型初始化时延迟加载。
    from handwash.models import mobilenet as _mobilenet  # noqa: F401
    from handwash.models import torchvision_baselines as _torchvision_baselines  # noqa: F401
    from handwash.models import yolo26_cls as _yolo26_cls  # noqa: F401


def build_model(rc, *, pretrained: bool | str | None = None):
    register_all()
    arch = str(rc.model.arch).strip().lower()
    try:
        model_cls = get_model_cls(arch)
    except ModelNotFoundError:
        raise ModelNotFoundError(arch, available=available_models()) from None
    use_config_pretrained = pretrained is None
    pretrained = rc.model.pretrained if use_config_pretrained else pretrained
    if use_config_pretrained and rc.weight_path is not None:
        if model_cls.__module__ != "handwash.models.yolo26_cls":
            raise ModelError(
                f"{arch} 不支持从本地 checkpoint 初始化",
                hint="当前 torchvision 基线只支持 auto / imagenet / false；本地 .pt 权重只接入 Ultralytics 分类模型。",
            )
        pretrained = str(rc.weight_path)
    options = {
        "num_classes": len(get_label_space(rc.label_space)),
        "temporal": rc.model.temporal,
        "dropout": rc.model.dropout,
        "pretrained": pretrained,
        "arch_name": arch,
    }
    return model_cls(**options)
