"""通用注册表：把"名字 -> 实现"的绑定从代码里解耦。

存在的理由（CONTRIBUTING.md R12）：
    契约层 core 不允许 import torch / ultralytics / cv2。但 pipelines 又需要
    "按配置里的字符串拿模型/数据集"。注册表就是这个解耦点：
        models/__init__.py 负责 import 具体实现触发注册（L3 层，可以碰 torch），
        pipelines 只跟 registry 打交道（L1 接口）。

用法::

    @register_model("yolo26n-cls")
    class Yolo26Classifier: ...

    cls = get_model_cls("yolo26n-cls")
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Final, TypeVar

from handwash.errors import ConfigError, ModelNotFoundError

__all__ = [
    "Registry",
    "register_model",
    "register_dataset",
    "register_transform",
    "get_model_cls",
    "get_dataset_cls",
    "get_transform_cls",
    "available_models",
    "available_datasets",
    "available_transforms",
]

T = TypeVar("T")


class Registry:
    """一个极简的名字 -> 类 注册表。

    与 ``entry_points`` 相比：零打包开销、组员一眼看得懂、报错信息可控。
    """

    __slots__ = ("kind", "_items")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str, *, override: bool = False, **metadata: object) -> Callable[[type[T]], type[T]]:
        """装饰器：注册一个类。重名默认报错（防止两人各写一份同名实现）。"""
        # 先做类型检查，再 strip 后判空：否则 "   " 会通过"非空"检查，
        # 最后以空字符串为键注册，成为一个永远取不到的幽灵条目。
        if not isinstance(name, str):
            raise ConfigError(f"{self.kind} 的注册名必须是字符串，实际 {type(name).__name__}={name!r}")
        key = name.strip().lower()
        if not key:
            raise ConfigError(f"{self.kind} 的注册名必须是非空字符串，实际 {name!r}")

        def decorator(cls: type[T]) -> type[T]:
            if key in self._items and not override:
                existing = self._items[key]
                raise ConfigError(
                    f"{self.kind} 名称冲突：`{name}` 已被 {existing.__module__}.{existing.__qualname__} 注册",
                    hint="换一个名字，或确认是否重复实现；确需覆盖请显式传 override=True 并说明理由。",
                )
            if metadata:
                setattr(cls, "__registry_metadata__", {**getattr(cls, "__registry_metadata__", {}), **metadata})
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        key = str(name).strip().lower()
        if key not in self._items:
            raise ModelNotFoundError(name, available=sorted(self._items))
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return str(name).strip().lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)


MODELS: Final[Registry] = Registry("模型")
DATASETS: Final[Registry] = Registry("数据集")
TRANSFORMS: Final[Registry] = Registry("预处理变换")

register_model = MODELS.register
register_dataset = DATASETS.register
register_transform = TRANSFORMS.register

get_model_cls = MODELS.get
get_dataset_cls = DATASETS.get
get_transform_cls = TRANSFORMS.get

available_models = MODELS.names
available_datasets = DATASETS.names
available_transforms = TRANSFORMS.names
