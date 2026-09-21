"""注册表契约测试（src/handwash/core/registry.py）。

注册表是"契约层不依赖 torch"的解耦点：模型名来自配置文件里的字符串，如果两个人
各自注册了一个同名实现，最终跑的是谁完全取决于 import 顺序 —— 这类问题不写测试
根本发现不了，所以重名必须报错，除非显式声明 override。
"""

from __future__ import annotations

from typing import Any

import pytest

from handwash.core.registry import (
    DATASETS,
    MODELS,
    TRANSFORMS,
    Registry,
    available_datasets,
    get_dataset_cls,
    get_model_cls,
    register_dataset,
    register_model,
    register_transform,
)
from handwash.errors import ConfigError, ModelNotFoundError

pytestmark = pytest.mark.unit


@pytest.fixture
def registry() -> Registry:
    """每个用例一个全新注册表：全局注册表被测试污染会让其他用例互相干扰。"""
    return Registry("测试模型")


def test_register_returns_the_class_itself(registry: Registry) -> None:
    """注册装饰器必须原样返回类，否则 ``@register(...)`` 之后类名就变成 None 了。"""

    @registry.register("demo")
    class Demo:
        pass

    assert Demo is registry.get("demo")
    assert issubclass(Demo, object)


def test_register_and_get_are_case_insensitive(registry: Registry) -> None:
    """名字来自 YAML/CLI，大小写不该决定能不能跑起来。"""

    @registry.register("Yolo26n-Cls")
    class Model:
        pass

    assert registry.get("yolo26n-cls") is Model
    assert registry.get("  YOLO26N-CLS  ") is Model


def test_duplicate_name_raises_config_error(registry: Registry) -> None:
    """重名默认报错：静默覆盖等于"我训练的不是我配置的那个模型"。"""

    @registry.register("same")
    class First:
        pass

    with pytest.raises(ConfigError, match="名称冲突"):

        @registry.register("Same")
        class Second:
            pass


def test_override_true_replaces_existing_registration(registry: Registry) -> None:
    """确需覆盖时必须显式声明 override=True，代码评审才看得到。"""

    @registry.register("demo")
    class First:
        pass

    @registry.register("demo", override=True)
    class Second:
        pass

    assert registry.get("demo") is Second
    assert registry.names() == ["demo"]


def test_get_missing_name_raises_with_available_names(registry: Registry) -> None:
    """报错必须列出已注册的名字，否则使用者只能靠翻代码猜拼写。"""

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    with pytest.raises(ModelNotFoundError) as excinfo:
        registry.get("gamma")
    message = str(excinfo.value)
    assert "gamma" in message
    assert "alpha" in message and "beta" in message
    assert excinfo.value.name == "gamma"


def test_get_missing_name_on_empty_registry_mentions_nothing_available(registry: Registry) -> None:
    """空注册表也要给出可读的报错，而不是空字符串拼接出奇怪的句子。"""
    with pytest.raises(ModelNotFoundError, match="未注册"):
        registry.get("nothing")


def test_names_are_sorted_and_iterable(registry: Registry) -> None:
    """名字列表要稳定排序，否则 help 输出与日志每次都变。"""
    for name in ("zeta", "alpha", "mid"):
        registry.register(name)(type(name, (), {}))
    assert registry.names() == ["alpha", "mid", "zeta"]
    assert list(registry) == ["alpha", "mid", "zeta"]
    assert len(registry) == 3


def test_contains_checks_membership_case_insensitively(registry: Registry) -> None:
    """``in`` 用于"这个配置项是否受支持"的快速判断。"""
    registry.register("demo")(type("Demo", (), {}))
    assert "demo" in registry
    assert "DEMO" in registry
    assert "other" not in registry


def test_register_rejects_empty_name(registry: Registry) -> None:
    """空名字注册进去之后无法用字符串取回，属于必错。"""
    with pytest.raises(ConfigError, match="非空字符串"):
        registry.register("")


def test_register_normalizes_surrounding_whitespace(registry: Registry) -> None:
    """名字两侧的空格（从 YAML 复制粘贴时很常见）必须被归一化掉。"""
    registry.register("  spaced  ")(type("Spaced", (), {}))
    assert registry.names() == ["spaced"]


def test_register_rejects_whitespace_only_name(registry: Registry) -> None:
    """只有空白的名字（例如从 YAML 复制粘贴多打了空格）必须报错。

    反例（曾经的真实 bug）：先去空白再判空，``"   "`` 会通过"非空"检查，
    最后以空字符串为键注册，成为一个**永远取不到**的幽灵条目。
    现在先 strip 再判空，因此直接抛 ConfigError。
    """
    with pytest.raises(ConfigError, match="非空字符串"):
        registry.register("   ")
    assert registry.names() == []
    assert len(registry) == 0


def test_register_rejects_non_string_name(registry: Registry) -> None:
    """非字符串名字（例如误传 None）必须报错，而不是在字典里塞入奇怪键。"""
    with pytest.raises(ConfigError, match="必须是字符串"):
        registry.register(None)  # type: ignore[arg-type]
    assert registry.names() == []


def test_register_stores_metadata_on_the_class(registry: Registry) -> None:
    """元数据（例如输入尺寸、论文出处）挂在类上，报告与 help 都靠它。"""

    @registry.register("demo", backbone="yolo26n", params=2_400_000)
    class Demo:
        pass

    assert Demo.__registry_metadata__ == {"backbone": "yolo26n", "params": 2_400_000}


def test_registry_kind_appears_in_error_message() -> None:
    """不同注册表的报错要能区分是模型、数据集还是变换，方便定位配置写错在哪。"""
    registry = Registry("数据集")
    with pytest.raises(ConfigError, match="数据集"):
        registry.register("")(type("X", (), {}))


def test_global_registries_support_the_same_protocol() -> None:
    """三个全局注册表必须行为一致，避免"模型能注册、数据集不能"的不对称。"""
    for registry in (MODELS, DATASETS, TRANSFORMS):
        assert isinstance(registry, Registry)
        assert isinstance(registry.names(), list)


def test_module_level_decorators_register_into_global_registries() -> None:
    """便捷装饰器是 models/__init__.py 的实际用法，必须真的写进全局表。"""

    @register_model("pytest-fake-model")
    class FakeModel:
        pass

    @register_dataset("pytest-fake-dataset")
    class FakeDataset:
        pass

    @register_transform("pytest-fake-transform")
    class FakeTransform:
        pass

    try:
        assert get_model_cls("pytest-fake-model") is FakeModel
        assert get_dataset_cls("pytest-fake-dataset") is FakeDataset
        assert "pytest-fake-dataset" in available_datasets()
        assert "pytest-fake-transform" in TRANSFORMS.names()
        assert "pytest-fake-model" in MODELS.names()
    finally:
        # 全局注册表是进程级状态，必须清理，否则会污染同一次 pytest 会话里的其他用例
        for registry, name in (
            (MODELS, "pytest-fake-model"),
            (DATASETS, "pytest-fake-dataset"),
            (TRANSFORMS, "pytest-fake-transform"),
        ):
            registry._items.pop(name, None)  # noqa: SLF001 - 清理测试写入的全局状态


def test_global_registry_names_are_sorted_strings() -> None:
    """全局表的名字用于 CLI help 展示，类型与顺序都要稳定。"""
    names: list[Any] = MODELS.names()
    assert names == sorted(names)
    assert all(isinstance(name, str) for name in names)
