"""统一日志契约测试（src/handwash/logging.py）。

规则（CONTRIBUTING R8）：全项目禁止 ``print`` 输出运行信息。日志的一个典型事故是
重复配置 handler —— 一个日志打印三遍，或者导入某个模块时把根 logger 配成 DEBUG
导致第三方库刷屏。所以这里锁住命名规则与 setup_logging 的幂等性。
"""

from __future__ import annotations

import logging

import pytest

from handwash.logging import DEFAULT_LEVEL, LOG_FORMAT, get_logger, setup_logging

pytestmark = pytest.mark.unit


# --- get_logger -------------------------------------------------------------
def test_get_logger_prefixes_plain_name_with_package() -> None:
    """裸名字必须落到 handwash.* 命名空间，方便统一控制级别与输出。"""
    assert get_logger("x").name == "handwash.x"


def test_get_logger_keeps_already_prefixed_name() -> None:
    """``get_logger("handwash.foo")`` 不能再加一层前缀，否则过滤规则会失效。"""
    assert get_logger("handwash.foo").name == "handwash.foo"
    assert get_logger("handwash").name == "handwash"


def test_get_logger_does_not_double_prefix_deep_module_names() -> None:
    """``__name__`` 传入时本身就是 handwash.core.labels 这种全名。"""
    assert get_logger("handwash.core.labels").name == "handwash.core.labels"


def test_get_logger_is_stable_and_returns_the_same_object() -> None:
    """logging 模块自己缓存 logger 对象；重复取用不能产生新实例。"""
    assert get_logger("dup") is get_logger("dup")


def test_get_logger_keeps_dunder_and_unknown_namespaces_untouched() -> None:
    """``__main__`` 这类特殊名字不加前缀，否则脚本日志会被改名。"""
    assert get_logger("__main__").name == "__main__"


def test_get_logger_returns_a_usable_logger() -> None:
    """返回对象必须能直接调用 info/debug（配置由 get_logger 内部保证）。"""
    logger = get_logger("usable")
    assert isinstance(logger, logging.Logger)
    logger.debug("这行不应让测试失败")


# --- setup_logging ----------------------------------------------------------
def test_setup_logging_is_idempotent() -> None:
    """入口脚本与 notebook 顶部都会调用它，重复调用不能叠加 handler（否则日志打印多遍）。"""
    setup_logging(force=True)
    root = logging.getLogger("handwash")
    handlers_after_first = list(root.handlers)

    setup_logging()
    setup_logging()

    assert root.handlers == handlers_after_first
    assert len(root.handlers) == 1


def test_setup_logging_sets_level_and_stops_propagation() -> None:
    """handwash logger 自己输出，不能向上冒泡到 root（否则会和 pytest 的捕获重复）。"""
    setup_logging("DEBUG", force=True)
    root = logging.getLogger("handwash")
    assert root.level == logging.DEBUG
    assert root.propagate is False


def test_setup_logging_uses_shared_format_and_stream() -> None:
    """所有实验输出格式必须一致，报告与日志才好对照。"""
    setup_logging(force=True)
    handler = logging.getLogger("handwash").handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter is not None
    assert handler.formatter._fmt == LOG_FORMAT

def test_setup_logging_quiets_noisy_third_party_loggers() -> None:
    """ultralytics / matplotlib 的 INFO 会淹没实验日志，必须降噪。"""
    setup_logging(force=True)
    for noisy in ("matplotlib", "PIL", "urllib3", "filelock", "numexpr"):
        assert logging.getLogger(noisy).level == logging.WARNING


def test_setup_logging_accepts_numeric_level() -> None:
    """允许直接传 logging 常量，方便在代码里按需调整。"""
    setup_logging(logging.WARNING, force=True)
    assert logging.getLogger("handwash").level == logging.WARNING


def test_setup_logging_does_not_touch_unrelated_loggers_level() -> None:
    """降噪白名单之外的 logger 不该被改写级别，否则会隐藏别人的调试信息。"""
    setup_logging(force=True)
    assert logging.getLogger("some.other.library").level == logging.NOTSET


def test_default_level_constant_is_a_valid_logging_name() -> None:
    """DEFAULT_LEVEL 是字符串形式，必须能被 getattr(logging, ...) 解析。"""
    assert isinstance(DEFAULT_LEVEL, str)
    assert isinstance(getattr(logging, DEFAULT_LEVEL), int)


def test_level_env_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI 里用 HANDWASH_LOG_LEVEL=DEBUG 打开详细输出，不改代码。"""
    monkeypatch.setenv("HANDWASH_LOG_LEVEL", "DEBUG")
    setup_logging(force=True)
    assert logging.getLogger("handwash").level == logging.DEBUG
    monkeypatch.delenv("HANDWASH_LOG_LEVEL", raising=False)


def test_unknown_level_env_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量写错不应导致 AttributeError，回退到 INFO 最合理。"""
    monkeypatch.setenv("HANDWASH_LOG_LEVEL", "SHOUTING")
    setup_logging(force=True)
    assert logging.getLogger("handwash").level == logging.INFO
    monkeypatch.delenv("HANDWASH_LOG_LEVEL", raising=False)


@pytest.mark.parametrize("method_name", ["_fmt", "_style"])
def test_formatter_template_is_the_module_constant(method_name: str) -> None:
    """两种取模板的方式（旧 ``_fmt`` / 新 ``_style._fmt``）都应指向同一常量。"""
    setup_logging(force=True)
    formatter = logging.getLogger("handwash").handlers[0].formatter
    assert formatter is not None
    template = getattr(formatter, method_name)
    if method_name == "_style":
        template = template._fmt
    assert template == LOG_FORMAT
