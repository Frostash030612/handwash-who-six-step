"""统一日志：全项目禁止使用 ``print`` 输出运行信息（见 CONTRIBUTING.md R8）。

用法::

    from handwash.logging import get_logger
    log = get_logger(__name__)
    log.info("loading %s", path)      # 惰性格式化，禁止 f-string 传参

本模块属于 L0 层，只能依赖标准库。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

__all__ = ["get_logger", "setup_logging", "LOG_FORMAT", "DEFAULT_LEVEL"]

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL: Final[str] = "INFO"

_CONFIGURED: bool = False


def _level_from_env() -> int:
    raw = os.environ.get("HANDWASH_LOG_LEVEL", DEFAULT_LEVEL).upper()
    return getattr(logging, raw, logging.INFO)


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """配置根 logger。CLI 入口与 notebooks 顶部各调用一次即可（幂等）。"""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = _level_from_env() if level is None else level
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger("handwash")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    root.propagate = False

    # 第三方库降噪：否则 ultralytics / matplotlib 会淹没实验日志
    for noisy in ("matplotlib", "PIL", "urllib3", "filelock", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """返回 ``handwash.*`` 命名空间下的 logger。"""
    setup_logging()
    if name.startswith("handwash") or name.startswith("__"):
        return logging.getLogger(name)
    return logging.getLogger(f"handwash.{name}")
