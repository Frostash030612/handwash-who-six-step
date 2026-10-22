"""Application logging configuration and named loggers."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

__all__ = ["DEFAULT_LEVEL", "LOG_FORMAT", "get_logger", "setup_logging"]

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL: Final[str] = "INFO"

_CONFIGURED: bool = False


def _level_from_env() -> int:
    raw = os.environ.get("HANDWASH_LOG_LEVEL", DEFAULT_LEVEL).upper()
    return getattr(logging, raw, logging.INFO)


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:

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

    for noisy in ("matplotlib", "PIL", "urllib3", "filelock", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:

    setup_logging()
    if name.startswith("handwash") or name.startswith("__"):
        return logging.getLogger(name)
    return logging.getLogger(f"handwash.{name}")
