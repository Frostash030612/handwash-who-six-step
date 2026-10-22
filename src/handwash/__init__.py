"""Handwashing action recognition and assessment; packaged source utilities."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["PROJECT_ROOT", "__version__", "ensure_src_on_path"]

__version__ = "0.1.0"


def ensure_src_on_path() -> Path:

    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.is_dir():
        src_str = str(src)
        if src_str in sys.path:
            sys.path.remove(src_str)
        sys.path.insert(0, src_str)
    return root


try:
    from handwash.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = ensure_src_on_path()
