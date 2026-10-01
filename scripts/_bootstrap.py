"""轻量脚本引导（供 ``scripts/`` 下的入口使用）。

为什么需要它：src-layout 下直接 ``python scripts/train.py`` 时，
Python 找得到 ``scripts/`` 却找不到 ``src/handwash``。
与其让每个脚本各写一遍 sys.path 魔法（极易写错且难以维护），
统一在这里做一次。

用法（每个 scripts/*.py 开头）::

    try:                      # 直接运行：python scripts/foo.py
        from _bootstrap import PROJECT_ROOT
    except ImportError:       # 以模块方式运行：python -m scripts.foo
        from ._bootstrap import PROJECT_ROOT
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "cli_main",
    "handwash_root",
    "run_cli",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"


def handwash_root() -> Path:
    """把 ``src`` 与仓库根加入 ``sys.path``（幂等），返回仓库根目录。

    加入仓库根是为了让 ``scripts`` 能作为包被导入（``python -m scripts.x``）。
    """
    for entry in (str(SRC_DIR), str(PROJECT_ROOT)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return PROJECT_ROOT


def cli_main(argv: list[str] | None = None) -> int:
    """转发到 ``handwash`` CLI（让脚本可以 `--` 之后直接透传子命令）。"""
    from handwash.cli import main

    return main(argv)


def run_cli(argv: list[str] | None = None) -> int:
    """``cli_main`` 的别名，语义更直白：把参数交给 CLI 处理。"""
    return cli_main(argv)


handwash_root()
