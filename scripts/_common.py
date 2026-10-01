#!/usr/bin/env python
"""通用配置引导：让 ``scripts/`` 下的脚本共用一套参数解析。

约定（CONTRIBUTING.md R19）：脚本里**不要**自己写 argparse 的参数定义，
一律用这里的 ``build_config_parser``，这样 ``config=`` 叠加 + ``key=value``
覆盖的语义在所有脚本里完全一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: F401  （副作用：注册导入路径）
from handwash.core.config import ResolvedConfig, load_config, parse_overrides
from handwash.logging import setup_logging

__all__ = ["add_common_flags", "build_config_parser", "resolve_from_args"]


def add_common_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给脚本加上统一的配置相关参数。"""
    parser.add_argument(
        "--config",
        default=None,
        help="叠加配置文件（默认 configs/config.yaml；可多次指定，后者覆盖前者）",
        action="append",
    )
    parser.add_argument("--run-name", default=None, help="覆盖 runtime.run_name（输出目录名）")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING / ERROR")
    return parser


def build_config_parser(description: str) -> argparse.ArgumentParser:
    """构造一个只含通用参数的解析器（脚本可再加自己的参数）。"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "额外可用位置参数：key.sub=value（与 Makefile / handwash CLI 写法一致）\n"
            "示例：python scripts/train.py config=configs/experiments/smoke.yaml train.epochs=2"
        ),
    )
    return add_common_flags(parser)


def _split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    """复用 CLI 的切分逻辑，保证行为一致。"""
    from handwash.cli import split_argv

    return split_argv(argv)


def resolve_from_args(argv: list[str] | None = None, *, description: str = "") -> tuple[Any, ResolvedConfig]:
    """解析 ``sys.argv``，返回 ``(args, resolved_config)``。

    返回值里已经包含配置覆盖与日志初始化，脚本可以立刻开干。
    """
    import sys

    raw = list(sys.argv[1:] if argv is None else argv)
    plain, override_items = _split_overrides(raw)

    parser = build_config_parser(description)
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")

    setup_logging(args.log_level, force=True)
    rc = load_config_from_args(args, parse_overrides(override_items))
    return args, rc


def load_config_from_args(args: Any, overrides: dict[str, Any]) -> ResolvedConfig:
    """把 ``--config`` / ``config=`` / ``--run-name`` 与覆盖项合并成最终配置。

    为什么必须有这个函数（真实 bug 来源）：
        CLI 侧（``handwash.cli._load``）会把位置参数 ``config=<文件>`` 从覆盖项里
        **摘出来**再合并，因为它不是 AppConfig 的字段；
        但 scripts/ 下的几个脚本各自手写 ``load_config(list(args.config), overrides=overrides)``，
        忘了摘 —— 于是 ``config=configs/data/pskuss.yaml`` 被当成"未知配置键 config"，
        报错信息还很不直观。这里统一一份实现，所有脚本复用，避免再各写一遍。

    叠加语义与 CLI 完全一致：
        * ``--config <文件>``（可多次）：作为**基础**配置，默认 ``configs/config.yaml``；
        * 位置参数 ``config=<文件>``（可多次）：作为**叠加层**，按顺序覆盖前面的。
    """
    overrides = dict(overrides)
    overlay = overrides.pop("config", None)
    paths: list[str] = list(args.config or ["configs/config.yaml"])
    if overlay is not None:
        # 位置参数可重复，覆盖项解析成的是单个值或列表
        paths.extend(overlay if isinstance(overlay, list) else [overlay])

    if getattr(args, "run_name", None):
        runtime = {**overrides.get("runtime", {}), "run_name": args.run_name}
        overrides["runtime"] = runtime

    return load_config([str(path) for path in paths], overrides=overrides)
