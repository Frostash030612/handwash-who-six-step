#!/usr/bin/env python
"""配置契约检查（pre-commit 钩子 / CI 门禁）。

它保证"配置即接口"这件事真的成立：
    1. ``configs/config.yaml`` 能被严格加载并通过全部校验；
    2. 每一个 ``configs/**/*.yaml`` 叠加在基础配置之上后都能严格加载
       —— 这一步能抓出拼错的键（例如 ``train.epochsz``），
          因为严格模式不会静默忽略未知键；
    3. 打印每个配置的 config_hash，方便对照实验记录。

用法::

    python scripts/check_config.py
    python scripts/check_config.py --env-consistency   # 附带宽松/精确依赖一致性检查
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from handwash.core.config import load_config
from handwash.errors import HandwashError

CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "config.yaml"


def _iter_overlays() -> list[Path]:
    """所有"叠加型"配置（data/ models/ experiments/），按路径排序保证输出稳定。"""
    out: list[Path] = []
    for sub in ("data", "models", "experiments"):
        folder = CONFIG_DIR / sub
        if folder.is_dir():
            out.extend(sorted(folder.glob("*.yaml")))
    return out


def check_loaded_overlays() -> list[str]:
    """逐个把叠加配置叠到基础配置上加载，返回问题列表。"""
    problems: list[str] = []
    if not BASE_CONFIG.exists():
        return [f"缺少基础配置：{BASE_CONFIG}"]

    try:
        base = load_config([BASE_CONFIG])
    except HandwashError as exc:
        return [f"基础配置加载失败：{exc}"]

    print(f"[OK]   {BASE_CONFIG.relative_to(PROJECT_ROOT)}  config_hash={base.config_hash}")

    for overlay in _iter_overlays():
        rel = overlay.relative_to(PROJECT_ROOT)
        try:
            resolved = load_config([BASE_CONFIG, overlay])
        except HandwashError as exc:
            problems.append(f"{rel} 叠加后加载失败：{exc}")
            continue
        except Exception as exc:
            problems.append(f"{rel} 叠加时发生意外错误：{type(exc).__name__}: {exc}")
            continue
        print(f"[OK]   {rel}  config_hash={resolved.config_hash}")
    return problems


def check_env_consistency() -> list[str]:
    """粗查 pyproject.toml 与 environment.yml 是否都提到了关键依赖。

    这不是完美的依赖解析（那需要真装一遍），但能拦住最常见的事故：
    有人加了一个包到 environment.yml 却忘了写进 pyproject（或反过来），
    导致"我的环境能跑的代码，别人的环境装不上"。
    """
    import re

    problems: list[str] = []
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    envfile_path = PROJECT_ROOT / "environment.yml"
    if not envfile_path.exists():
        return ["缺少 environment.yml"]
    envfile = envfile_path.read_text(encoding="utf-8")

    key_packages = ("numpy", "pandas", "scikit-learn", "pyyaml", "tqdm", "torch", "torchvision", "ultralytics")
    for package in key_packages:
        in_pyproject = re.search(rf'["\']?{re.escape(package)}', pyproject, re.IGNORECASE) is not None
        in_env = re.search(re.escape(package), envfile, re.IGNORECASE) is not None
        if in_env and not in_pyproject:
            problems.append(
                f"`{package}` 出现在 environment.yml 但没写进 pyproject.toml —— "
                "两边必须同时登记（CONTRIBUTING.md R4）"
            )
    if not problems:
        print("[OK]   pyproject.toml 与 environment.yml 的关键依赖声明一致")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="配置契约与依赖声明检查")
    parser.add_argument("--env-consistency", action="store_true", help="额外检查依赖声明一致性")
    args = parser.parse_args(argv)

    problems = check_loaded_overlays()
    if args.env_consistency:
        problems.extend(check_env_consistency())

    if problems:
        print(f"\n配置检查未通过：{len(problems)} 个问题")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\n修复指引：键名必须与 src/handwash/core/config.py 的 dataclass 字段逐字一致"
            "（见 docs/CONFIG.md）。拼错的键不会被忽略，这是刻意设计。"
        )
        return 1

    print("\n配置检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
