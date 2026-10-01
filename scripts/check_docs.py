#!/usr/bin/env python
"""文档链接体检：确保双语镜像结构里没有断链。

为什么需要它：仓库采用"英文为主 + 中文镜像"的双语文档结构，文件被移动过
（中文版下沉到 docs/zh/），最容易出现的缺陷就是文档里的相对链接失效 ——
而这种错误在 CI 里不会报错，只有读者点进去才发现是 404。

检查内容：
    1. 每个 Markdown 文件里的相对链接（`[文字](路径)`）是否指向真实存在的文件；
    2. 每个双语文档是否都有语言切换行（见 scripts/sync_doc_locales.py 的 PAIRS）；
    3. 语言切换行是否指向正确对端。

用法::

    python scripts/check_docs.py
    python scripts/check_docs.py --fix-switchers   # 顺手补齐缺失的切换行

退出码 0 表示通过；非 0 表示存在断链或缺失的切换行。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT

#: 双语文档配对表（与 scripts/sync_doc_locales.py 保持同步）
PAIRS: tuple[tuple[str, str], ...] = (
    ("README.md", "README.zh-CN.md"),
    ("CONTRIBUTING.md", "docs/zh/CONTRIBUTING.md"),
    ("docs/ARCHITECTURE.md", "docs/zh/ARCHITECTURE.md"),
    ("docs/PROTOCOL.md", "docs/zh/PROTOCOL.md"),
    ("docs/CONFIG.md", "docs/zh/CONFIG.md"),
    ("docs/DATA.md", "docs/zh/DATA.md"),
    ("docs/RULES_CARD.md", "docs/zh/RULES_CARD.md"),
    ("docs/PROJECT_PLAN_4_WEEK.md", "docs/zh/PROJECT_PLAN_4_WEEK.md"),
    ("docs/PROJECT_PROPOSAL.md", "docs/zh/PROJECT_PROPOSAL.md"),
    ("docs/DATA_COLLABORATION.md", "docs/zh/DATA_COLLABORATION.md"),
)

#: 只检查这些文件（其余是内部中文文档，不参与双语结构）
CHECKED_FILES: tuple[str, ...] = (
    "README.md",
    "README.zh-CN.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/ARCHITECTURE.md",
    "docs/CONFIG.md",
    "docs/DATA.md",
    "docs/PROTOCOL.md",
    "docs/RULES_CARD.md",
    "docs/PROJECT_PLAN_4_WEEK.md",
    "docs/IMPLEMENTATION_PLAN.zh-CN.md",
    "docs/PROJECT_PROPOSAL.md",
    "docs/DATA_COLLABORATION.md",
    "docs/EXPERIMENTS.md",
    "docs/SELF_RECORDING.md",
    "docs/zh/ARCHITECTURE.md",
    "docs/zh/CONFIG.md",
    "docs/zh/CONTRIBUTING.md",
    "docs/zh/DATA.md",
    "docs/zh/PROTOCOL.md",
    "docs/zh/RULES_CARD.md",
    "docs/zh/PROJECT_PLAN_4_WEEK.md",
    "docs/zh/MODEL_FRAMEWORK.md",
    "docs/zh/PROJECT_PROPOSAL.md",
    "docs/zh/DATA_COLLABORATION.md",
    "tests/README.md",
    "data/README.md",
)

#: 匹配 Markdown 行内链接与图片链接
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: 允许的外部协议（不检查可达性，避免 CI 依赖网络）
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#")


def iter_links(text: str) -> list[tuple[int, str]]:
    """返回 ``(行号, 链接目标)``，跳过代码块内的内容。"""
    out: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # 行内代码里的 `[...](...)` 不是链接，先摘掉再匹配
        cleaned = re.sub(r"`[^`]*`", "", line)
        for match in _LINK_RE.finditer(cleaned):
            out.append((lineno, match.group(1).strip()))
    return out


def check_links(rel_path: str) -> list[str]:
    """检查一个文件的所有相对链接。返回问题列表。"""
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return [f"{rel_path}: 文件不存在"]

    problems: list[str] = []
    for lineno, target in iter_links(path.read_text(encoding="utf-8")):
        if target.startswith(_EXTERNAL_PREFIXES):
            continue
        # 仓库根绝对路径（GitHub 上可用）
        if target.startswith("/"):
            resolved = PROJECT_ROOT / target.lstrip("/")
        else:
            resolved = (path.parent / target).resolve()
        # 允许带锚点
        candidate = Path(str(resolved).split("#")[0])
        if not candidate.exists():
            problems.append(f"{rel_path}:{lineno}: 断链 -> {target}")
    return problems


def check_switchers() -> list[str]:
    """检查双语文档是否有正确的语言切换行。"""
    problems: list[str] = []
    for en, zh in PAIRS:
        for rel, expected_target in ((en, zh), (zh, en)):
            path = PROJECT_ROOT / rel
            if not path.exists():
                problems.append(f"{rel}: 双语配对文件缺失（对端：{expected_target}）")
                continue
            head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:14])
            if "English" not in head:
                problems.append(f"{rel}: 缺少语言切换行（应有指向 {expected_target} 的链接）")
                continue
            if f"/{expected_target}" not in head:
                problems.append(f"{rel}: 语言切换行没有指向对端 {expected_target}")
    return problems


def _sync_switchers() -> int:
    """调用同步脚本补齐切换行。"""
    import runpy

    script = PROJECT_ROOT / "scripts" / "sync_doc_locales.py"
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文档链接与双语结构体检")
    parser.add_argument("--fix-switchers", action="store_true", help="顺手补齐缺失的语言切换行")
    args = parser.parse_args(argv)

    if args.fix_switchers:
        _sync_switchers()
        print()

    problems: list[str] = []
    for rel in CHECKED_FILES:
        problems.extend(check_links(rel))
    problems.extend(check_switchers())

    if problems:
        print(f"文档体检未通过：{len(problems)} 个问题")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\n修复指引：文件被移动后必须同步重写文档内的相对链接；"
            "语言切换行可用 `python scripts/check_docs.py --fix-switchers` 自动补齐。"
        )
        return 1

    print(f"文档体检通过：检查了 {len(CHECKED_FILES)} 个文件、{len(PAIRS)} 组双语配对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
