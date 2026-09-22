#!/usr/bin/env python
"""一次性维护脚本：给双语文档加/更新语言切换头。

用途：仓库采用"英文为主版本 + 中文镜像"的双语文档结构：
    英文（主）  英文（主）
    README.md                 <->  README.zh-CN.md
    CONTRIBUTING.md           <->  docs/zh/CONTRIBUTING.md
    docs/ARCHITECTURE.md      <->  docs/zh/ARCHITECTURE.md
    docs/PROTOCOL.md          <->  docs/zh/PROTOCOL.md
    docs/CONFIG.md            <->  docs/zh/CONFIG.md
    docs/DATA.md              <->  docs/zh/DATA.md
    docs/RULES_CARD.md        <->  docs/zh/RULES_CARD.md

本脚本做两件事：
    1. 在每个文件正文的**第一行**插入（或就地替换）语言切换行；
    2. 幂等：重复运行不会叠加，也不会改动正文其余内容。

切换行使用**仓库根绝对路径**（以 / 开头），例如 `/docs/zh/CONFIG.md`。
这样无论文件处在哪一层目录、无论从哪个页面点进来都能正确跳转，
也避免了"中文版比英文版深一层，相对路径要写 ../"这类易错细节。

规则依据：CONTRIBUTING.md R2（UTF-8 + LF）。本脚本属于一次性维护工具，
新增双语文档时改下面的 PAIRS 表并重跑即可。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (英文主版本路径, 中文镜像路径)，均为相对仓库根
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

#: 切换行模板：英文版指向中文镜像，中文版指向英文主版本
_EN_ROW = "> **English** | [中文](/{zh})"
_ZH_ROW = "> [English](/{en}) | **中文**"

#: 已经存在的切换行（用于幂等替换）
_SWITCHER_RE = re.compile(r"^>\s*(?:\*\*English\*\*|\[English\]).*$")


def switcher_for(rel_path: str, *, is_english: bool) -> str:
    """生成该文件应插入的切换行。"""
    for en, zh in PAIRS:
        if is_english and rel_path == en:
            return _EN_ROW.format(zh=zh)
        if not is_english and rel_path == zh:
            return _ZH_ROW.format(en=en)
    raise KeyError(f"文件不在 PAIRS 表里：{rel_path}")


def apply(path: Path, row: str) -> str:
    """在正文起始处插入/就地更新切换行。幂等。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "empty"

    # 1) 幂等：文件开头附近若已有切换行，就地替换
    for index in range(min(14, len(lines))):
        if _SWITCHER_RE.match(lines[index]):
            if lines[index] == row:
                return "unchanged"
            lines[index] = row
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            return "replaced"

    # 2) 否则插到第一个 H1 之前（没有 H1 就插到最前面）
    insert_at = 0
    for index in range(min(12, len(lines))):
        if lines[index].startswith("# "):
            insert_at = index
            break
    lines[insert_at:insert_at] = [row, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return "inserted"


def main() -> int:
    results: list[tuple[str, str, bool]] = []
    for en, zh in PAIRS:
        for rel, is_english in ((en, True), (zh, False)):
            path = ROOT / rel
            if not path.exists():
                results.append((rel, "MISSING", False))
                continue
            try:
                row = switcher_for(rel, is_english=is_english)
            except KeyError:
                row = _ZH_ROW.format(en=en) if not is_english else _EN_ROW.format(zh=zh)
            status = apply(path, row)
            results.append((rel, status, True))

    width = max(len(rel) for rel, _, _ in results)
    for rel, status, ok in results:
        mark = "OK " if ok and status != "MISSING" else "!! "
        print(f"{mark}{rel:<{width}}  {status}")

    missing = [rel for rel, status, _ in results if status == "MISSING"]
    if missing:
        print(f"\n缺失文件 {len(missing)} 个：{missing}")
        print("（英文主版本可能还在翻译中；翻译完成后重跑本脚本即可。）")
        return 0
    print(f"\n完成：{len(results)} 个文件已处理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
