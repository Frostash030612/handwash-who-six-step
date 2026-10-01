#!/usr/bin/env python
"""`.gitignore` 的数据规则体检：确认"该忽略的被忽略、该提交的能提交"。

**为什么需要这个脚本**

`.gitignore` 的数据规则踩过三次坑，每次都是"静默失效"——没有报错，只是行为变了：

  1. 早期写成"排除 `data/raw/` 整个目录、再放行 SOURCES.json"。Git 的铁律是
     *被排除的目录内无法再包含文件*，所以 SOURCES.json 实际进不了 Git。
  2. 调整规则顺序后，`!data/**` 把按扩展名排除的规则推翻，
     `DataSet4.zip` 与 `00000.jpg` 变成"可提交"——17 GB 数据差点被提交。
  3. 更严重的一次：`manifest.csv`（划分契约）被提交进 Git，而它来自只下了 1/11 分片的
     子集（39 段 / 全集 3185 段）。**任何人 pull 下来都会拿到这份错误的划分并在上面训练**，
     而文件本身看起来完全正常。

本脚本用 `git check-ignore` 对着真实路径断言，把这三类事故都钉死。
它在 `make check`、pre-commit 与 CI 里都会跑（改 .gitignore 是最容易顺手犯的错）。

用法::

    python scripts/check_data_ignores.py
    python scripts/check_data_ignores.py --verbose
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT

#: (路径, 是否应当被忽略, 为什么)
#:
#: 这些路径是**当前实际存在或即将存在**的，不是假想路径 —— 白名单故意做得很窄
#: （只有 data/README.md 与 data/raw/SOURCES.json），因为 gitignore 无法干净地
#: 表达"任意深度只放行某文件名"，范围一大就会静默失效。
CASES: tuple[tuple[str, bool, str], ...] = (
    # --- 必须被忽略：大数据 -------------------------------------------------
    ("data/raw/pskuss/DataSet4.zip", True, "原始分片 17.1 GiB，绝不能进 Git"),
    ("data/raw/DataSet1.zip", True, "原始分片"),
    ("data/raw/pskuss/extracted/DataSet4/Videos/a.mp4", True, "原始视频"),
    ("data/raw/pskuss/extracted/DataSet4/Annotations/Annotator1/a.csv", True, "逐帧标注"),
    ("data/raw/pskuss/extracted/DataSet4/Annotations/Annotator1/a.json", True, "标注元数据"),
    ("data/raw/pskuss/extracted/DataSet4/statistics.csv", True, "数据集自带统计"),
    ("data/raw/pskuss/DataSet4.zip.extracted", True, "解压完成标记，本机产物"),
    ("data/raw/pskuss/SHARD_PLAN.md", True, "分片方案，本机生成的中间文件"),
    # --- 必须被忽略：抽帧与评估产物 ----------------------------------------
    ("data/processed/pskuss/frames/c/00000.jpg", True, "抽帧图像，数千张"),
    ("data/processed/pskuss/predictions_test.jsonl", True, "逐帧预测"),
    ("data/interim/synthetic/frames/syn_0000/00000.jpg", True, "合成数据产物"),
    ("data/external/self_recorded/full/demo.mp4", True, "组员自采原始视频"),
    # --- 必须被忽略：划分契约（最重要的两条）-------------------------------
    (
        "data/processed/pskuss/manifest.csv",
        True,
        "划分契约：子集产生的 manifest 一旦提交，全组都会用到错误划分"
        "（真实事故：一份 39 段 / 全集 3185 段的 manifest 被提交）",
    ),
    ("data/processed/pskuss/split_report.json", True, "同上，划分报告"),
    # --- 必须能提交：白名单里的两个文件 ------------------------------------
    ("data/raw/SOURCES.json", False, "来源 + md5 清单，协作契约的一部分"),
    ("data/README.md", False, "data/ 目录说明（唯一一份）"),
)


def is_ignored(path: str) -> bool:
    """问 git 这个路径是否被忽略（返回码 0 = 被忽略）。"""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore 异常退出（{result.returncode}）："
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=".gitignore 数据规则体检")
    parser.add_argument("--verbose", action="store_true", help="打印全部用例")
    args = parser.parse_args(argv)

    if not (PROJECT_ROOT / ".git").exists():
        print("不在 Git 仓库内，跳过检查。")
        return 0

    problems: list[str] = []
    for path, should_ignore, why in CASES:
        ignored = is_ignored(path)
        ok = ignored == should_ignore
        if args.verbose or not ok:
            mark = "OK  " if ok else "FAIL"
            state = "被忽略" if ignored else "可提交"
            expect = "应被忽略" if should_ignore else "应可提交"
            print(f"  [{mark}] {state:<6} （{expect}） {path}")
            if not ok:
                print(f"         {why}")
        if not ok:
            problems.append(f"{path}：{state}，但{expect}（{why}）")

    if problems:
        print(f"\n.gitignore 数据规则体检未通过：{len(problems)} 个问题")
        for item in problems:
            print(f"  - {item}")
        print(
            "\n修复指引：见 .gitignore 顶部『数据』一节的四步说明 ——\n"
            "    1) `data/*` 顶层默认全忽略；\n"
            "    2) `!data/raw/` 放行目录本身；\n"
            "    3) `data/raw/**` 忽略其下全部内容；\n"
            "    4) 只白名单放行 `data/README.md` 与 `data/raw/SOURCES.json`。\n"
            "  放行新的深层文件时，必须把路径上每一级目录都显式放行，\n"
            "  否则规则会静默失效 —— 改完务必再跑一次本体检。\n"
            "  不要在 `.gitignore` 之外用 `git add -f` 绕过本检查。"
        )
        return 1

    print(f".gitignore 数据规则体检通过：检查了 {len(CASES)} 条路径。")
    if not args.verbose:
        print("  （用 --verbose 看逐条结果）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
