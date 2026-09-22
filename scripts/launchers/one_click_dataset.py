#!/usr/bin/env python
"""双击友好的数据准备入口：一键把数据集从零跑到"可以发给队友的分享包"。

这是给**不熟悉命令行**的组员用的入口。双击下面任一文件即可：

    Windows          scripts/launchers/一键出结果.bat
    macOS / Linux    scripts/launchers/一键出结果.command

它自己只做三件事：定位仓库、打印说明、调用 ``scripts/bootstrap_dataset.py``。
之所以不让双击入口带任何选项（除了 ``--dataset``）：双击的人不一定读参数说明，
留了选项就可能出现"只下了一半分片就产出划分"这类**不会有任何报错**的错误 ——
产出的 manifest 看起来完全正常，只是全组实验结果都不可比。
需要 ``--shards`` / ``--skip-*`` 等变体时，请直接使用命令行脚本。

本文件也可以直接运行::

    python scripts/launchers/one_click_dataset.py
    python scripts/launchers/one_click_dataset.py --dataset metc
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: 仓库根目录：本文件在 scripts/launchers/ 下，所以要上跳两级
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 真正干活的脚本
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_dataset.py"


def _pause_if_double_clicked() -> None:
    """双击运行时（终端窗口是一次性的）在结束前等一下，否则错误信息会一闪而过。

    判定方式：Windows 上双击 .bat 时 stdin 不是交互式终端；
    稳妥起见只在"非交互式"时暂停，命令行里运行则直接退出。
    """
    if sys.stdin is not None and sys.stdin.isatty():
        return
    try:
        input("\n按回车键关闭此窗口……")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键准备数据（双击版）：下载 -> 抽帧 -> 划分 -> 打包",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="pskuss",
        help="数据集名（默认 pskuss；可选 metc / jurmala）",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="结束后不等待回车（供脚本调用）",
    )
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"忽略无法识别的参数：{unknown}")
        print("（双击入口不支持 --shards / --skip-* 等变体；请改用 scripts/bootstrap_dataset.py）")
        print()

    if not BOOTSTRAP.exists():
        print(f"[错误] 找不到 {BOOTSTRAP}")
        print("       请确认本文件位于仓库的 scripts/launchers/ 目录下，")
        print("       且 scripts/bootstrap_dataset.py 存在（git pull 一下试试）。")
        if not args.no_pause:
            _pause_if_double_clicked()
        return 1

    print(f"仓库根目录 : {REPO_ROOT}")
    print(f"数据集     : {args.dataset}")
    print(f"即将执行   : python scripts/bootstrap_dataset.py --dataset {args.dataset}")
    print()

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", str(BOOTSTRAP), "--dataset", args.dataset],
        cwd=str(REPO_ROOT),
        check=False,
    )

    if completed.returncode != 0:
        print()
        print("=" * 70)
        print("没有全部完成。最常见的原因是「数据没下全」——")
        print("直接重新双击一次即可，脚本会自动补齐缺失的分片（已有的会校验后跳过）。")
        print("=" * 70)

    if not args.no_pause:
        _pause_if_double_clicked()
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
