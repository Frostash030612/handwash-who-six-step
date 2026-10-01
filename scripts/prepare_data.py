#!/usr/bin/env python
"""数据准备脚本：扫描 -> 划分 -> 抽帧 -> 生成 manifest。

等价于 ``handwash prepare``，但额外支持：
    * ``--inspect``：只统计不写盘，用于先看清数据规模再决定是否全量抽帧。

用法::

    python scripts/prepare_data.py config=configs/data/kaggle.yaml
    python scripts/prepare_data.py config=configs/data/pskuss.yaml --inspect
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_config_parser, load_config_from_args  # noqa: E402

from handwash.cli import split_argv  # noqa: E402
from handwash.core.config import parse_overrides  # noqa: E402
from handwash.logging import setup_logging  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--stage", choices=("all", "scan", "split", "frames"), default="all")
    parser.add_argument("--inspect", action="store_true", help="只统计，不写任何文件")

    plain, override_items = split_argv(argv)
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)

    rc = load_config_from_args(args, parse_overrides(override_items))

    if args.inspect:
        return _inspect(rc)

    from handwash.pipelines.prepare import prepare

    result = prepare(rc, stage=args.stage)
    print(f"\n准备完成：{result.num_clips} 段视频 / {result.num_frames} 帧")
    if result.manifest_path:
        print(f"manifest：{result.manifest_path}")
    for split, info in result.splits.items():
        print(f"  {split:<8} {info['num_clips']:>5} 段 / {info['num_frames']:>7} 帧")
    if args.stage == "scan":
        print("\n下一步：python scripts/prepare_data.py --stage split")
    elif args.stage == "split":
        print("\n下一步：python scripts/prepare_data.py --stage frames")
    else:
        print("\n下一步：python scripts/train_model.py（或 make train）")
    return 0


def _inspect(rc) -> int:
    """只统计原始数据的规模与标签分布，不写盘。

    用途：18GB 的 PSKUS 全量抽帧前，先确认目录结构和标签映射对不对。
    """
    from collections import Counter

    from handwash.core.labels import Step
    from handwash.pipelines.prepare import (
        _discover_source,
        _looks_like_image_tree,
        _records_from_frames_dir,
        _records_from_metc,
        _records_from_pskuss,
        _records_from_video_dirs,
    )

    if rc.dataset.name == "synthetic":
        print("synthetic 是临时生成的数据集，没有原始文件可扫描；用 configs/experiments/smoke.yaml 检查流程。")
        return 0

    name, base = _discover_source(rc)
    if name == "kaggle" or (name != "metc" and _looks_like_image_tree(base)):
        frames, clips = _records_from_frames_dir(rc, base)
    elif name == "pskuss":
        frames, clips = _records_from_pskuss(rc, base)
    elif name == "metc":
        frames, clips = _records_from_metc(rc, base)
    else:
        frames, clips = _records_from_video_dirs(rc, base)

    labels = Counter(record.label.value for record in frames if record.label is not Step.UNKNOWN)
    unknown = sum(record.label is Step.UNKNOWN for record in frames)
    print(f"数据集：{name}（标签空间 {rc.label_space}）")
    print(f"根目录：{base}")
    print(f"视频/片段：{len(clips)}")
    print(f"标注记录：{len(frames)}")
    print("标签记录数：")
    for label, count in sorted(labels.items()):
        print(f"  {label:<36} {count:>8}")
    if unknown:
        print(f"  unknown（会在抽帧时跳过）        {unknown:>8}")
    print("\n统计来自数据适配器，不解码视频、不写入文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
