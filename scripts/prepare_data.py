#!/usr/bin/env python
"""数据准备脚本：扫描 -> 划分 -> 抽帧 -> 生成 manifest。

等价于 ``handwash prepare``，但额外支持：
    * ``--inspect``：只统计不写盘，用于先看清数据规模再决定是否全量抽帧。

用法::

    python scripts/prepare_data.py --config configs/data/kaggle.yaml
    python scripts/prepare_data.py --config configs/data/pskuss.yaml --inspect
"""

#!/usr/bin/env python
"""数据准备脚本：扫描 -> 划分 -> 抽帧 -> 生成 manifest。

等价于 ``handwash prepare``，但额外支持：
    * ``--inspect``：只统计不写盘，用于先看清数据规模再决定是否全量抽帧。

用法::

    python scripts/prepare_data.py --config configs/data/kaggle.yaml
    python scripts/prepare_data.py --config configs/data/pskuss.yaml --inspect
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from _common import build_config_parser, load_config_from_args  # noqa: E402

from handwash.cli import split_argv  # noqa: E402
from handwash.core.config import load_config, parse_overrides  # noqa: E402
from handwash.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    argv = sys.argv[1:]
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--stage", choices=("all", "scan", "split", "frames"), default="all")
    parser.add_argument("--inspect", action="store_true", help="只统计，不写任何文件")

    from handwash.cli import split_argv

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
    print("\n下一步：python scripts/train_model.py（或 make train）")
    return 0


def _inspect(rc) -> int:
    """只统计原始数据的规模与标签分布，不写盘。

    用途：18GB 的 PSKUS 全量抽帧前，先确认目录结构和标签映射对不对。
    """
    from collections import Counter
    from pathlib import Path

    from handwash.core.labels import get_label_space
    from handwash.io.utils import VIDEO_EXTENSIONS, list_files
    from handwash.paths import PROJECT_ROOT as root

    spec = rc.dataset_spec()
    base = Path(str(spec.get("root") or rc.dataset.root))
    if not base.is_absolute():
        base = root / base

    space = get_label_space(rc.label_space)
    videos = list_files(base, extensions=VIDEO_EXTENSIONS, recursive=True)
    by_label: Counter[str] = Counter()
    unknown: Counter[str] = Counter()

    for video in videos:
        # 目录名作为标签（各数据集的通用约定）
        candidate = video.parent.name
        try:
            by_label[space.canonicalize(candidate).value] += 1
        except Exception:  # noqa: BLE001 - 无法识别的目录名：统计出来给人看
            unknown[candidate] += 1

    print(f"数据集：{rc.dataset.name}（标签空间 {rc.label_space}）")
    print(f"根目录：{base}")
    print(f"视频总数：{len(videos)}")
    print("按目录名推断的标签分布：")
    for name, count in by_label.most_common():
        print(f"  {name:<32} {count:>6}")
    if unknown:
        print("无法识别的目录名（需要补 core/labels.py 的别名映射）：")
        for name, count in unknown.most_common(20):
            print(f"  {name!r:<32} {count:>6}")
    print("\n这只是估算：真实划分与帧数在 prepare 阶段确定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
