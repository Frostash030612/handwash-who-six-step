#!/usr/bin/env python
"""评估脚本：内部测试 + 跨场景测试，并打印对比表。

用法::

    python scripts/evaluate_model.py --checkpoint outputs/e2/models/best.pt
    python scripts/evaluate_model.py --checkpoint outputs/e2/models/best.pt --splits test external

报告里需要的三组数字（同场景 / 跨场景 / 自采）建议分三次调用后汇总。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: F401
from _common import build_config_parser, load_config_from_args
from handwash.cli import split_argv
from handwash.core.config import parse_overrides
from handwash.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认 <out_dir>/models/best.pt）")
    parser.add_argument("--splits", nargs="*", default=None, help="要评估的 split，如 test external")

    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)

    rc = load_config_from_args(args, parse_overrides(override_items))

    from handwash.pipelines.evaluate import evaluate

    bundle = evaluate(rc, checkpoint=args.checkpoint, splits=args.splits)
    print("\n" + bundle.summary_table())
    print("\n产物目录：", rc.resolve_out_dir() / "eval")
    print("报告里请直接引用 confusion_matrix_*.png 与 per_class_*.csv。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
