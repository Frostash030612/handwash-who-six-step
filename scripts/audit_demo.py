#!/usr/bin/env python
"""批量核验本地视频：exp.pt 逐帧基线与 GRU 的六步报告并排输出。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: F401
from _common import build_config_parser, load_config_from_args
from handwash.cli import split_argv
from handwash.core.config import parse_overrides
from handwash.errors import HandwashError
from handwash.logging import get_logger, setup_logging
from handwash.pipelines.demo_audit import audit_demo_videos

log = get_logger(__name__)


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--folder", default="data/external/pskus_demo_validation", help="递归扫描的视频目录")
    parser.add_argument("--checkpoint", default="exp.pt", help="七类 Ultralytics 分类权重")
    parser.add_argument("--temporal-head", default="exp_temporal_head.pt", help="与 exp.pt 匹配的 GRU 权重")
    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)
    overrides = parse_overrides(override_items)
    overrides["runtime"] = {"run_name": "exp_demo", **overrides.get("runtime", {})}
    try:
        rc = load_config_from_args(args, overrides)
        summary = audit_demo_videos(
            rc, folder=args.folder, checkpoint=args.checkpoint, temporal_head=args.temporal_head
        )
    except HandwashError as exc:
        log.error("%s", exc)
        return getattr(exc, "exit_code", 1)
    print(f"本地核验完成：{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
