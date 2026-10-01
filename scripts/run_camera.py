#!/usr/bin/env python
"""启动外接摄像头网页：python scripts/run_camera.py config=configs/experiments/live_yolo_frame.yaml。"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: F401
from _common import build_config_parser, load_config_from_args
from handwash.camera_app import serve_camera
from handwash.cli import split_argv
from handwash.core.config import parse_overrides
from handwash.errors import HandwashError
from handwash.logging import setup_logging


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--checkpoint", default=None, help="权重路径；正式版默认 best.pt，--demo-exp 默认根目录 exp.pt")
    parser.add_argument("--port", type=int, default=8765, help="本机网页端口")
    parser.add_argument("--demo-exp", action="store_true", help="使用项目根目录的 exp.pt 七类演示模型")
    plain, override_items = split_argv(sys.argv[1:])
    args, unknown = parser.parse_known_args(plain)
    if unknown:
        parser.error(f"无法识别的参数：{unknown}")
    setup_logging(args.log_level, force=True)
    overrides = parse_overrides(override_items)
    if args.demo_exp:
        overrides["runtime"] = {"run_name": "exp_demo", **overrides.get("runtime", {})}
    rc = load_config_from_args(args, overrides)
    try:
        serve_camera(rc, checkpoint=args.checkpoint, port=args.port, demo_exp=args.demo_exp)
    except HandwashError as exc:
        print(f"摄像头启动失败：{exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    except OSError as exc:
        print(f"摄像头服务无法监听本机端口 {args.port}：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
