#!/usr/bin/env python
"""Run the local full-video comparison page using the archived models."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: F401
from _common import build_config_parser, load_config_from_args
from handwash.compare_app import serve_compare


def main() -> int:
    parser = build_config_parser(__doc__ or "")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    cfg = load_config_from_args(args, {"runtime": {"run_name": "model_compare"}})
    serve_compare(cfg, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
