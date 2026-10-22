#!/usr/bin/env python3
"""Launch the frozen self-recorded camera application."""

import argparse
from pathlib import Path
import yaml
from _bootstrap import PROJECT_ROOT
from handwash.self_scene_camera import serve_self_scene_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT.parent / "runtime")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/experiments/self_scene_camera.yaml"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    minimum = float(config["assess"]["min_total_duration_s"])
    serve_self_scene_camera(args.runtime_dir, port=args.port, minimum_wash_seconds=minimum)


if __name__ == "__main__":
    main()
