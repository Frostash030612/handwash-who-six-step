#!/usr/bin/env python
"""Launch the self-scene camera dashboard; weights remain in the local runtime directory."""

import sys

from _bootstrap import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli(["self-scene-camera", *sys.argv[1:]]))
