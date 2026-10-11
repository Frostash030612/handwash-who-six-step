#!/usr/bin/env python
"""Unattended, resumable full Zenodo download with bounded retries.

The underlying downloader verifies each archive's official MD5. This wrapper
re-runs it after transient network failures; existing verified files are
skipped and incomplete .part files resume. No extraction or training happens.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("pskuss", "jurmala", "metc"))
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--retry-delay", type=int, default=60)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.retry_delay < 0:
        parser.error("--max-attempts 必须 >=1，--retry-delay 必须 >=0")

    script = Path(__file__).with_name("download_data.py")
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n===== {args.dataset}: attempt {attempt}/{args.max_attempts} =====", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(script), "--dataset", args.dataset, "--all"],
            cwd=script.parent.parent,
            check=False,
        )
        if result.returncode == 0:
            print(f"{args.dataset}: all files passed MD5", flush=True)
            return 0
        if attempt < args.max_attempts:
            print(f"{args.dataset}: retry in {args.retry_delay}s", flush=True)
            time.sleep(args.retry_delay)
    print(f"{args.dataset}: still incomplete after {args.max_attempts} attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
