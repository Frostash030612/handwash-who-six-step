"""Expose this distribution's source tree to standalone scripts."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
