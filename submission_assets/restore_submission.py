"""Reconstruct and verify the single Canvas submission ZIP from Git LFS parts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ASSETS = Path(__file__).resolve().parent
REPOSITORY = ASSETS.parent
MANIFEST = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
OUTPUT = REPOSITORY / "deliverables" / MANIFEST["filename"]
BUFFER_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and OUTPUT.stat().st_size == MANIFEST["bytes"]:
        if sha256(OUTPUT) == MANIFEST["sha256"]:
            print(f"Verified existing archive: {OUTPUT}")
            return

    temporary = OUTPUT.with_suffix(".rebuilding.zip")
    whole = hashlib.sha256()
    try:
        with temporary.open("wb") as destination:
            for part in MANIFEST["parts"]:
                source = ASSETS / part["file"]
                if not source.is_file() or source.stat().st_size != part["bytes"]:
                    raise RuntimeError(f"Missing or incomplete LFS part: {source}")
                current = hashlib.sha256()
                with source.open("rb") as stream:
                    while chunk := stream.read(BUFFER_SIZE):
                        destination.write(chunk)
                        current.update(chunk)
                        whole.update(chunk)
                if current.hexdigest() != part["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: {source}")
                print(f"Verified {source.name}")
        if temporary.stat().st_size != MANIFEST["bytes"] or whole.hexdigest() != MANIFEST["sha256"]:
            raise RuntimeError("Reconstructed archive failed SHA-256 verification")
        os.replace(temporary, OUTPUT)
        print(f"Ready for Canvas submission: {OUTPUT}")
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
