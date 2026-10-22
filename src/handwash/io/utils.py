"""JSON, YAML and CSV serialization helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from handwash.errors import DataError
from handwash.paths import ensure_dir

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "append_jsonl",
    "file_size_mb",
    "human_size",
    "list_files",
    "list_videos",
    "read_csv",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "write_csv",
    "write_json",
    "write_jsonl",
    "write_text",
    "write_yaml",
]

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================================


# ============================================================================
def _atomic_write_text(path: Path, content: str) -> Path:

    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


# ============================================================================
# JSON / JSONL
# ============================================================================
def read_json(path: str | Path, *, default: Any = None) -> Any:

    target = Path(path)
    if not target.exists():
        if default is not None:
            return default
        raise DataError(f"JSON file not found: {target}")
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataError(
            f"JSON parsing failed: {target}(row {exc.lineno} rows)",
            hint="The file may be incomplete; regenerate it from the corresponding source step.",
        ) from exc


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:

    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=False)
    return _atomic_write_text(Path(path), text + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:

    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return _atomic_write_text(Path(path), content)


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:

    target = Path(path)
    if not target.exists():
        raise DataError(f"JSONL file not found: {target}")
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise DataError(f"JSONL line {lineno} parse failed: {target} ({exc})") from exc
    return rows


# ============================================================================
# CSV
# ============================================================================
def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:

    target = Path(path)
    ensure_dir(target.parent)
    if fieldnames is None:
        if not rows:
            raise DataError("write_csv requires explicit fieldnames when rows is empty")
        fieldnames = list(rows[0].keys())

    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_csv(path: str | Path) -> list[dict[str, str]]:

    target = Path(path)
    if not target.exists():
        raise DataError(f"CSV file not found: {target}")
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ============================================================================
# YAML
# ============================================================================
def read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise DataError(f"YAML file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded or {})


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    text = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False, default_flow_style=False)
    return _atomic_write_text(Path(path), text)


def write_text(path: str | Path, content: str) -> Path:

    return _atomic_write_text(Path(path), content)


# ============================================================================


# ============================================================================
def list_files(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    recursive: bool = True,
) -> list[Path]:

    base = Path(root)
    if not base.exists():
        return []
    suffixes = {e.lower() for e in extensions} if extensions else None
    iterator = base.rglob("*") if recursive else base.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes) and not p.name.startswith(".")
    ]
    return sorted(files)


def list_videos(root: str | Path, *, recursive: bool = True) -> list[Path]:

    return list_files(root, extensions=VIDEO_EXTENSIONS, recursive=recursive)


def file_size_mb(path: str | Path) -> float:
    target = Path(path)
    if not target.exists():
        return 0.0
    if target.is_file():
        return target.stat().st_size / (1024 * 1024)
    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def human_size(num_bytes: float) -> str:

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"
