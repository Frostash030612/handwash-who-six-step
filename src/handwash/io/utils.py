"""通用文件读写工具：JSON / JSONL / CSV / YAML，全部原子写。

规则（CONTRIBUTING.md R16）
    * 写文件一律走这里，保证：UTF-8、LF 换行、父目录自动创建、**原子替换**。
      原子写很关键：实验跑到一半断电，也不会留下半个 JSON 让下游解析崩溃。
    * 禁止在业务代码里 ``open(..., "w")``。
"""

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
    "read_json",
    "write_json",
    "append_jsonl",
    "read_jsonl",
    "write_csv",
    "read_csv",
    "read_yaml",
    "write_yaml",
    "list_files",
    "list_videos",
    "file_size_mb",
    "human_size",
    "VIDEO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
]

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================================
# 原子写
# ============================================================================
def _atomic_write_text(path: Path, content: str) -> Path:
    """原子写文本：先写同目录临时文件，再 ``os.replace``。"""
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        # 失败时清理临时文件，避免 data/ 里堆积垃圾
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


# ============================================================================
# JSON / JSONL
# ============================================================================
def read_json(path: str | Path, *, default: Any = None) -> Any:
    """读 JSON。``default`` 不为 None 时，文件不存在返回 default 而不是报错。"""
    target = Path(path)
    if not target.exists():
        if default is not None:
            return default
        raise DataError(f"JSON 文件不存在：{target}")
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataError(
            f"JSON 解析失败：{target}（第 {exc.lineno} 行）",
            hint="文件可能写了一半就被中断；删掉后重跑对应步骤。",
        ) from exc


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """写 JSON（UTF-8、保留中文、末尾换行）。"""
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=False)
    return _atomic_write_text(Path(path), text + "\n")


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """追加式写 JSONL（训练日志/逐帧预测用：崩溃也不会丢掉已完成部分）。"""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL，跳过空行；坏行报错并指出行号。"""
    target = Path(path)
    if not target.exists():
        raise DataError(f"JSONL 文件不存在：{target}")
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise DataError(f"JSONL 第 {lineno} 行解析失败：{target}（{exc}）") from exc
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
    """写 CSV。列顺序显式指定，避免不同人跑出来的 manifests 列序不一致。"""
    target = Path(path)
    ensure_dir(target.parent)
    if fieldnames is None:
        if not rows:
            raise DataError("write_csv 在 rows 为空时必须显式提供 fieldnames")
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
    """读 CSV 为字典列表（空字符串保留，类型转换交给调用方）。"""
    target = Path(path)
    if not target.exists():
        raise DataError(f"CSV 文件不存在：{target}")
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ============================================================================
# YAML
# ============================================================================
def read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise DataError(f"YAML 文件不存在：{target}")
    with target.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded or {})


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    text = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False, default_flow_style=False)
    return _atomic_write_text(Path(path), text)


# ============================================================================
# 目录扫描
# ============================================================================
def list_files(
    root: str | Path,
    *,
    extensions: Sequence[str] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """列出目录下的文件（已排序，保证跨平台结果一致）。"""
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
    """列出目录下所有视频文件。"""
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
    """人类可读体积（日志里报告数据集大小用）。"""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"
