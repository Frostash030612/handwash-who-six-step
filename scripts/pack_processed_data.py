#!/usr/bin/env python
"""把抽帧结果打包成一个压缩包：给网盘分享用。

**为什么需要这个脚本**

数据不能进 Git，Git LFS 也顶不住（见 docs/DATA_COLLABORATION.md §5）。
最省事的协作方式是：**一个人下载全量原始数据、抽一次帧，把抽帧结果打包传到网盘，
其他人下载同一个包**。本脚本就负责"打包"和"校验"这两步。

**为什么强烈建议先抽帧再打包**

    原始 zip 分片      17.1 GiB
    抽帧结果（实测）    见下表 —— 用 5 fps/256px 时只降到 11.6 GB，几乎没省；
                       换成 224px/q85 或 2 fps 才能真正降到 3-7 GB。

> 实测（PSKUS 原视频只有 320x240，所以"抽帧"并不天然变小）：
>     5 fps, 256px, q92 -> 全集约 11.6 GB
>     5 fps, 224px, q85 -> 约 7.1 GB
>     2 fps, 224px, q85 -> 约 2.8 GB
>
> 因为要传网盘，**务必先跑一次 `--report` 看体积**，再决定抽帧参数。
> 调整 fps 会改变时序模型能看到的动作细节，属于实验设计决策，要写进报告。

**打包方式的重要细节**

包内路径保持**相对数据集根目录**的形式（例如 ``frames/<clip_id>/00003.jpg``），
因为 ``manifest.csv`` 里存的就是这种相对路径。这样解包后立刻就能用，
不需要重新生成 manifest，也保证了所有人的划分完全一致。

常用命令::

    # 1) 先看体积，决定要不要降 fps
    python scripts/pack_processed_data.py --dataset pskuss --report

    # 2) 打包（默认 zip，Windows 双击可解压）
    python scripts/pack_processed_data.py --dataset pskuss --build

    # 3) 把生成的 .zip 和 .sha256 一起传到网盘，把链接发群里

    # 4) 队友下载后校验 + 解包（会自动放进自己的 data/processed/pskuss/）
    python scripts/pack_processed_data.py --dataset pskuss --verify <下载的包>
    python scripts/pack_processed_data.py --dataset pskuss --unpack <下载的包>

设计约束：落盘走 handwash.paths 与 handwash.io.utils，不硬编码目录（CONTRIBUTING.md R16）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401

#: 打进包里的相对路径白名单（相对数据集根目录）。
#: 包含可直接训练并能复现原始划分的产物：帧图像、manifest、划分契约与报告。
#: 刻意**不**包含原始视频与 zip —— 那些从公开来源下载更省流量。
INCLUDE_PATTERNS: tuple[str, ...] = (
    "frames",            # 抽帧结果（目录）
    "manifest.csv",
    "clip_splits.json",  # stage=frames 重跑时复用同一份视频级划分
    "split_report.json",
    "SOURCES.json",      # 来源与校验清单：让队友能证明拿到的是同一份数据
    "README.md",
)

#: 打包时跳过的临时/中间文件
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_dataset_name(dataset: str) -> str:
    """Keep dataset names as one portable path component."""
    if not _DATASET_NAME_RE.fullmatch(dataset):
        raise SystemExit(f"非法数据集名称：{dataset!r}（仅允许字母、数字、点、下划线和连字符）")
    return dataset


def data_root() -> Path:
    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return PROJECT_ROOT / "data"


def processed_dir(dataset: str) -> Path:
    """抽帧结果的落地目录：``<data_root>/processed/<dataset>``。"""
    validate_dataset_name(dataset)
    return data_root() / "processed" / dataset


def human(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def collect_entries(dataset: str) -> tuple[Path, list[Path]]:
    """收集要打包的文件，返回 ``(根目录, 相对路径列表)``。"""
    root = processed_dir(dataset)
    if not root.is_dir():
        raise SystemExit(
            f"找不到处理结果目录：{root}\n"
            "先抽帧：python scripts/prepare_data.py --config configs/data/"
            f"{dataset}.yaml"
        )

    entries: list[Path] = []
    for pattern in INCLUDE_PATTERNS:
        candidate = root / pattern
        if candidate.is_file():
            entries.append(candidate)
        elif candidate.is_dir():
            entries.extend(
                path
                for path in sorted(candidate.rglob("*"))
                if path.is_file() and path.name not in SKIP_NAMES
            )
    if not entries:
        raise SystemExit(
            f"{root} 下没有可打包的内容（期望 frames/、manifest.csv、clip_splits.json、split_report.json）。\n"
            "确认抽帧是否成功完成。"
        )
    return root, entries


def report(dataset: str) -> int:
    """按类别统计体积，帮助决定抽帧参数。"""
    root, entries = collect_entries(dataset)
    by_kind: dict[str, tuple[int, int]] = {}
    for path in entries:
        rel = path.relative_to(root)
        kind = rel.parts[0] if len(rel.parts) > 1 else "(顶层文件)"
        count, size = by_kind.get(kind, (0, 0))
        by_kind[kind] = (count + 1, size + path.stat().st_size)

    total_files = sum(c for c, _ in by_kind.values())
    total_size = sum(s for _, s in by_kind.values())
    print(f"数据集     : {dataset}")
    print(f"目录       : {root}")
    print(f"磁盘可用   : {shutil.disk_usage(str(root)).free / 1024**3:.1f} GB")
    print()
    print(f"{'内容':<22}{'文件数':>10}{'体积':>12}")
    print("-" * 46)
    for kind, (count, size) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"{kind:<22}{count:>10,}{human(size):>12}")
    print("-" * 46)
    print(f"{'合计':<22}{total_files:>10,}{human(total_size):>12}")
    print()
    print("=== PSKUS 全集抽帧体积（实测每帧字节数 × 官方 summary.csv 的真实时长）===")
    print("  数据集概况：3,185 段 / 139,881 秒（38.9 小时）/ 原生 30 fps / 平均每段 43.9 秒")
    print("  截断规则：split.max_frames_per_clip_train=300, _eval=150（等间隔，非取前 N 帧）")
    print()
    print(f"  {'抽帧配置':<24}{'全集体积':>12}")
    print("  " + "-" * 36)
    print(f"  {'5 fps + 256px + q92':<24}{'约 10.4 GB':>12}")
    print(f"  {'5 fps + 224px + q85（旧默认）':<24}{'约  6.4 GB':>12}")
    print(f"  {'2 fps + 224px + q85（推荐）':<24}{'约  2.8 GB':>12}")
    print(f"  {'1 fps + 224px + q85':<24}{'约  1.4 GB':>12}")
    print()
    print("  为什么推荐 2 fps：WHO 六步每步通常持续 5-15 秒，2 fps 即每步 10-30 帧，")
    print("  足够时序模型建模；而 5 fps 会把网盘传输量翻一倍以上。")
    print("  若后续要做「某一步时长明显不足」的细粒度判定，再补一组 5 fps 的对比实验。")
    print()
    print("  压缩包体积会接近上表数值（jpg 已是压缩格式，zip 几乎不会再缩小）。")
    return 0


def _sha256(path: Path, *, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build(dataset: str, *, out_dir: Path, fmt: str) -> int:
    """打包并写校验文件。"""
    root, entries = collect_entries(dataset)
    total_size = sum(path.stat().st_size for path in entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    archive = out_dir / f"{dataset}_frames_{stamp}.{fmt}"

    print(f"打包 {len(entries):,} 个文件，约 {human(total_size)}")
    print(f"输出：{archive}")
    started = time.time()

    if fmt == "zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
            for index, path in enumerate(entries, start=1):
                handle.write(path, arcname=str(path.relative_to(root)).replace("\\", "/"))
                if index % 2000 == 0 or index == len(entries):
                    done = index / len(entries)
                    print(f"\r  [{done*100:5.1f}%] {index:,}/{len(entries):,}", end="", flush=True)
    elif fmt == "tar.gz":
        with tarfile.open(archive, "w:gz") as handle:
            for index, path in enumerate(entries, start=1):
                handle.add(path, arcname=str(path.relative_to(root)).replace("\\", "/"))
                if index % 2000 == 0 or index == len(entries):
                    done = index / len(entries)
                    print(f"\r  [{done*100:5.1f}%] {index:,}/{len(entries):,}", end="", flush=True)
    else:
        raise SystemExit(f"不支持的格式：{fmt}（可用 zip / tar.gz）")
    print()

    elapsed = time.time() - started
    size = archive.stat().st_size
    print(f"完成：{human(size)}，用时 {elapsed/60:.1f} 分钟")
    print(f"压缩率：{size/total_size*100:.1f}%（jpg 已是压缩格式，zip 不会再变小）")

    digest = _sha256(archive)
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    checksum_file.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")

    meta = {
        "dataset": dataset,
        "archive": archive.name,
        "sha256": digest,
        "size_bytes": size,
        "num_files": len(entries),
        "created": stamp,
        "included": list(INCLUDE_PATTERNS),
        "note": (
            "包内路径相对数据集根目录，与 manifest.csv 里的相对路径一致；"
            "解包到 <data_root>/processed/<dataset>/ 即可直接训练。"
        ),
    }
    meta_file = out_dir / f"{dataset}_package.json"
    meta_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print()
    print("请把这三个文件一起传到网盘，并把链接发到群里：")
    print(f"  {archive.name}          ← 数据包（{human(size)}）")
    print(f"  {checksum_file.name}    ← sha256 校验值")
    print(f"  {meta_file.name}        ← 包说明（含 sha256、文件数、生成日期）")
    print()
    print("队友下载后运行（会先校验再解包）：")
    print(f"  python scripts/pack_processed_data.py --dataset {dataset} --verify {archive.name}")
    print(f"  python scripts/pack_processed_data.py --dataset {dataset} --unpack {archive.name}")
    return 0


def _expected_sha256(archive: Path) -> str:
    """从匹配当前包的 checksum sidecar 或 package metadata 读取期望值。"""
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    if checksum_file.exists():
        line = checksum_file.read_text(encoding="utf-8").strip()
        digest, separator, referenced_name = line.partition("  ")
        if not separator:
            fields = line.split(maxsplit=1)
            digest = fields[0] if fields else ""
            referenced_name = fields[1] if len(fields) > 1 else ""
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise SystemExit(f"SHA-256 sidecar 格式无效：{checksum_file}")
        if referenced_name and referenced_name.lstrip("*") != archive.name:
            raise SystemExit(
                f"SHA-256 sidecar 指向 {referenced_name!r}，与当前压缩包 {archive.name!r} 不匹配"
            )
        return digest.lower()
    # 也接受同目录下的 <dataset>_package.json
    meta = archive.parent / f"{archive.name.split('_frames_')[0]}_package.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SystemExit(f"package metadata 无法读取：{meta}（{exc}）") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"package metadata 顶层必须是 JSON 对象：{meta}")
        # 多次打包会更新同一个 metadata 文件；旧压缩包不能误用新包的 hash。
        if payload.get("archive") != archive.name:
            return ""
        digest = payload.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise SystemExit(f"package metadata 中的 SHA-256 无效：{meta}")
        return digest.lower()
    return ""


def resolve_archive(raw: str | Path) -> Path:
    """把一个"包参数"解析成真实路径。

    允许三种写法，减少队友敲错路径的概率（这是实际协作里最容易出问题的一步）：
        * 绝对路径：传入压缩包在当前机器上的实际位置
        * 相对路径：``data/packages/pskuss_frames_2026-09-22.zip``
        * **裸文件名**：``pskuss_frames_2026-09-22.zip``
          -> 自动在 ``<data_root>/packages/`` 与当前目录下查找
        * 通配：``pskuss_frames_*.zip``（取匹配到的最后一个，即最新日期）
    """
    candidate = Path(raw)
    if candidate.exists():
        return candidate

    search_dirs = [data_root() / "packages", Path.cwd(), PROJECT_ROOT]
    for directory in search_dirs:
        found = directory / str(raw)
        if found.exists():
            return found
        if any(ch in str(raw) for ch in "*?["):
            matches = sorted(directory.glob(str(raw)))
            if matches:
                return matches[-1]

    searched = "\n".join(f"    {d}" for d in search_dirs)
    packages = data_root() / "packages"
    listing = ""
    if packages.is_dir():
        names = sorted(p.name for p in packages.glob("*.*"))
        if names:
            listing = "\n该目录下现有的包：\n" + "\n".join(f"    {n}" for n in names)
    raise SystemExit(
        f"找不到数据包：{raw}\n"
        f"已在这些位置查找：\n{searched}{listing}\n\n"
        "可以给出完整路径，或只给文件名（脚本会自动在 data/packages/ 下找）。"
    )


def verify(dataset: str, archive: Path) -> int:
    """校验包的 sha256 并列出内容摘要。"""
    archive = resolve_archive(archive)
    if not archive.exists():
        raise SystemExit(f"找不到包：{archive}")
    print(f"校验 {archive.name}（{human(archive.stat().st_size)}）")
    print("计算 sha256 中……")
    started = time.time()
    actual = _sha256(archive)
    print(f"  sha256  = {actual}   用时 {time.time()-started:.1f}s")

    expected = _expected_sha256(archive)
    if expected:
        if actual == expected:
            print(f"  期望值  = {expected}")
            print("  ✅ 一致，包完整。")
        else:
            print(f"  期望值  = {expected}")
            print("  ❌ 不一致！下载可能损坏，请重新下载。")
            return 1
    else:
        print("  （未找到 .sha256 或 _package.json，无法自动比对；")
        print("   请把上面对话里的 sha256 与群里的值手工核对。）")

    # 内容摘要
    print()
    print("包内容：")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
    else:
        with tarfile.open(archive) as handle:
            names = handle.getnames()
    tops: dict[str, int] = {}
    for name in names:
        top = name.split("/")[0] if "/" in name else name
        tops[top] = tops.get(top, 0) + 1
    for top, count in sorted(tops.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {top:<24}{count:>8,} 项")
    print(f"  合计 {len(names):,} 项")
    return 0


def unpack(dataset: str, archive: Path, *, into: Path | None = None) -> int:
    """解包到 ``<data_root>/processed/<dataset>/``（已存在同名文件时覆盖）。

    刻意**先解到临时目录再合并**。解包校验可用的 SHA-256 sidecar，拒绝
    路径越界、符号链接和特殊文件；合并时也拒绝目标目录中的链接。
    """
    archive = resolve_archive(archive)
    if not archive.exists():
        raise SystemExit(f"找不到包：{archive}")
    target = (into or processed_dir(dataset)).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset}_unpacking_", dir=target.parent))

    print(f"解包 {archive.name} -> {target}")
    started = time.time()
    expected = _expected_sha256(archive)
    if expected:
        actual = _sha256(archive)
        if actual != expected:
            shutil.rmtree(staging, ignore_errors=True)
            raise SystemExit(f"包校验失败：期望 {expected}，实际 {actual}；未解包。")
        print("SHA-256 校验通过。")
    else:
        print("⚠️ 未找到 SHA-256 sidecar；继续解包前请先手工核对发布者提供的校验值。")
    try:
        _extract_archive_safely(archive, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"解包完成，用时 {(time.time()-started)/60:.1f} 分钟")

    # 合并到目标目录
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    try:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(staging)
            destination = target / rel
            parent = target
            for component in rel.parts[:-1]:
                parent = parent / component
                if parent.is_symlink():
                    raise ValueError(f"目標目錄包含符号链接，拒绝写入：{parent}")
                parent.mkdir(exist_ok=True)
                if not parent.is_dir():
                    raise ValueError(f"目标路径的父项不是目录：{parent}")
            if destination.is_symlink():
                raise ValueError(f"目标文件是符号链接，拒绝覆盖：{destination}")
            path.replace(destination)
            moved += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"已合并 {moved:,} 个文件到 {target}")

    manifest = target / "manifest.csv"
    if manifest.exists():
        rows = sum(1 for _ in manifest.open(encoding="utf-8")) - 1
        print()
        print(f"manifest.csv 存在，约 {rows:,} 行 —— 划分已随包带来，无需重新生成。")
    else:
        print()
        print("⚠️  包里没有 manifest.csv。请让打包的人确认抽帧与划分都已完成。")
    print()
    print("下一步：")
    print(f"  python scripts/train_model.py --config configs/experiments/exp02_yolo26n_gru.yaml")
    print("  （dataset.root 需要在 configs/data/ 里指向抽帧目录，或用 HANDWASH_DATA_ROOT）")
    return 0


def _archive_destination(staging: Path, raw_name: str, seen: set[str]) -> Path:
    """Validate a portable relative member path before writing it to disk."""
    if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
        raise ValueError(f"压缩包内路径不是安全的相对 POSIX 路径：{raw_name!r}")
    member = PurePosixPath(raw_name)
    parts = member.parts
    windows_reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    invalid_component = any(
        ":" in part
        or "\x00" in part
        or part.endswith((".", " "))
        or part.rstrip(" .").split(".", 1)[0].upper() in windows_reserved
        for part in parts
    )
    if (
        not parts
        or any(part in ("", ".", "..") for part in parts)
        or invalid_component
        or parts[0] not in {"frames", "manifest.csv", "clip_splits.json", "split_report.json", "SOURCES.json", "README.md"}
    ):
        raise ValueError(f"压缩包内路径超出数据包白名单：{raw_name!r}")
    normalized = member.as_posix().casefold()
    if normalized in seen:
        raise ValueError(f"压缩包包含重复或仅大小写不同的路径：{raw_name!r}")
    seen.add(normalized)
    destination = staging.joinpath(*parts)
    if not destination.resolve().is_relative_to(staging.resolve()):
        raise ValueError(f"压缩包内路径试图越出解包目录：{raw_name!r}")
    return destination


def _extract_archive_safely(archive: Path, staging: Path) -> None:
    """Extract only regular files/directories from the package allowlist."""
    seen: set[str] = set()
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                destination = _archive_destination(staging, info.filename.rstrip("/"), seen)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if (
                    stat.S_ISLNK(mode)
                    or file_type not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (file_type == stat.S_IFDIR and not info.is_dir())
                ):
                    raise ValueError(f"ZIP 包含不允许的链接或特殊文件：{info.filename!r}")
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(info) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
        return

    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            destination = _archive_destination(staging, member.name.rstrip("/"), seen)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"TAR 包含不允许的链接或特殊文件：{member.name!r}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"无法读取压缩包成员：{member.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把抽帧结果打包/校验/解包，用于通过网盘分享数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "推荐流程（一个人做前三步，把链接发群里）：\n"
            "  python scripts/pack_processed_data.py --dataset pskuss --report\n"
            "  python scripts/pack_processed_data.py --dataset pskuss --build\n"
            "  # 上传 .zip / .sha256 / _package.json 到网盘\n"
            "队友拿到后：\n"
            "  python scripts/pack_processed_data.py --dataset pskuss --verify <包>\n"
            "  python scripts/pack_processed_data.py --dataset pskuss --unpack <包>\n"
        ),
    )
    parser.add_argument("--dataset", "-d", default="pskuss", help="数据集名（默认 pskuss）")
    parser.add_argument("--report", action="store_true", help="只看体积，不打包")
    parser.add_argument("--build", action="store_true", help="打包")
    parser.add_argument("--verify", metavar="ARCHIVE", help="校验一个包（sha256 + 内容摘要）")
    parser.add_argument("--unpack", metavar="ARCHIVE", help="解包到 data/processed/<dataset>/")
    parser.add_argument(
        "--format", default="zip", choices=("zip", "tar.gz"), help="包格式（默认 zip）"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="**仅用于 --build**：包的输出目录（默认 <data_root>/packages）",
    )
    parser.add_argument(
        "--into",
        default=None,
        help="**仅用于 --unpack**：解包目标目录（默认 <data_root>/processed/<dataset>）",
    )
    args = parser.parse_args(argv)
    validate_dataset_name(args.dataset)

    out_dir = Path(args.out_dir) if args.out_dir else data_root() / "packages"

    if args.report:
        return report(args.dataset)
    if args.build:
        if args.into:
            parser.error("--into 只对 --unpack 有效；打包请用 --out-dir")
        return build(args.dataset, out_dir=out_dir, fmt=args.format)
    if args.verify:
        return verify(args.dataset, Path(args.verify))
    if args.unpack:
        if args.out_dir:
            parser.error("--out-dir 只对 --build 有效；解包请用 --into")
        return unpack(args.dataset, Path(args.unpack), into=Path(args.into) if args.into else None)

    parser.print_help()
    print("\n提示：先跑 --report 看体积，再决定是否调整 configs 里的 fps / resize_hw。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
