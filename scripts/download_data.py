#!/usr/bin/env python
"""从公开来源下载数据集：断点续传 + md5 校验 + 分片分配 + 来源记账。

**这个脚本存在的意义**：数据集有 2–17 GB，不能进 Git（也不能进 Git LFS，
理由与算式见 docs/DATA_COLLABORATION.md §5）。但每个数据集都在 Zenodo 上有
匿名直链且**每个文件都带 md5**，所以正确做法是：

    GitHub 存"怎么拿到数据"（本脚本 + data/raw/SOURCES.json + 划分），
    数据本身留在公开来源上，各人自行下载并用校验值证明一致。

四个人怎么分片：``--share i/4`` 会按**体积均衡**把分片分给第 i 个人，
分配结果是确定性的（同一份文件清单 -> 同一份分配），所以不需要开会协调，
四个人跑各自的命令就能覆盖全集且互不重复。

常用命令::

    # 看有什么、多大、哪些已经下好校验通过
    python scripts/download_data.py --dataset pskus --list

    # 只下小分片，先验证整条流水线（约 1 GB）
    python scripts/download_data.py --dataset pskus --files DataSet4.zip,DataSet3.zip

    # 按体积均衡分给 4 个人：第 1 个人跑 1/4，第 2 个人跑 2/4 ……
    python scripts/download_data.py --dataset pskus --share 1/4

    # 全量（约 17 GB，挂机跑）
    python scripts/download_data.py --dataset pskus --all

    # 校验已有的文件（不下东西，只核对 md5）
    python scripts/download_data.py --dataset pskus --verify-only

    # 解压已下好的分片
    python scripts/download_data.py --dataset pskus --extract

    # 看磁盘占用与剩余空间
    python scripts/download_data.py --status

设计约束（CONTRIBUTING.md R3 / R16）
    * 数据集登记（记录号、许可、校验值）在 ``src/handwash/data_sources.py``；
    * 落盘一律走 ``handwash.io.utils`` 与 ``handwash.paths``；
    * 数据目录来自 ``HANDWASH_DATA_ROOT``（默认 ``<repo>/data``），
      所以数据可以完全放在仓库外。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT  # noqa: E402,F401

from handwash.data_sources import (  # noqa: E402
    ZENODO_API,
    ZENODO_FILE_URL,
    all_dataset_names,
    get_dataset,
    iter_registered_files,
    requires_manual_download,
)

#: 每个数据集自己的来源清单文件名（**要提交进 Git**，见文档第 4 节）
SOURCES_FILENAME = "SOURCES.json"

#: 断点续传的临时后缀：下载完成后原子改名，避免半个文件被当成完整文件
PART_SUFFIX = ".part"

_USER_AGENT = "handwash-data-fetcher/1.0 (+educational group project)"
_CHUNK = 1024 * 256


# ===========================================================================
# 路径与磁盘
# ===========================================================================
def data_root() -> Path:
    """数据根目录：``HANDWASH_DATA_ROOT`` 优先，否则 ``<repo>/data``。"""
    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return PROJECT_ROOT / "data"


def raw_dir(dataset: str) -> Path:
    """原始数据的落地目录：``<data_root>/raw/<dataset>``。"""
    return data_root() / "raw" / dataset


def sources_path(dataset: str) -> Path:
    """来源清单路径。放在 data/raw/<dataset>/ 下，且已在 .gitignore 白名单里。"""
    return raw_dir(dataset) / SOURCES_FILENAME


def human(num_bytes: float) -> str:
    """人类可读体积。"""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def disk_report(path: Path) -> dict[str, float]:
    """返回目标盘的 ``(总, 已用, 可用)``（GB）。目录不存在时向上找最近的已存在祖先。"""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(str(probe))
    return {
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
    }


# ===========================================================================
# Zenodo 元数据
# ===========================================================================
def _http_get_json(url: str, *, timeout: int = 30) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_record_files(dataset: str, *, offline: bool = False) -> list[dict]:
    """查询 Zenodo，返回该记录的 ``[{key, size, md5}]``（按文件名排序）。

    ``offline=True`` 时改用本地登记表的估算体积，md5 留空
    （这样没有网络也能列出"该下哪些文件"并生成分片分配方案）。
    """
    entry = get_dataset(dataset)
    record = entry.get("record")
    if record is None:
        raise SystemExit(
            f"数据集 `{dataset}` 没有 Zenodo 记录号，无法脚本化下载。\n"
            f"手工下载说明：{entry.get('notes', '')}"
        )

    if offline:
        # 登记表的体积单位是**字节**（见 data_sources.py 的单位约定），直接用
        return [
            {"key": key, "size": size, "md5": md5}
            for key, size, md5 in iter_registered_files(dataset)
        ]

    try:
        payload = _http_get_json(ZENODO_API.format(record=record))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"查询 Zenodo 失败（{type(exc).__name__}: {exc}）。\n"
            "如果是网络受限，加 --offline 用本地登记表的分片清单做事，"
            "或先检查代理设置。"
        ) from exc

    files: list[dict] = []
    for item in payload.get("files", []):
        checksum = str(item.get("checksum", ""))
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else ""
        files.append({"key": item["key"], "size": int(item.get("size", 0)), "md5": md5})
    return sorted(files, key=lambda f: f["key"])


# ===========================================================================
# 分片分配（四人各下一份，合起来是全集）
# ===========================================================================
def shard_files(files: list[dict], *, always: tuple[str, ...]) -> list[dict]:
    """把"大文件"和"人人都要的小元数据文件"分开，只对大文件做分片。

    返回顺序为「大文件 + 元数据文件」，调用方据此决定怎么切。
    """
    meta = [f for f in files if f["key"] in always]
    payload = [f for f in files if f["key"] not in always]
    return payload + meta


def allocate_shares(
    files: list[dict],
    *,
    shares: int,
    always: tuple[str, ...] = (),
) -> dict[int, list[dict]]:
    """按**体积均衡**把大文件分给 ``shares`` 个人；小元数据文件人人都有。

    算法：按体积降序，每次把下一个文件给"当前总负载最小"的那个人
    （贪心的最长处理时间优先，LPT）。结果**确定性**：同样的文件清单必然给出
    同样的分配，因此四个人不需要开会协调，各自跑自己的份额即可覆盖全集。

    Returns
    -------
    ``{1: [文件...], 2: [...], ...}``（键从 1 开始，对应 ``--share 1/4``）
    """
    if shares < 1:
        raise SystemExit("--share 的分母必须 >= 1")
    meta = [f for f in files if f["key"] in always]
    payload = sorted(
        (f for f in files if f["key"] not in always),
        key=lambda f: (-f["size"], f["key"]),
    )

    buckets: dict[int, list[dict]] = {i: [] for i in range(1, shares + 1)}
    loads: dict[int, int] = {i: 0 for i in range(1, shares + 1)}
    for item in payload:
        target = min(loads, key=lambda i: (loads[i], i))  # 平票取下标小的，保证确定性
        buckets[target].append(item)
        loads[target] += item["size"]

    for index in buckets:
        buckets[index].extend(meta)
        buckets[index].sort(key=lambda f: f["key"])
    return buckets


def parse_share(text: str) -> tuple[int, int]:
    """解析 ``i/n`` 形式的份额参数。"""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not match:
        raise SystemExit(f"--share 格式应为 i/n（例如 1/4），实际 {text!r}")
    index, total = int(match.group(1)), int(match.group(2))
    if not 1 <= index <= total:
        raise SystemExit(f"--share 的 i 必须在 1..n 之间，实际 {index}/{total}")
    return index, total


# ===========================================================================
# 下载与校验
# ===========================================================================
def _md5_of(path: Path, *, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - 对齐 Zenodo 公布的校验算法，不用于安全用途
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_md5: str) -> tuple[bool, str]:
    """校验单个文件。``expected_md5`` 为空时只检查大小 > 0。"""
    if not path.exists():
        return False, "不存在"
    if path.stat().st_size == 0:
        return False, "零字节"
    if not expected_md5:
        return True, "未提供校验值，仅检查存在性"
    actual = _md5_of(path)
    if actual == expected_md5:
        return True, "md5 一致"
    return False, f"md5 不一致（期望 {expected_md5[:8]}…，实际 {actual[:8]}…）"


def download_file(
    dataset: str,
    item: dict,
    *,
    destination: Path,
    retries: int = 4,
    quiet: bool = False,
) -> tuple[bool, str]:
    """下载一个文件，支持断点续传；下完校验 md5。

    Returns
    -------
    ``(是否成功, 说明)``
    """
    target = destination / item["key"]
    part = target.with_suffix(target.suffix + PART_SUFFIX)
    expected_md5 = item.get("md5", "")

    # 已存在且校验通过 -> 直接跳过（重复运行零成本）
    if target.exists():
        ok, reason = verify_file(target, expected_md5)
        if ok:
            return True, f"已存在（{reason}），跳过"
        if not quiet:
            print(f"    已有文件校验失败：{reason}，将重新下载")

    entry = get_dataset(dataset)
    url = ZENODO_FILE_URL.format(record=entry["record"], name=item["key"])

    for attempt in range(1, retries + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": _USER_AGENT}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                # 206 = 服务器接受了 Range，继续追加；200 = 服务器忽略 Range，从头写
                mode = "ab" if resume_from and response.status == 206 else "wb"
                if mode == "wb" and resume_from:
                    resume_from = 0
                total = int(response.headers.get("Content-Length", 0)) + resume_from
                written = resume_from
                started = time.time()
                with part.open(mode) as handle:
                    while True:
                        block = response.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
                        if not quiet and total > 0:
                            elapsed = max(time.time() - started, 1e-6)
                            speed = (written - resume_from) / elapsed
                            done = written / total
                            bar = "#" * int(done * 24)
                            print(
                                f"\r    [{bar:<24}] {done*100:5.1f}%  "
                                f"{human(written)}/{human(total)}  {human(speed)}/s   ",
                                end="",
                                flush=True,
                            )
                if not quiet:
                    print()

            # 原子改名：只有完整下载的文件才会出现在最终路径上
            part.replace(target)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt >= retries:
                return False, f"下载失败（已重试 {retries} 次）：{type(exc).__name__}: {exc}"
            wait = min(2**attempt, 30)
            if not quiet:
                print(f"\n    第 {attempt} 次失败（{type(exc).__name__}），{wait}s 后续传重试")
            time.sleep(wait)
            continue

        ok, reason = verify_file(target, expected_md5)
        if ok:
            return True, reason
        # 校验失败的常见原因是传输损坏：删掉重来而不是无限续传同一个坏文件
        if not quiet:
            print(f"    校验失败：{reason}；删除后重试")
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    return False, "重试次数用尽"


# ===========================================================================
# 来源记账（SOURCES.json 要提交进 Git）
# ===========================================================================
def write_sources(
    dataset: str,
    *,
    record_url: str,
    files: list[dict],
    verified: dict[str, bool],
) -> Path:
    """写 ``data/raw/<dataset>/SOURCES.json``。

    这是**唯一需要提交进 Git 的数据相关文件**：它让报告能写清"哪份数据、
    哪个版本、何时获取、如何校验"，也让别人能证明自己拿到的是同一份数据。
    """
    entry = get_dataset(dataset)
    target = sources_path(dataset)
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    previous = existing.get("files", {})
    payload = {
        "dataset": dataset,
        "name": entry.get("name", dataset),
        "record": record_url,
        "home": entry.get("home", ""),
        "license": entry.get("license", ""),
        "paper": entry.get("paper", ""),
        "notes": entry.get("notes", ""),
        "generated_by": "scripts/download_data.py",
        "updated": date.today().isoformat(),
        "files": {
            item["key"]: {
                "size": item["size"],
                "md5": item.get("md5", ""),
                # 已在本机校验通过过就保持 true（同一份来源，md5 相同即等价）
                "verified": bool(verified.get(item["key"], previous.get(item["key"], {}).get("verified", False))),
                "retrieved": previous.get(item["key"], {}).get("retrieved")
                or (date.today().isoformat() if verified.get(item["key"]) else None),
            }
            for item in files
        },
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return target


def load_sources(dataset: str) -> dict:
    """读回来源清单（不存在则返回空 dict）。"""
    target = sources_path(dataset)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def local_verified(dataset: str) -> dict[str, bool]:
    """从已有的 SOURCES.json 恢复"本机已验证"状态。"""
    return {
        key: bool(info.get("verified"))
        for key, info in load_sources(dataset).get("files", {}).items()
    }


def scan_verified(dataset: str, files: list[dict]) -> dict[str, bool]:
    """扫描磁盘，返回每个文件当前的**实际**校验结果。

    SOURCES.json 必须反映磁盘真实状态。如果只依据"本轮下载循环的返回值"记账，
    那么"上次已经下好、这次被跳过"的文件会被误标成未验证 —— 这个 bug 真实发生过：
    下载成功后 SOURCES.json 里所有文件仍是 ``verified: false``。
    """
    destination = raw_dir(dataset)
    out: dict[str, bool] = {}
    for item in files:
        path = destination / item["key"]
        if not path.exists():
            out[item["key"]] = False
            continue
        ok, _ = verify_file(path, item.get("md5", ""))
        out[item["key"]] = ok
    return out


# ===========================================================================
# 子命令实现
# ===========================================================================
def cmd_list(dataset: str, *, offline: bool) -> int:
    files = fetch_record_files(dataset, offline=offline)
    entry = get_dataset(dataset)
    verified = local_verified(dataset)
    destination = raw_dir(dataset)

    print(f"数据集 : {dataset}  ({entry.get('name', '')})")
    print(f"来源   : {entry.get('home', '(无)')}")
    print(f"许可   : {entry.get('license', '(未知)')}")
    print(f"落地到 : {destination}")
    print(f"说明   : {entry.get('notes', '')}")
    print()
    total = sum(f["size"] for f in files)
    print(f"{'文件':<28}{'大小':>11}   {'本地状态':<22} md5")
    print("-" * 96)
    for item in files:
        path = destination / item["key"]
        if not path.exists():
            status = "未下载"
        else:
            ok, reason = verify_file(path, item.get("md5", ""))
            status = ("已验证 " if ok else "校验失败 ") + f"({human(path.stat().st_size)})"
            if not ok:
                status = f"校验失败：{reason[:16]}"
        md5 = (item.get("md5") or "(离线模式无)")[:12]
        print(f"{item['key']:<28}{human(item['size']):>11}   {status:<22} {md5}")
    print("-" * 96)
    print(f"合计 {len(files)} 个文件，{human(total)}")
    already = sum(
        1
        for item in files
        if (destination / item["key"]).exists()
        and verify_file(destination / item["key"], item.get("md5", ""))[0]
    )
    print(f"本机已完成校验：{already}/{len(files)}")
    disk = disk_report(destination)
    print(
        f"磁盘：可用 {disk['free_gb']:.1f} GB / 共 {disk['total_gb']:.1f} GB"
        f"（还需约 {human(max(0, total - sum(f['size'] for f in files if (destination / f['key']).exists())))}）"
    )
    return 0


def cmd_download(
    dataset: str,
    *,
    files: list[str] | None,
    share: str | None,
    take_all: bool,
    offline: bool,
    extract: bool,
    verify_only: bool,
) -> int:
    entry = get_dataset(dataset)
    if requires_manual_download(dataset):
        print(f"数据集 `{dataset}` 需要手工下载：\n  {entry.get('notes', '')}")
        return 2

    available = fetch_record_files(dataset, offline=offline)
    always = tuple(entry.get("always", ()))

    # --- 决定这一次要处理哪些文件 ----------------------------------------
    if share:
        index, total = parse_share(share)
        buckets = allocate_shares(available, shares=total, always=always)
        selected = buckets[index]
        print(f"分片方案：{total} 人分担（按体积均衡），你负责第 {index} 份")
        print(f"  你的清单：{', '.join(f['key'] for f in selected)}")
        print(f"  你的体积：{human(sum(f['size'] for f in selected))}")
        for other, bucket in buckets.items():
            mark = "  <- 你" if other == index else ""
            print(f"  第 {other} 份合计 {human(sum(f['size'] for f in bucket))}{mark}")
        print()
        print("分配是确定性的：别人跑 `--share 2/4` 等即可，合起来刚好是全集。")
        print()
    elif files:
        wanted = {name.strip() for name in files if name.strip()}
        known = {f["key"] for f in available}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(
                f"不认识的文件名：{unknown}\n可用：{sorted(known)}"
            )
        selected = [f for f in available if f["key"] in wanted]
    elif take_all:
        selected = available
    else:
        raise SystemExit(
            "请指定要下什么：\n"
            "  --all                 下载全部\n"
            "  --files A.zip,B.zip   只下指定文件\n"
            "  --share 1/4           按体积均衡只下第 1/4 份\n"
            "  --list                先看看有哪些文件"
        )

    destination = raw_dir(dataset)
    destination.mkdir(parents=True, exist_ok=True)

    # --- 空间预检（避免下到一半磁盘满）----------------------------------
    needed = sum(f["size"] for f in selected if not (destination / f["key"]).exists())
    disk = disk_report(destination)
    if needed > disk["free_gb"] * 1024**3:
        raise SystemExit(
            f"磁盘空间不足：需要约 {human(needed)}，可用 {disk['free_gb']:.1f} GB。\n"
            "建议：改用 --share 只下自己那份，或把 HANDWASH_DATA_ROOT 指向别的盘。"
        )
    print(f"目标目录：{destination}")
    print(f"本次下载：{len(selected)} 个文件，约 {human(needed)}（已下的会跳过）")
    print()

    verified = local_verified(dataset)
    failures: list[tuple[str, str]] = []
    for position, item in enumerate(selected, start=1):
        print(f"[{position}/{len(selected)}] {item['key']}  ({human(item['size'])})")
        if verify_only:
            ok, reason = verify_file(destination / item["key"], item.get("md5", ""))
            print(f"    校验：{reason}")
            verified[item["key"]] = ok
            if not ok:
                failures.append((item["key"], reason))
            continue

        ok, reason = download_file(dataset, item, destination=destination)
        print(f"    -> {reason}")
        if not ok:
            # 只有失败才覆盖已有状态：成功与否最终以磁盘上的实际校验结果为准
            verified[item["key"]] = False
            failures.append((item["key"], reason))

    # 记账以**磁盘实际状态**为准，而不是本轮循环的返回值 ——
    # 否则"上次下好了、这次被跳过"的文件会被误标为未验证（曾出现过这个 bug）。
    verified = scan_verified(dataset, available)
    record_url = entry.get("home") or ZENODO_API.format(record=entry.get("record"))
    path = write_sources(dataset, record_url=record_url, files=available, verified=verified)
    print()
    print(f"来源清单已更新：{path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    print("  这个文件很小，**请提交进 Git** —— 它让别人能核对你用的是同一份数据：")
    print(f"    git add {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")

    if failures:
        print(f"\n有 {len(failures)} 个文件未通过：")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        print("重新运行同样的命令即可续传（已校验通过的文件会被跳过）。")
        return 1

    print("\n全部完成并通过 md5 校验。")
    disk = disk_report(destination)
    print(f"当前磁盘：可用 {disk['free_gb']:.1f} GB")

    if extract:
        return cmd_extract(dataset)
    if not share:
        print("\n下一步：解压 -> 划分 -> 生成 manifest")
        print(f"  python scripts/download_data.py --dataset {dataset} --extract")
    return 0


def cmd_extract(dataset: str) -> int:
    """解压已校验通过的 zip 分片。"""
    destination = raw_dir(dataset)
    entry = get_dataset(dataset)
    extract_to = destination / str(entry.get("extract_to", dataset))
    extract_to.mkdir(parents=True, exist_ok=True)

    zips = sorted(destination.glob("*.zip"))
    if not zips:
        raise SystemExit(f"没有可解压的 zip：{destination}")

    for archive in zips:
        done_marker = archive.with_suffix(archive.suffix + ".extracted")
        if done_marker.exists():
            print(f"跳过（已解压过）：{archive.name}")
            continue
        print(f"解压 {archive.name} -> {extract_to}")
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extract_to)
        except zipfile.BadZipFile as exc:
            print(f"  解压失败（文件可能损坏，重新下载该分片）：{exc}")
            continue
        done_marker.write_text("ok\n", encoding="utf-8")
    print(f"\n解压完成：{extract_to}")
    print("下一步：")
    print(f"  python scripts/prepare_data.py --config configs/data/{dataset}.yaml")
    return 0


def cmd_allocate(dataset: str, *, shares: int, offline: bool, write: bool) -> int:
    """打印（并可写出）n 人分片方案，方便直接发到群里。"""
    available = fetch_record_files(dataset, offline=offline)
    always = tuple(get_dataset(dataset).get("always", ()))
    buckets = allocate_shares(available, shares=shares, always=always)

    lines = [
        f"# {dataset} 分片方案（{shares} 人，按体积均衡）",
        f"# 生成于 {date.today().isoformat()}；分配是确定性的，可重复生成",
        "",
    ]
    for index, bucket in buckets.items():
        size = human(sum(f["size"] for f in bucket))
        names = ",".join(f["key"] for f in bucket)
        lines.append(f"第 {index} 人（{size}）：--dataset {dataset} --files {names}")
    text = "\n".join(lines) + "\n"

    print(text)
    if write:
        plan = sources_path(dataset).parent / "SHARD_PLAN.md"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(text, encoding="utf-8", newline="\n")
        print(f"已写出：{plan}")
    return 0


def cmd_status(*, offline: bool = True) -> int:
    """总览：每个数据集下了多少、还差多少、磁盘还剩多少。"""
    print(f"数据根目录：{data_root()}")
    env = os.environ.get("HANDWASH_DATA_ROOT", "")
    print(f"HANDWASH_DATA_ROOT：{env if env else '(未设置，使用 <repo>/data)'}")
    print()
    print(f"{'数据集':<12}{'登记体积':>10}{'本地占用':>11}   {'状态'}")
    print("-" * 78)

    grand_total = 0
    for name in all_dataset_names():
        directory = raw_dir(name)
        used = 0
        if directory.exists():
            used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        grand_total += used

        if name == "synthetic" or requires_manual_download(name):
            state = "手工/自动生成，无需下载"
            registered = "—"
        else:
            try:
                files = fetch_record_files(name, offline=True)
                expected = sum(f["size"] for f in files)
                registered = human(expected)
                done = local_verified(name)
                verified_count = sum(1 for key in done if done.get(key))
                state = f"已校验 {verified_count}/{len(files)} 个文件"
            except SystemExit:
                registered, state = "—", "查询失败"
        print(f"{name:<12}{registered:>10}{human(used):>11}   {state}")

    print("-" * 78)
    disk = disk_report(data_root())
    print(f"合计本地占用：{human(grand_total)}")
    print(f"磁盘：可用 {disk['free_gb']:.1f} GB / 共 {disk['total_gb']:.1f} GB")
    print()
    print("提示：`--share i/4` 只下自己那份；`--list` 看单个数据集的明细。")
    return 0


# ===========================================================================
# 入口
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从公开来源下载数据集（断点续传 + md5 校验 + 分片分配 + 来源记账）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "四个人的推荐用法（PSKUS 17 GB，分片分摊）：\n"
            "  第 1 人：python scripts/download_data.py --dataset pskus --share 1/4 --extract\n"
            "  第 2 人：python scripts/download_data.py --dataset pskus --share 2/4 --extract\n"
            "  第 3 人：python scripts/download_data.py --dataset pskus --share 3/4 --extract\n"
            "  第 4 人：python scripts/download_data.py --dataset pskus --share 4/4 --extract\n"
            "\n"
            "METC 只有 1.98 GB，建议每人都下全量：\n"
            "  python scripts/download_data.py --dataset metc --all --extract\n"
            "\n"
            "数据放哪由 HANDWASH_DATA_ROOT 决定，可以完全放在仓库外：\n"
            '  $env:HANDWASH_DATA_ROOT = "D:\\handwash-data"\n'
        ),
    )
    parser.add_argument("--dataset", "-d", help=f"数据集名，可选：{all_dataset_names()}")
    parser.add_argument("--list", action="store_true", help="列出该数据集的文件与本地状态")
    parser.add_argument("--all", action="store_true", help="下载全部文件")
    parser.add_argument("--files", help="只下这些文件，逗号分隔（例如 DataSet1.zip,DataSet4.zip）")
    parser.add_argument("--share", help="按体积均衡只下第 i/n 份（例如 1/4）")
    parser.add_argument("--allocate", type=int, metavar="N", help="打印 N 人分片方案，不下载")
    parser.add_argument("--write-plan", action="store_true", help="配合 --allocate，把方案写成文件")
    parser.add_argument("--extract", action="store_true", help="下载后解压 zip 分片")
    parser.add_argument("--verify-only", action="store_true", help="只校验已有文件的 md5，不下载")
    parser.add_argument("--offline", action="store_true", help="不查 Zenodo，用本地登记表的分片清单")
    parser.add_argument("--status", action="store_true", help="总览所有数据集的本地占用")
    args = parser.parse_args(argv)

    if args.status:
        return cmd_status()

    if args.allocate:
        if not args.dataset:
            raise SystemExit("--allocate 需要同时指定 --dataset")
        return cmd_allocate(
            args.dataset, shares=args.allocate, offline=args.offline, write=args.write_plan
        )

    if not args.dataset:
        parser.print_help()
        print(f"\n可用数据集：{all_dataset_names()}")
        return 2

    if args.list:
        return cmd_list(args.dataset, offline=args.offline)

    files = [name for name in (args.files or "").split(",") if name.strip()]
    return cmd_download(
        args.dataset,
        files=files or None,
        share=args.share,
        take_all=args.all,
        offline=args.offline,
        extract=args.extract,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
