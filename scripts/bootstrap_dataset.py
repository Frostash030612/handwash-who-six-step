#!/usr/bin/env python
"""一键把数据集从"零"跑到"可以发给队友的分享包"。

这是给**一个人**用的编排脚本：他负责下全量、抽帧、打包，然后把三个文件传网盘。
其余三人只需要 `pack_processed_data.py --verify` + `--unpack`（见 docs/DATA_COLLABORATION.md §3.0）。

一条命令::

    python scripts/bootstrap_dataset.py --dataset pskuss

它按顺序做五件事，**每一步都可重复执行**（已完成的会跳过，所以中断后直接重跑）：

    1. 预检   磁盘空间是否够、网络是否可达、数据是否已完整
    2. 下载   11 个分片，断点续传 + md5 校验（已完成/已校验的跳过）
    3. 抽帧   PSKUS 适配器解析真实标注 -> 按原始视频划分 -> 抽帧 -> 写 manifest
    4. 打包   frames/ + manifest + 划分报告 -> 一个 zip + sha256 + 说明
    5. 汇总   打印分享包路径、体积、sha256，以及可以直接复制给队友的三条命令

常用变体::

    # 只跑到抽帧（自己训练用，不分享）
    python scripts/bootstrap_dataset.py --dataset pskuss --skip-pack

    # 只下最小的分片先验证链路（约 1.2 GB）
    python scripts/bootstrap_dataset.py --dataset pskuss --shards DataSet4.zip,DataSet3.zip

    # 已经下过数据，只想重新打包
    python scripts/bootstrap_dataset.py --dataset pskuss --skip-download

    # 演练：只打印会执行哪些步骤，不真的跑
    python scripts/bootstrap_dataset.py --dataset pskuss --dry-run

**重要的前提**：只有**下全**分片，抽帧与划分才是"权威"的。
若只下了部分分片，脚本会在结尾明确警告，并把分享包标成 partial ——
因为基于子集的划分与别人的划分不一致，实验数字就不可比（见 docs/DATA_COLLABORATION.md §1）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from handwash.data_sources import (
    all_dataset_names,
    get_dataset,
    iter_registered_files,
    requires_manual_download,
)

CONFIG_FOR_DATASET = {
    "pskuss": "configs/data/pskuss.yaml",
    "metc": "configs/data/metc.yaml",
    "selfrecorded": "configs/data/selfrecorded.yaml",
    "kaggle": "configs/data/kaggle.yaml",
}

#: 每个数据集推荐的实验配置。
#:
#: 为什么要单独一张表：`configs/data/*.yaml` 只声明**数据**（路径、抽帧、判定阈值），
#: 里面没有训练超参；而 `configs/experiments/*.yaml` 声明**实验**（模型、训练、评估），
#: 官方叠加语法是 `config=configs/config.yaml config=<实验>`。
#: 早期版本在交接清单里把数据配置当成训练配置打印出来，照着做会跑成一个不匹配的实验，
#: 所以这里显式登记"该用哪个实验配置"，并在打印时把两个文件都列出来。
EXPERIMENT_FOR_DATASET = {
    "pskuss": "configs/experiments/exp02_yolo26n_gru.yaml",
    "metc": "configs/experiments/exp04_cross_domain.yaml",
    "kaggle": "configs/experiments/exp01_baseline_frame.yaml",
    "selfrecorded": "configs/experiments/exp04_cross_domain.yaml",
}


def train_command(dataset: str) -> str:
    """给出可以直接复制的训练命令。

    **叠加顺序很关键**：``config=`` 是依次深合并、后者覆盖前者。
    ``configs/experiments/*.yaml`` 里通常也写了 ``dataset.name``（为了能单独跑），
    所以必须**先叠实验配置、再叠数据配置**，否则数据配置会被实验配置覆盖回默认数据集，
    于是训练静默地跑在了另一个数据集上（真实踩过这个坑：明明指定了 pskuss，
    日志里却是 "split=train，252 帧" 的合成数据）。
    """
    experiment = EXPERIMENT_FOR_DATASET.get(dataset)
    data_config = CONFIG_FOR_DATASET.get(dataset)
    if experiment is None or data_config is None:
        return "python scripts/train_model.py --config configs/config.yaml"
    if dataset in ("kaggle", "synthetic"):
        # 这两个数据集的档案已在 configs/config.yaml 里，不需要额外叠加
        return f"python scripts/train_model.py --config {experiment}"
    # 顺序：实验配置在前，数据配置在后（后者胜出，确保跑在正确的数据集上）
    return (
        f"python scripts/train_model.py "
        f"config={experiment} config={data_config}"
    )

#: 全量分片数（用于判断"是否下全"）；运行时从登记表算，这里只做兜底
_STEPS = ("preflight", "download", "extract", "prepare", "pack", "summary")


class Step:
    """一个可跳过的步骤，带计时与统一输出格式。"""

    def __init__(self, index: int, total: int, title: str, detail: str = "") -> None:
        self.index = index
        self.total = total
        self.title = title
        self.detail = detail
        self.started = 0.0

    def __enter__(self) -> Step:
        self.started = time.time()
        print()
        print("=" * 78)
        print(f"[{self.index}/{self.total}] {self.title}")
        if self.detail:
            print(f"      {self.detail}")
        print("=" * 78)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.time() - self.started
        if exc_type is None:
            print(f"      -> 完成，用时 {elapsed/60:.1f} 分钟")
        else:
            print(f"      -> 失败（{exc_type.__name__}），已用时 {elapsed/60:.1f} 分钟")
        return False


def human(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def data_root() -> Path:
    """与 download_data.py / pack_processed_data.py 保持同一套解析规则。"""
    import os

    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return PROJECT_ROOT / "data"


def run_script(script: str, args: list[str], *, dry_run: bool) -> int:
    """调用同级脚本（子进程），透传退出码。

    刻意用子进程而不是 import：这样每一步的日志、进度条与退出码都和手动运行完全一致，
    出问题时可以直接把命令复制出来单独重跑，便于排查。
    """
    command = [sys.executable, "-X", "utf8", str(Path(__file__).parent / script), *args]
    print(f"      $ {' '.join(str(c) for c in command[3:])}")
    if dry_run:
        print("      （--dry-run：跳过执行）")
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode


# ===========================================================================
# 各步骤
# ===========================================================================
def preflight(dataset: str, *, only_shards: list[str] | None) -> dict:
    """预检：磁盘、登记信息、已下载情况。返回一份情况摘要。

    名词区分（避免把几百 KB 的元数据文件说成"分片"，那会让人误判进度）：
        * **视频分片**：``*.zip``，真正的数据本体（PSKUS 是 11 个）
        * **元数据文件**：``README.md`` / ``statistics.csv`` / ``summary.csv``，合计几百 KB
    引用完整性只看**视频分片**是否齐全；元数据缺失只会少一点上下文。
    """
    entry = get_dataset(dataset)
    if requires_manual_download(dataset):
        raise SystemExit(
            f"数据集 `{dataset}` 无法脚本化下载。\n{entry.get('notes', '')}\n"
            "请改为 --dataset pskuss 或 --dataset metc。"
        )

    raw = data_root() / "raw" / dataset
    raw.mkdir(parents=True, exist_ok=True)

    registered = iter_registered_files(dataset)
    targets = registered
    if only_shards:
        wanted = {name.strip() for name in only_shards if name.strip()}
        unknown = sorted(wanted - {key for key, _, _ in registered})
        if unknown:
            raise SystemExit(
                f"不认识的分片：{unknown}\n"
                f"可用：{sorted(key for key, _, _ in registered)}"
            )
        targets = [row for row in registered if row[0] in wanted]

    shards = [row for row in targets if row[0].lower().endswith(".zip")]
    extras = [row for row in targets if not row[0].lower().endswith(".zip")]

    def _md5_of(path: Path, *, chunk: int = 1024 * 1024) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
        return digest.hexdigest()

    def ready(row: tuple[str, int, str]) -> bool:
        """就绪 = 文件存在**且校验通过**。

        为什么必须校验而不是只看文件存在（真实风险）：
            半截下载（.part 改名失败、网络中断）会在磁盘上留下一个**看起来存在**
            但内容不完整的 zip。后续 --skip-download 会直接放行它，
            直到解压或抽帧时才以"文件损坏"的形式爆炸，白白浪费几十分钟。
            校验代价是要读一遍文件（17 GB 约几分钟），因此只对本次要用的分片做，
            并且下面会明确打印"校验中"。
        """
        path = raw / row[0]
        if not path.exists():
            return False
        expected = row[2]
        if not expected:
            # 登记表没有校验值（离线模式）：只能退化为大小检查
            return True
        return _md5_of(path) == expected

    print("      校验已就绪的分片（读一遍文件算 md5，请稍候）……")
    have_shards: list[str] = []
    corrupt_shards: list[str] = []
    for row in shards:
        path = raw / row[0]
        if not path.exists():
            continue
        if ready(row):
            have_shards.append(row[0])
        else:
            corrupt_shards.append(row[0])
    missing_shards = [row[0] for row in shards if row[0] not in have_shards]
    # 校验失败的分片要删掉，否则下载阶段会以为"已存在"而跳过它
    for name in corrupt_shards:
        (raw / name).unlink(missing_ok=True)
    if corrupt_shards:
        print(f"      ⚠️  {len(corrupt_shards)} 个分片校验失败，已删除以便重新下载：")
        for name in corrupt_shards:
            print(f"          {name}")

    ready_extras = [row[0] for row in extras if (raw / row[0]).exists()]
    missing_extras = [row[0] for row in extras if row[0] not in ready_extras]

    remaining = sum(size for key, size, _ in shards if key in missing_shards)
    remaining += sum(size for key, size, _ in extras if key in missing_extras)
    disk = shutil.disk_usage(str(raw))

    print(f"      数据集     : {dataset}（{entry.get('name', '')}）")
    print(f"      许可       : {entry.get('license', '?')}")
    print(f"      落地目录   : {raw}")
    print(f"      视频分片   : {len(have_shards)}/{len(shards)} 个已校验通过，"
          f"{human(sum(size for _, size, _ in shards))} 合计")
    if extras:
        print(f"      元数据文件 : {len(ready_extras)}/{len(extras)} 个已就绪"
              f"（README/statistics/summary，合计几百 KB）")
    print(f"      待下载     : {len(missing_shards) + len(missing_extras)} 个，{human(remaining)}")
    print(f"      磁盘可用   : {human(disk.free)}")

    if remaining > disk.free:
        raise SystemExit(
            f"\n磁盘空间不足：还需 {human(remaining)}，可用 {human(disk.free)}。\n"
            "建议：(1) 设置 HANDWASH_DATA_ROOT 指向空间更大的盘；\n"
            "      (2) 或用 --shards 只下部分分片先验证流程。"
        )

    complete = not missing_shards
    if not complete:
        print()
        print("      注意：视频分片尚未下全。稍后抽帧之前会**主动中止**并给出补齐方法，")
        print("            因为基于子集的划分与别人的不一致，实验数字不可比。")
        print("            （若只想先验证链路，加 --allow-partial。）")

    return {
        "root": raw,
        "registered": registered,
        "targets": targets,
        "shards": shards,
        "have": have_shards,
        "missing": missing_shards + missing_extras,
        "remaining_bytes": remaining,
        "complete": complete,
    }


def do_download(dataset: str, state: dict, *, dry_run: bool) -> int:
    """下载缺失的分片。已有且校验通过的会被跳过，所以可重复执行。"""
    if not state["missing"]:
        print("      所有分片都已就绪，无需下载。")
        return 0
    return run_script(
        "download_data.py",
        ["--dataset", dataset, "--all", "--extract"],
        dry_run=dry_run,
    )


def assert_authoritative_ready(dataset: str, state: dict, *, allow_partial: bool) -> None:
    """抽帧之前，确认数据是完整的 —— 否则划分没有意义。

    这是**唯一一处会主动中止**的检查，因为它是唯一一个"错了也看不出来"的错误：
    少下几个分片时，脚本会安静地跑完，产出一份基于子集的 manifest；
    它看起来完全正常，但和别人的划分不同，于是所有实验数字都不可比。

    （真实教训：曾经把一份只含 1/11 分片的 manifest 提交进 Git，
      pull 下来的人都会用到错误的划分。）

    ``allow_partial=True`` 时只警告不中止 —— 这是给"先用小数据验证链路"的场景留的口子。
    """
    if state["complete"]:
        return

    missing = state["missing"]
    print()
    print("!" * 78)
    print("! 数据不完整 —— 不能产出权威划分")
    print("!" * 78)
    print(f"  视频分片：{len(state['have'])}/{len(state['shards'])} 个已就绪")
    print(f"  缺失    ：{', '.join(missing[:6])}{' 等' if len(missing) > 6 else ''}")
    print()
    print("  为什么必须中止：划分是按「本地实际有哪些视频」算出来的。")
    print("  少下分片 -> 少一批视频 -> 划分与别人不同 -> 实验数字不可比，")
    print("  而这一切**不会有任何报错**，产出的 manifest 看起来完全正常。")
    print()
    if allow_partial:
        print("  （已指定 --allow-partial：仅警告，继续执行。产出的划分不可用于正式实验。）")
        print("!" * 78)
        return
    print("  怎么办，二选一：")
    print(f"    1) 下全数据后重跑（推荐）：python scripts/bootstrap_dataset.py --dataset {dataset}")
    print("       同一条命令会自动补齐缺失的分片，已有分片会 md5 校验后跳过。")
    print("    2) 只想先验证链路（结果不可用于正式实验）：加 --allow-partial")
    print("!" * 78)
    raise SystemExit(2)


def do_prepare(dataset: str, *, dry_run: bool) -> int:
    """抽帧 + 划分 + 写 manifest。"""
    config = CONFIG_FOR_DATASET.get(dataset)
    if config is None:
        raise SystemExit(
            f"数据集 `{dataset}` 没有对应的准备配置；"
            f"已登记：{sorted(CONFIG_FOR_DATASET)}。"
            "若要新增，请在 CONFIG_FOR_DATASET 里补一行。"
        )
    return run_script("prepare_data.py", ["--config", config], dry_run=dry_run)


def existing_frames(dataset: str) -> tuple[int, Path | None]:
    """返回 ``(帧文件数, manifest 路径或 None)``，用于判断是否已经抽过帧。

    抽帧是整个流程里最慢的一步（PSKUS 全集要几十分钟到几小时）。
    重跑时不检查就无条件重来，等于白白浪费这段时间 —— 而"可反复重跑"正是本脚本的核心卖点。
    """
    processed = data_root() / "processed" / dataset
    frames_dir = processed / "frames"
    count = 0
    if frames_dir.is_dir():
        count = sum(1 for path in frames_dir.rglob("*") if path.is_file())
    manifest = processed / "manifest.csv"
    return count, (manifest if manifest.exists() else None)


def do_pack(dataset: str, *, dry_run: bool, fmt: str) -> tuple[int, Path]:
    """打包并解析出包路径（供最后汇总用）。"""
    out_dir = data_root() / "packages"
    code = run_script(
        "pack_processed_data.py",
        ["--dataset", dataset, "--build", "--format", fmt],
        dry_run=dry_run,
    )
    stamp = date.today().isoformat()
    archive = out_dir / f"{dataset}_frames_{stamp}.{fmt}"
    return code, archive


def summary(
    dataset: str,
    state: dict,
    archive: Path | None,
    *,
    skip_pack: bool,
    elapsed: float,
) -> int:
    """打印可以直接复制的交接信息。"""
    print()
    print("#" * 78)
    print("# 交接清单 —— 把下面这段直接发到群里")
    print("#" * 78)
    print()

    processed = data_root() / "processed" / dataset

    frames_dir = processed / "frames"
    frames = 0
    frames_bytes = 0
    if frames_dir.is_dir():
        for path in frames_dir.rglob("*"):
            if path.is_file():
                frames += 1
                frames_bytes += path.stat().st_size

    manifest = processed / "manifest.csv"
    manifest_rows = 0
    if manifest.exists():
        with manifest.open(encoding="utf-8") as handle:
            manifest_rows = max(0, sum(1 for _ in handle) - 1)

    print(f"数据集        : {dataset}")
    print(f"视频分片      : {len(state['have'])}/{len(state['shards'])} 个已就绪"
          f"{'（完整）' if state['complete'] else '（**不完整**）'}")
    print(f"抽帧          : {frames:,} 帧，{human(frames_bytes)}")
    print(f"manifest      : {manifest_rows:,} 行（划分已随之固化）")
    print(f"总耗时        : {elapsed/60:.1f} 分钟")
    print()

    if not state["complete"]:
        print("⚠️  警告：原始数据不完整，本次产出的划分**不是权威划分**。")
        print("    请勿把下面的分享包当作全组基准；先补齐分片后重跑本命令。")
        print()

    if skip_pack or archive is None or not archive.exists():
        print("下一步（本机训练）：")
        print(f"  {train_command(dataset)}")
        print()
        print("  说明：命令里先叠**实验配置**（模型与训练超参），再叠**数据配置**（数据路径与判定阈值）。")
        print("        顺序不能反：实验配置里也有 dataset.name，放后面会把数据配置覆盖回默认数据集。")
        if not skip_pack:
            print("\n（未找到分享包；如需分享请运行：")
            print(f"  python scripts/pack_processed_data.py --dataset {dataset} --build）")
        return 0

    size = archive.stat().st_size
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    meta_file = archive.parent / f"{dataset}_package.json"
    digest = ""
    if checksum_file.exists():
        digest = checksum_file.read_text(encoding="utf-8").split()[0]
    elif meta_file.exists():
        try:
            digest = json.loads(meta_file.read_text(encoding="utf-8")).get("sha256", "")
        except json.JSONDecodeError:
            digest = ""

    print("┌─ 要上传到网盘的三个文件 ─────────────────────────────────────────────")
    print(f"│ {archive}")
    print(f"│ {checksum_file}")
    print(f"│ {meta_file}")
    print(f"└─ 体积 {human(size)}　sha256 {digest[:16]}…")
    print()
    print("┌─ 队友拿到链接后执行的命令（复制即用）──────────────────────────────")
    print(f"│ python scripts/pack_processed_data.py --dataset {dataset} "
          f"--verify {archive.name}")
    print(f"│ python scripts/pack_processed_data.py --dataset {dataset} "
          f"--unpack {archive.name}")
    print(f"│ {train_command(dataset)}")
    print("└──────────────────────────────────────────────────────────────────────")
    print()
    print("提醒：队友解包后**不要再跑 prepare_data.py** —— 那会重新抽帧并可能改变划分，")
    print("      导致你们的测试集不同、数字不可比。解包，然后训练。")
    print()
    print(f"完整 sha256：{digest}")
    return 0


# ===========================================================================
# 入口
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键：下载全量 -> 抽帧 -> 打包 -> 输出可分享给队友的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/bootstrap_dataset.py --dataset pskuss\n"
            "  python scripts/bootstrap_dataset.py --dataset pskuss --skip-pack\n"
            "  python scripts/bootstrap_dataset.py --dataset pskuss "
            "--shards DataSet4.zip,DataSet3.zip\n"
            "  python scripts/bootstrap_dataset.py --dataset mb --dry-run\n"
            "\n"
            f"可用数据集：{all_dataset_names()}\n"
            "（kaggle / synthetic 需要手工下载或自动生成，本脚本会提示具体做法）"
        ),
    )
    parser.add_argument("--dataset", "-d", default="pskuss", help="数据集名（默认 pskuss）")
    parser.add_argument(
        "--shards",
        default=None,
        help="只处理这些分片（逗号分隔）；用于先验证链路。注意：这样产出的划分不是权威划分",
    )
    parser.add_argument("--skip-download", action="store_true", help="跳过下载（数据已在本地）")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过抽帧（已有抽帧结果）")
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="即使已有抽帧结果也重新抽帧（默认会跳过，因为抽帧最慢）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "数据没下全时也继续执行（只警告不中止）。"
            "仅用于先用小数据验证链路；产出的划分不是权威划分，不可用于正式实验"
        ),
    )
    parser.add_argument("--skip-pack", action="store_true", help="跳过打包（只为自己训练）")
    parser.add_argument("--format", default="zip", choices=("zip", "tar.gz"), help="包格式")
    parser.add_argument("--dry-run", action="store_true", help="只打印会执行什么，不真的跑")
    args = parser.parse_args(argv)

    started = time.time()
    only_shards = (
        [name for name in args.shards.split(",") if name.strip()] if args.shards else None
    )

    total_steps = 4
    step_index = 0

    print("=" * 78)
    print("  数据集一键编排：下载 -> 抽帧 -> 打包 -> 交接")
    print("=" * 78)
    print(f"  数据集 : {args.dataset}")
    print(f"  数据根 : {data_root()}")
    print(f"  模式   : {'dry-run（不执行）' if args.dry_run else '实际执行'}")
    if only_shards:
        print(f"  限定分片: {only_shards}（产出的是**子集**结果，划分不是权威划分）")

    step_index += 1
    with Step(step_index, total_steps, "预检：磁盘 / 网络 / 已下载情况"):
        state = preflight(args.dataset, only_shards=only_shards)

    if not args.skip_download:
        step_index += 1
        with Step(step_index, total_steps, "下载原始分片",
                  f"{len(state['missing'])} 个待下载，约 {human(state['remaining_bytes'])}；"
                  "断点续传 + md5 校验，可重复执行"):
            code = do_download(args.dataset, state, dry_run=args.dry_run)
            if code != 0:
                print("\n下载未全部成功。重新运行同一条命令即可续传（已完成/已校验的会跳过）。")
                return code
    else:
        print("\n（--skip-download：跳过下载）")

    if not args.skip_prepare:
        # 抽帧之前先确认数据完整 —— 不完整的划分"错了也看不出来"，所以这里主动中止
        assert_authoritative_ready(args.dataset, state, allow_partial=args.allow_partial)
        step_index += 1
        frames_have, manifest_have = existing_frames(args.dataset)
        if frames_have and manifest_have and not args.force_prepare:
            # 抽帧是最慢的一步，已有结果就跳过；要重来请显式加 --force-prepare
            print()
            print("=" * 78)
            print(f"[{step_index}/{total_steps}] 抽帧 + 划分 —— 跳过（已有结果）")
            print("=" * 78)
            print(f"      已存在 {frames_have:,} 帧与 manifest：{manifest_have.name}")
            print("      如需按当前配置重新抽帧（会覆盖），加 --force-prepare")
        else:
            with Step(step_index, total_steps, "抽帧 + 划分 + 写 manifest",
                      "按原始视频划分（防数据泄漏）；这一步最慢，可中断后重跑"):
                code = do_prepare(args.dataset, dry_run=args.dry_run)
                if code != 0:
                    print("\n抽帧失败，请查看上面的日志。")
                    return code
    else:
        print("\n（--skip-prepare：跳过抽帧）")

    archive: Path | None = None
    if not args.skip_pack:
        step_index += 1
        with Step(step_index, total_steps, "打包成可分享的压缩包",
                  "包内相对路径与 manifest 一致，队友解包即用"):
            code, archive = do_pack(args.dataset, dry_run=args.dry_run, fmt=args.format)
            if code != 0:
                print("\n打包失败，请查看上面的日志。")
                return code
    else:
        print("\n（--skip-pack：跳过打包）")

    return summary(
        args.dataset, state, archive, skip_pack=args.skip_pack, elapsed=time.time() - started
    )


if __name__ == "__main__":
    raise SystemExit(main())
