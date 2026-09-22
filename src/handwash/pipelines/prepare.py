"""数据准备：把原始数据集变成"帧图像 + manifest + 划分"。

三个阶段（可用 ``--stage`` 单独跑，避免每次都重抽几万帧）
----------------------------------------------------------
``scan``      扫描原始数据，生成 clip 级清单（不写图像）
``split``     按**原始视频**划分 train/val/test（防数据泄漏的关键一步）
``frames``    按划分结果抽帧并写 manifest

**顺序不可颠倒**：必须先划分再抽帧。反过来的话，同一段视频的相邻帧会
同时出现在训练集和测试集里，指标会虚高（选题文档第五节）。

支持的数据集（``dataset.name``）：
    kaggle        Kaggle 七分类目录结构（快速原型）
    pskuss        PSKUS 医院数据集（主数据集）
    metc          METC 跨场景数据集
    jurmala       Jurmala 扩展数据集
    selfrecorded  组员自采视频（目录名即步骤名）
    synthetic     合成数据（冒烟测试）

新增数据集只允许改 ``data/sources/`` 下的适配器 + ``configs/data/`` + 本文件的
``_SOURCE_FACTORIES``，不要改 pipeline 主流程。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from handwash.core.config import ResolvedConfig
from handwash.core.labels import Step, get_label_space
from handwash.core.schema import ClipRecord, FrameRecord, Split
from handwash.errors import DatasetNotFoundError, DataError
from handwash.io.manifest import write_manifest
from handwash.io.split import assert_no_leakage, assign_frames, split_clips, split_report
from handwash.io.utils import write_json
from handwash.io.video import extract_frames, save_frame
from handwash.logging import get_logger
from handwash.paths import (
    DATA_PROCESSED_DIRNAME,
    SYNTHETIC_DIRNAME,
    PROJECT_ROOT,
    ensure_dir,
    resolve_relative,
)

__all__ = ["PrepareResult", "prepare", "SOURCE_NAMES"]

log = get_logger(__name__)

SOURCE_NAMES: tuple[str, ...] = ("kaggle", "pskuss", "metc", "jurmala", "selfrecorded", "synthetic", "frames")

#: 数据集 -> 专门的适配器函数名。结构特殊的数据集在这里登记自己的解析函数，
#: 主流程只按名字分派，不写 if/else 判断数据集（CONTRIBUTING.md R12）。
#: 未登记的数据集走 ``_records_from_video_dirs`` 通用兜底。
_SOURCE_ADAPTERS: dict[str, str] = {
    "pskuss": "_records_from_pskuss",
}


class PrepareResult:
    """准备结果摘要。"""

    def __init__(self, **kwargs: Any) -> None:
        self.manifest_path: Path | None = kwargs.get("manifest_path")
        self.num_clips: int = int(kwargs.get("num_clips", 0))
        self.num_frames: int = int(kwargs.get("num_frames", 0))
        self.splits: dict[str, Any] = dict(kwargs.get("splits", {}))
        self.stages_run: list[str] = list(kwargs.get("stages_run", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "num_clips": self.num_clips,
            "num_frames": self.num_frames,
            "splits": self.splits,
            "stages_run": self.stages_run,
        }


# ===========================================================================
# 数据集适配器：只负责"原始目录 -> 帧级或 clip 级记录"
# ===========================================================================
def _discover_source(rc: ResolvedConfig) -> tuple[str, Path]:
    """确定数据源类型与根目录。"""
    spec = rc.dataset_spec()
    name = str(rc.dataset.name).strip().lower()
    raw_root = spec.get("root") or rc.dataset.root
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if name != "synthetic" and not root.exists():
        raise DatasetNotFoundError(
            name, expected_at=root
        )
    return name, root


def _records_from_frames_dir(rc: ResolvedConfig, root: Path) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """从"已抽好帧 + 目录名=标签"的结构生成记录。

    期望结构（Kaggle / 自采视频最常用）::

        <root>/<split>/<label_name>/*.jpg|png
        或
        <root>/<label_name>/*.jpg            # 无 split 目录时由本流程重新划分

    文件名里的数字（若有）作为帧序，便于保持时间顺序。
    """
    space = get_label_space(rc.label_space)
    frames: list[FrameRecord] = []
    clips: dict[str, list[FrameRecord]] = defaultdict(list)

    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if label_dir.name.lower() in ("train", "val", "test", "external"):
            # 已经按 split 组织：递归下钻
            for split_dir in sorted(label_dir.iterdir()):
                if split_dir.is_dir():
                    frames.extend(
                        _records_from_label_dir(rc, split_dir, label_dir.name.lower(), space, clips)
                    )
            continue
        frames.extend(_records_from_label_dir(rc, label_dir, "train", space, clips))

    clip_records = [_clip_from_frames(clip_id, recs) for clip_id, recs in sorted(clips.items())]
    return frames, clip_records


def _records_from_label_dir(rc, label_dir: Path, split: str, space, clips) -> list[FrameRecord]:
    """一个"标签目录"-> 若干帧记录（每个子目录或每个文件视为一段独立视频片段）。"""
    from handwash.io.utils import IMAGE_EXTENSIONS, list_files

    by_label = space.canonicalize(label_dir.name)
    out: list[FrameRecord] = []
    videos = [p for p in sorted(label_dir.iterdir()) if p.is_dir()]
    if videos:
        for video_dir in videos:
            images = list_files(video_dir, extensions=IMAGE_EXTENSIONS)
            out.extend(_frames_from_images(rc, video_dir.name, video_dir, images, by_label, split))
    else:
        images = list_files(label_dir, extensions=IMAGE_EXTENSIONS)
        out.extend(_frames_from_images(rc, label_dir.name, label_dir, images, by_label, split))
    for rec in out:
        clips[rec.clip_id].append(rec)
    return out


def _frames_from_images(rc, clip_id: str, video_dir: Path, images: Sequence[Path], label, split: str):
    out: list[FrameRecord] = []
    for index, image in enumerate(sorted(images)):
        out.append(
            FrameRecord(
                clip_id=f"{clip_id}",
                frame_index=index,
                image_path=str(image.relative_to(_project_root())).replace("\\", "/"),
                label=label,
                dataset=str(rc.dataset.name),
                split=split,  # type: ignore[arg-type]
                timestamp_s=index / rc.dataset.prep.fps,
            )
        )
    return out


def _clip_from_frames(clip_id: str, recs: Sequence[FrameRecord]) -> ClipRecord:
    ordered = sorted(recs, key=lambda r: r.frame_index)
    fps = 5.0
    return ClipRecord(
        clip_id=clip_id,
        dataset=ordered[0].dataset,
        split=ordered[0].split,
        video_path=ordered[0].video_path,
        frame_count=len(ordered),
        fps=fps,
        duration_s=len(ordered) / fps,
        label_sequence=tuple(r.label for r in ordered),
        metadata={},
    )


def _records_from_pskuss(rc: ResolvedConfig, root: Path) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """PSKUS 真实结构适配器（已对着 DataSet4 的实际文件核对过）。

    解压后的真实结构（``<root>`` 指 ``data/raw/pskuss``）：:

        pskuss/                       # 解压落地目录（HANDWASH_DATA_ROOT 下）
          DataSet1/  DataSet2/  ...  DataSet11/
            Videos/2020-06-26_21-26-56_camera104.mp4
            Annotations/Annotator1/2020-06-26_21-26-56_camera104.csv
            Annotations/Annotator1/2020-06-26_21-26-56_camera104.json
            Annotations/Annotator2/...
            statistics.csv  summary.csv

    标注 CSV 三列：``frame_time``（毫秒）, ``is_washing``（0/1）, ``movement_code``（0-7）。
    movement_code 到 WHO 步骤的映射在 ``core/labels.py`` 的 ``_DATASET_ALIASES['pskuss']``，
    并且是用真实数据分布验证过的（code 0 = Other movement，占 62.6%）。

    **每段视频 = 一个 clip**：因此这里的 ``ClipRecord.label_sequence`` 是该视频完整
    的逐帧标签序列，划分阶段就能按视频单位做到互斥（防数据泄漏）。
    """
    from handwash.io.utils import read_csv

    space = get_label_space(rc.label_space)
    annotators = tuple(
        str(a).strip() for a in (rc.dataset_spec().get("annotators") or ["Annotator1"]) if str(a).strip()
    )

    frames: list[FrameRecord] = []
    clips: list[ClipRecord] = []
    skipped_no_annotation = 0
    skipped_unknown_label = 0

    dataset_dirs = sorted(p for p in root.rglob("DataSet*") if p.is_dir())
    if not dataset_dirs:
        raise DataError(
            f"在 {root} 下找不到 DataSet* 目录",
            hint="PSKUS 分片解压后应当出现 pskuss/DataSet1 ... DataSet11；"
            "先运行 python scripts/download_data.py --dataset pskuss --all --extract",
        )

    for dataset_dir in dataset_dirs:
        videos_dir = dataset_dir / "Videos"
        ann_root = dataset_dir / "Annotations"
        if not videos_dir.is_dir():
            continue

        # 每个视频在若干标注者目录里各有一份 csv；按优先级取第一个存在的
        for video in sorted(videos_dir.glob("*.mp4")):
            clip_id = video.stem  # 例如 2020-06-26_21-26-56_camera104
            annotation: Path | None = None
            for annotator in annotators:
                candidate = ann_root / annotator / f"{clip_id}.csv"
                if candidate.exists():
                    annotation = candidate
                    break
            if annotation is None:
                skipped_no_annotation += 1
                continue

            rows = read_csv(annotation)
            if not rows or "movement_code" not in rows[0]:
                skipped_no_annotation += 1
                continue

            clip_frames: list[FrameRecord] = []
            labels: list[Step] = []
            for index, row in enumerate(rows):
                try:
                    label = space.canonicalize(row["movement_code"])
                except Exception:  # noqa: BLE001 - 无法识别的 code 记为 unknown 并计数
                    skipped_unknown_label += 1
                    label = Step.UNKNOWN
                labels.append(label)
                clip_frames.append(
                    FrameRecord(
                        clip_id=clip_id,
                        frame_index=index,
                        image_path=str(video.relative_to(_project_root())).replace("\\", "/"),
                        label=label,
                        dataset=str(rc.dataset.name),
                        split="train",  # 占位：真实 split 由划分阶段写入
                        timestamp_s=_frame_time_to_seconds(row.get("frame_time", ""), index, rc),
                    )
                )

            frames.extend(clip_frames)
            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    dataset=str(rc.dataset.name),
                    split="train",
                    video_path=str(video.relative_to(_project_root())).replace("\\", "/"),
                    frame_count=len(clip_frames),
                    fps=rc.dataset.prep.fps,
                    duration_s=len(clip_frames) / rc.dataset.prep.fps,
                    label_sequence=tuple(labels),
                    metadata={
                        "pskuss_dataset": dataset_dir.name,
                        "annotation": str(annotation.relative_to(_project_root())).replace("\\", "/"),
                    },
                )
            )

    if skipped_no_annotation:
        log.warning("有 %d 段视频找不到标注文件，已跳过", skipped_no_annotation)
    if skipped_unknown_label:
        log.warning("有 %d 帧的 movement_code 无法识别，已记为 unknown", skipped_unknown_label)
    if not clips:
        raise DataError(
            f"没有解析出任何 PSKUS 片段（root={root}）",
            hint="检查目录结构是否为 <root>/DataSet*/Videos/*.mp4 与 Annotations/AnnotatorN/*.csv",
        )

    log.info(
        "PSKUS 适配完成：%d 段视频 / %d 帧（标注者优先级 %s）",
        len(clips), len(frames), list(annotators),
    )
    return frames, clips


def _frame_time_to_seconds(raw: str, index: int, rc: ResolvedConfig) -> float:
    """把 PSKUS 的 ``frame_time``（毫秒）换算成秒；解析失败则按抽帧率估算。"""
    try:
        return float(str(raw).strip()) / 1000.0
    except (TypeError, ValueError):
        return index / rc.dataset.prep.fps


def _records_from_video_dirs(rc: ResolvedConfig, root: Path) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """通用兜底适配器：``<root>/<label_name>/<video>.mp4`` 或 ``<root>/<clip_id>.csv``。

    用于 METC / Jurmala / 自采视频这类"目录名即标签"或"逐帧 csv"的结构。
    若某个数据集的结构与众不同，**在这里加一个专门的适配器函数**，
    并在 ``_SOURCE_FACTORIES`` 里注册，不要改主流程。
    """
    from handwash.io.utils import read_csv

    space = get_label_space(rc.label_space)
    frames: list[FrameRecord] = []
    clips: dict[str, list[FrameRecord]] = defaultdict(list)

    # 形式 1：目录名即标签
    video_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    handled = False
    for label_dir in video_dirs:
        try:
            label = space.canonicalize(label_dir.name)
        except Exception:  # noqa: BLE001 - 不是标签目录就跳过
            continue
        from handwash.io.utils import list_videos

        for video in list_videos(label_dir):
            clip_id = video.stem
            recs = _placeholder_frames(rc, clip_id, video, label)
            frames.extend(recs)
            clips[clip_id].extend(recs)
            handled = True

    # 形式 2：每段视频一个 csv
    for csv_path in sorted(root.glob("*.csv")):
        rows = read_csv(csv_path)
        if not rows or "label" not in rows[0]:
            continue
        clip_id = csv_path.stem
        recs: list[FrameRecord] = []
        for index, row in enumerate(rows):
            try:
                label = space.canonicalize(row["label"])
            except Exception:  # noqa: BLE001 - 无法识别的标签行跳过并计数
                continue
            recs.append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_index=int(row.get("frame", index) or index),
                    image_path="",  # 由抽帧阶段填充
                    label=label,
                    dataset=str(rc.dataset.name),
                    split="train",
                    timestamp_s=float(row["timestamp"]) if row.get("timestamp") else None,
                )
            )
        if recs:
            frames.extend(recs)
            clips[clip_id].extend(recs)
            handled = True

    if not handled:
        raise DataError(
            f"在 {root} 下没有识别出任何可用的标注结构",
            hint="见 docs/DATA.md 对各数据集目录结构的说明；"
            "必要时在 pipelines/prepare.py 里为该数据集新增一个专门适配器。",
        )

    clip_records = [
        ClipRecord(
            clip_id=clip_id,
            dataset=str(rc.dataset.name),
            split="train",  # 占位：真实 split 由 split 阶段写入
            video_path=str(recs[0].image_path or recs[0].clip_id),
            frame_count=len(recs),
            fps=rc.dataset.prep.fps,
            duration_s=len(recs) / rc.dataset.prep.fps,
            label_sequence=tuple(r.label for r in sorted(recs, key=lambda r: r.frame_index)),
        )
        for clip_id, recs in sorted(clips.items())
    ]
    return frames, clip_records


def _placeholder_frames(rc: ResolvedConfig, clip_id: str, video: Path, label) -> list[FrameRecord]:
    """形式 3 的帧记录：帧数未知，先用 1 条占位，抽帧阶段再展开。

    这样做的原因：先把"有哪几段视频、哪一步"确定下来，划分才能按视频进行；
    真正的帧数只有抽帧时才知道，不能反过来。
    """
    return [
        FrameRecord(
            clip_id=clip_id,
            frame_index=0,
            image_path=str(video.relative_to(_project_root())).replace("\\", "/"),
            label=label,
            dataset=str(rc.dataset.name),
            split="train",
            timestamp_s=0.0,
        )
    ]


def _project_root() -> Path:
    from handwash.paths import PROJECT_ROOT as root

    return root


def _resolve(relative: str | Path) -> Path:
    """把配置里的相对路径解析成绝对路径（相对仓库根目录）。"""
    return resolve_relative(relative)


# ===========================================================================
# 主入口
# ===========================================================================
def prepare(rc: ResolvedConfig, *, stage: str = "all") -> PrepareResult:
    """执行数据准备。

    Parameters
    ----------
    stage:
        ``all`` / ``scan`` / ``split`` / ``frames``
    """
    valid = ("all", "scan", "split", "frames")
    if stage not in valid:
        raise DataError(f"未知的 stage：{stage!r}", hint=f"允许：{list(valid)}")

    name, root = _discover_source(rc)
    spec = rc.dataset_spec()
    out_root = resolve_relative(spec.get("processed_dir") or DATA_PROCESSED_DIRNAME)
    frames_root = resolve_relative(
        spec.get("frames_dir") or f"{DATA_PROCESSED_DIRNAME}/{rc.dataset.name}/frames"
    )
    manifest_path = resolve_relative(spec.get("manifest") or (out_root / "manifest.csv"))

    log.info(
        "数据准备：dataset=%s，root=%s，stage=%s，label_space=%s",
        name, root, stage, rc.label_space,
    )
    ensure_dir(out_root)

    # --- 1) 取 clip 级清单 -------------------------------------------------
    if name == "synthetic":
        from handwash.data.synthetic import write_synthetic_dataset

        frames, clips = write_synthetic_dataset(
            resolve_relative(SYNTHETIC_DIRNAME),
            num_clips=24,
            frames_per_clip=18,
            image_size=96,
            seed=rc.runtime.seed,
        )
        split_map: dict[Split, list[ClipRecord]] = {"train": [], "val": [], "test": []}
        for clip in clips:
            split_map[clip.split].append(clip)  # type: ignore[index]
        prepared_frames = frames
    elif name == "kaggle" or _looks_like_image_tree(root):
        frames, clips = _records_from_frames_dir(rc, root)
        split_map = split_clips(
            clips,
            ratios={"train": rc.split.train, "val": rc.split.val, "test": rc.split.test},
            seed=rc.split.seed,
            group_key=rc.split.group_key,
            stratify_by=rc.split.stratify_by,
        )
        prepared_frames = frames
    else:
        # 结构特殊的数据集走专门适配器；其余走通用兜底
        adapter_name = _SOURCE_ADAPTERS.get(name)
        adapter = globals().get(adapter_name) if adapter_name else None
        if adapter is None:
            adapter = _records_from_video_dirs
        elif adapter_name:
            log.info("使用专门适配器：%s", adapter_name)
        _, clips = adapter(rc, root)
        split_map = split_clips(
            clips,
            ratios={"train": rc.split.train, "val": rc.split.val, "test": rc.split.test},
            seed=rc.split.seed,
            group_key=rc.split.group_key,
            stratify_by=rc.split.stratify_by,
        )
        prepared_frames = []

    if rc.split.guard_leakage:
        assert_no_leakage(split_map, group_key=rc.split.group_key)

    # --- 2) 抽帧（仅当需要时执行）-----------------------------------------
    if stage in ("all", "frames") and not prepared_frames:
        prepared_frames = _extract_and_register(
            rc, split_map, frames_root=frames_root, out_root=out_root
        )
    elif stage in ("all", "frames") and prepared_frames and name != "synthetic":
        # 已有帧目录（如 Kaggle）：只把 split 写回记录
        prepared_frames = _apply_split(prepared_frames, split_map)

    if not prepared_frames:
        prepared_frames = _apply_split([], split_map) if stage == "scan" else prepared_frames

    report = split_report(split_map)
    if prepared_frames:
        write_manifest(manifest_path, prepared_frames)

    # 划分报告里同时给出"抽帧前"和"实际写入 manifest"的帧数。
    # 两者常常不等，因为 split.max_frames_per_clip_* 会对长视频等间隔截断；
    # 只报一个数字会让人以为 manifest 写坏了（真实踩过这个坑）。
    actual_by_split: dict[str, int] = defaultdict(int)
    for record in prepared_frames:
        actual_by_split[record.split] += 1
    for split_name, info in report.items():
        written = actual_by_split.get(split_name, 0)
        info["num_frames_extracted"] = info["num_frames"]
        info["num_frames_in_manifest"] = written
        info["num_frames"] = written
        if written and written != info["num_frames_extracted"]:
            info["note"] = (
                f"抽帧得到 {info['num_frames_extracted']} 帧，"
                f"按 max_frames_per_clip 等间隔截断后写入 manifest {written} 帧"
            )

    write_json(
        out_root / "split_report.json",
        {
            "dataset": name,
            "root": str(root),
            "manifest": str(manifest_path),
            "split": report,
            "config_hash": rc.config_hash,
            "notes": (
                "num_frames_extracted 是抽帧产出的帧数；num_frames/num_frames_in_manifest "
                "是实际写入 manifest 的帧数（受 split.max_frames_per_clip_* 限制）。"
                "训练与评估用的是后者。"
            ),
        },
    )

    result = PrepareResult(
        manifest_path=manifest_path if prepared_frames else None,
        num_clips=sum(len(v) for v in split_map.values()),
        num_frames=len(prepared_frames),
        splits=report,
        stages_run=[stage],
    )
    log.info(
        "准备完成：%d 段视频 / manifest 中 %d 帧；manifest=%s",
        result.num_clips, result.num_frames, result.manifest_path,
    )
    return result


def _looks_like_image_tree(root: Path) -> bool:
    """判断目录树是"图像目录"还是"视频+标注"。"""
    from handwash.io.utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, list_files

    images = list_files(root, extensions=IMAGE_EXTENSIONS, recursive=True)
    videos = list_files(root, extensions=VIDEO_EXTENSIONS, recursive=True)
    return len(images) > len(videos)


def _apply_split(
    frames: Sequence[FrameRecord], split_map: dict[Split, list[ClipRecord]]
) -> list[FrameRecord]:
    """把划分结果写回帧记录（clip -> split）。"""
    clip_split: dict[str, Split] = {}
    for split_name, clip_list in split_map.items():
        for clip in clip_list:
            clip_split[clip.clip_id] = split_name  # type: ignore[assignment]
    out: list[FrameRecord] = []
    for rec in frames:
        target = clip_split.get(rec.clip_id, "train")
        out.append(
            FrameRecord(
                clip_id=rec.clip_id,
                frame_index=rec.frame_index,
                image_path=rec.image_path,
                label=rec.label,
                dataset=rec.dataset,
                split=target,
                timestamp_s=rec.timestamp_s,
            )
        )
    return out


def _extract_and_register(
    rc: ResolvedConfig,
    split_map: dict[Split, list[ClipRecord]],
    *,
    frames_root: Path,
    out_root: Path,
) -> list[FrameRecord]:
    """按划分结果抽帧并登记（**先划分、后抽帧**）。"""
    ensure_dir(frames_root)
    records: list[FrameRecord] = []
    caps = {
        "train": rc.split.max_frames_per_clip_train,
        "val": rc.split.max_frames_per_clip_eval,
        "test": rc.split.max_frames_per_clip_eval,
        "external": rc.split.max_frames_per_clip_eval,
    }

    for split_name, clip_list in split_map.items():
        for clip in clip_list:
            video_path = Path(clip.video_path)
            if not video_path.is_absolute():
                video_path = _project_root() / video_path
            if not video_path.exists() or video_path.suffix.lower() not in (
                ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg",
            ):
                # 已经是帧目录（Kaggle 类）或视频缺失：跳过抽帧，交给 _apply_split
                continue
            try:
                indices, stamps, frames = extract_frames(
                    video_path,
                    sample_fps=rc.dataset.prep.fps,
                    frame_step=rc.dataset.prep.frame_step,
                    max_frames=caps.get(split_name),
                )
            except Exception as exc:  # noqa: BLE001 - 单段失败不中断整批
                log.error("抽帧失败，已跳过：%s（%s）", video_path, exc)
                continue

            if len(frames) < rc.dataset.prep.min_frames_per_clip:
                log.warning(
                    "视频过短（%d 帧 < %d），已跳过：%s",
                    len(frames), rc.dataset.prep.min_frames_per_clip, video_path.name,
                )
                continue

            label = clip.label_sequence[0] if clip.label_sequence else None

            for local_index, (_, stamp, frame) in enumerate(zip(indices, stamps, frames, strict=True)):
                # 【关键】image_path 记录成**相对 frames_root** 的路径（`<clip_id>/00000.jpg`），
                # 与 data/dataset.py 的 _image_root_from() 约定一致：那边把 image_root
                # 解析为配置里的 frames_dir，再用 image_root / image_path 打开文件。
                # 早期版本这里写成 `frames/<clip_id>/...`（相对数据集 root），
                # 导致训练时报"帧图像不存在：<dataset.root>/frames/..."，
                # 而实际帧落在 data/processed/<dataset>/frames/ 下。
                relative = Path(clip.clip_id) / f"{local_index:05d}.jpg"
                save_frame(frame, frames_root / clip.clip_id / f"{local_index:05d}.jpg",
                           quality=rc.dataset.prep.jpeg_quality)
                records.append(
                    FrameRecord(
                        clip_id=clip.clip_id,
                        frame_index=local_index,
                        image_path=str(relative).replace("\\", "/"),
                        label=label if label is not None else Step.UNKNOWN,
                        dataset=clip.dataset,
                        split=split_name,  # type: ignore[arg-type]
                        timestamp_s=stamp,
                    )
                )
    if not records:
        log.warning(
            "抽帧阶段没有产生任何帧（可能数据源已经是图像目录）。"
            "若使用 Kaggle 类数据，请确认 frames_dir 已正确指向图像根目录。"
        )
    return records
