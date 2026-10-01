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

import bisect
import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from handwash.core.config import ResolvedConfig
from handwash.core.labels import NON_WASH_STEPS, Step, get_label_space
from handwash.core.schema import ClipRecord, FrameRecord, Split
from handwash.errors import ConfigError, DataError, DatasetNotFoundError
from handwash.io.manifest import write_manifest
from handwash.io.split import assert_no_leakage, split_clips, split_report
from handwash.io.utils import write_json
from handwash.io.video import extract_frames, save_frame
from handwash.logging import get_logger
from handwash.paths import (
    DATA_PROCESSED_DIRNAME,
    SYNTHETIC_DIRNAME,
    ensure_dir,
    resolve_relative,
)

__all__ = ["SOURCE_NAMES", "PrepareResult", "prepare"]

log = get_logger(__name__)

SOURCE_NAMES: tuple[str, ...] = ("kaggle", "pskuss", "metc", "jurmala", "selfrecorded", "synthetic", "frames")

#: 数据集 -> 专门的适配器函数名。结构特殊的数据集在这里登记自己的解析函数，
#: 主流程只按名字分派，不写 if/else 判断数据集（CONTRIBUTING.md R12）。
#: 未登记的数据集走 ``_records_from_video_dirs`` 通用兜底。
_SOURCE_ADAPTERS: dict[str, str] = {
    "pskuss": "_records_from_pskuss",
    "metc": "_records_from_metc",
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
    root = resolve_relative(Path(str(raw_root)).expanduser())
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

    clip_records = [
        _clip_from_frames(clip_id, recs, fps=rc.dataset.prep.fps)
        for clip_id, recs in sorted(clips.items())
    ]
    return frames, clip_records


def _records_from_label_dir(rc, label_dir: Path, split: str, space, clips) -> list[FrameRecord]:
    """一个"标签目录"-> 若干帧记录（每个子目录或每个文件视为一段独立视频片段）。"""
    from handwash.io.utils import IMAGE_EXTENSIONS, list_files

    by_label = space.canonicalize(label_dir.name)
    if by_label not in space:
        if by_label in NON_WASH_STEPS and not rc.dataset.include_non_wash:
            log.info("按配置排除辅助类别目录：%s", label_dir)
            return []
        raise DataError(
            f"标签 {by_label.value!r} 不在当前标签空间 {space.name!r} 中",
            hint="调整 dataset.label_space / include_non_wash，确保数据标签可被模型表示。",
        )
    out: list[FrameRecord] = []
    videos = [p for p in sorted(label_dir.iterdir()) if p.is_dir()]
    if videos:
        for video_dir in videos:
            images = sorted(list_files(video_dir, extensions=IMAGE_EXTENSIONS), key=_natural_sort_key)
            clip_id = f"{label_dir.name}/{video_dir.name}"
            out.extend(_frames_from_images(rc, clip_id, video_dir, images, by_label, split))
    else:
        images = list_files(label_dir, extensions=IMAGE_EXTENSIONS)
        # A flat image folder contains independent images unless filenames share
        # an explicit source-video frame suffix (for example clip_f00123.jpg).
        for image in images:
            prefix_match = re.match(r"^frame[_-]?(\d+)[_-](.+)$", image.stem, flags=re.IGNORECASE)
            suffix_match = re.match(r"^(.*?)[_-]f(?:rame)?[_-]?(\d+)$", image.stem, flags=re.IGNORECASE)
            if prefix_match:
                clip_id = f"source/{prefix_match.group(2)}"
                frame_index = int(prefix_match.group(1))
            elif suffix_match:
                clip_id = f"source/{suffix_match.group(1)}"
                frame_index = int(suffix_match.group(2))
            else:
                clip_id = f"{label_dir.name}/{image.name}"
                frame_index = 0
            out.append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    image_path=str(image.resolve()),
                    label=by_label,
                    dataset=str(rc.dataset.name),
                    split=split,  # type: ignore[arg-type]
                    timestamp_s=None,
                )
            )
    for rec in out:
        clips[rec.clip_id].append(rec)
    return out


def _frames_from_images(rc, clip_id: str, video_dir: Path, images: Sequence[Path], label, split: str):
    out: list[FrameRecord] = []
    for index, image in enumerate(sorted(images, key=_natural_sort_key)):
        out.append(
            FrameRecord(
                clip_id=f"{clip_id}",
                frame_index=index,
                image_path=str(image.resolve()),
                label=label,
                dataset=str(rc.dataset.name),
                split=split,  # type: ignore[arg-type]
                timestamp_s=index / rc.dataset.prep.fps,
            )
        )
    return out


def _natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Sort frame_2 before frame_10 while keeping deterministic path ordering."""
    parts: list[tuple[int, object]] = []
    for token in re.split(r"(\d+)", path.as_posix().lower()):
        if token.isdigit():
            parts.append((1, int(token)))
        else:
            parts.append((0, token))
    return tuple(parts)


def _clip_from_frames(clip_id: str, recs: Sequence[FrameRecord], *, fps: float) -> ClipRecord:
    ordered = sorted(recs, key=lambda r: r.frame_index)
    if len({record.frame_index for record in ordered}) != len(ordered):
        raise DataError(
            f"图像 clip={clip_id} 中 frame_index 重复",
            hint="检查多类别目录中是否重复保存了同一源视频帧，或修正帧号解析规则。",
        )
    splits = {record.split for record in ordered}
    if len(splits) != 1:
        raise DataError(
            f"同一图像 clip={clip_id} 出现在多个预置 split：{sorted(splits)}",
            hint="按原视频分组的数据不能同时出现在 train/val/test；请修正预置目录。",
        )
    return ClipRecord(
        clip_id=clip_id,
        dataset=ordered[0].dataset,
        split=ordered[0].split,
        video_path=clip_id,
        frame_count=len(ordered),
        fps=fps,
        duration_s=len(ordered) / fps,
        label_sequence=tuple(r.label for r in ordered),
        metadata={"label_timestamps_s": tuple(record.timestamp_s for record in ordered)},
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
        from handwash.io.utils import list_videos

        for video in list_videos(videos_dir):
            clip_id = f"{dataset_dir.name}/{video.stem}"
            annotation_stem = video.stem
            annotation: Path | None = None
            for annotator in annotators:
                candidate = ann_root / annotator / f"{annotation_stem}.csv"
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
                except Exception:
                    skipped_unknown_label += 1
                    label = Step.UNKNOWN
                labels.append(label)
                clip_frames.append(
                    FrameRecord(
                        clip_id=clip_id,
                        frame_index=index,
                        image_path=_portable_path(video),
                        label=label,
                        dataset=str(rc.dataset.name),
                        split="train",  # 占位：真实 split 由划分阶段写入
                        timestamp_s=_frame_time_to_seconds(row.get("frame_time", "")),
                    )
                )

            frames.extend(clip_frames)
            valid_times = [rec.timestamp_s for rec in clip_frames if rec.timestamp_s is not None]
            clip_duration = (
                max(valid_times) - min(valid_times)
                if len(valid_times) >= 2
                else len(clip_frames) / rc.dataset.prep.fps
            )
            clips.append(
                ClipRecord(
                    clip_id=clip_id,
                    dataset=str(rc.dataset.name),
                    split="train",
                    video_path=_portable_path(video),
                    frame_count=len(clip_frames),
                    fps=rc.dataset.prep.fps,
                    duration_s=clip_duration,
                    label_sequence=tuple(labels),
                    metadata={
                        "pskuss_dataset": dataset_dir.name,
                        "annotation": _portable_path(annotation),
                        "label_timestamps_s": tuple(rec.timestamp_s for rec in clip_frames),
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


def _frame_time_to_seconds(raw: str) -> float | None:
    """把 PSKUS 的 ``frame_time``（毫秒）换算成秒；解析失败返回 None。"""
    try:
        value = float(str(raw).strip()) / 1000.0
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def _records_from_metc(rc: ResolvedConfig, root: Path) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """Read Zenodo METC videos and their same-stem, per-frame JSON annotations.

    The published subset uses movement codes 0..6 (0=other, 1..6=WHO steps).
    Its labels contain no separate faucet events.
    """
    from handwash.io.utils import VIDEO_EXTENSIONS

    space = get_label_space(rc.label_space)
    videos_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos_by_stem[path.stem].append(path)
    annotations = sorted(root.rglob("*.json"))
    if not annotations:
        raise DataError(
            f"METC 数据目录 {root} 中没有 JSON 标注",
            hint="解压 Zenodo 的 Interface_number_1..3.zip；每段视频需要同名 JSON 标注。",
        )

    frames: list[FrameRecord] = []
    clips: list[ClipRecord] = []
    missing_video = 0
    for annotation in annotations:
        candidates = videos_by_stem.get(annotation.stem, [])
        if not candidates:
            missing_video += 1
            continue
        ann_parts = annotation.parent.relative_to(root).parts
        ranked = sorted(
            candidates,
            key=lambda video: sum(
                1
                for left, right in zip(ann_parts, video.parent.relative_to(root).parts, strict=False)
                if left == right
            ),
            reverse=True,
        )
        if len(ranked) > 1:
            best_score = sum(
                1
                for left, right in zip(ann_parts, ranked[0].parent.relative_to(root).parts, strict=False)
                if left == right
            )
            next_score = sum(
                1
                for left, right in zip(ann_parts, ranked[1].parent.relative_to(root).parts, strict=False)
                if left == right
            )
            if best_score == next_score:
                raise DataError(
                    f"METC 标注 {annotation} 对应多个同名视频：{ranked[:2]}",
                    hint="请检查解压目录中是否存在重名视频/标注。",
                )
        video = ranked[0]
        try:
            payload = json.loads(annotation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"无法读取 METC 标注文件：{annotation}", hint=str(exc)) from exc

        raw_labels: Any = payload
        if isinstance(payload, dict):
            raw_labels = next(
                (payload[key] for key in ("labels", "movement_codes", "annotations", "frames") if key in payload),
                None,
            )
        if isinstance(raw_labels, dict):
            def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
                try:
                    return (0, f"{int(item[0]):012d}")
                except ValueError:
                    return (1, item[0])

            raw_labels = [value for _, value in sorted(raw_labels.items(), key=sort_key)]
        if not isinstance(raw_labels, list) or not raw_labels:
            raise DataError(
                f"METC 标注格式无效：{annotation}",
                hint="期望顶层 labels/frames 为非空逐帧列表或按帧号索引的对象。",
            )

        sequence: list[Step] = []
        timestamps: list[float | None] = []
        for index, item in enumerate(raw_labels):
            value = item
            timestamp: float | None = None
            if isinstance(item, dict):
                value = next(
                    (item[key] for key in ("movement_code", "label", "code", "movement", "class") if key in item),
                    None,
                )
                raw_time = next(
                    (item[key] for key in ("timestamp_s", "timestamp", "time", "frame_time") if key in item),
                    None,
                )
                if raw_time not in (None, ""):
                    raw_time_text = str(raw_time).strip()
                    is_milliseconds = raw_time_text.lower().endswith("ms")
                    numeric_time = raw_time_text[:-2].strip() if is_milliseconds else raw_time_text
                    try:
                        timestamp = float(numeric_time)
                    except ValueError as exc:
                        raise DataError(
                            f"METC 标注时间戳无法解析：{raw_time!r}（{annotation}，第 {index + 1} 条）",
                            hint="时间戳必须是数字秒数，或以 ms 结尾的毫秒数。",
                        ) from exc
                    if is_milliseconds:
                        timestamp /= 1000.0
                if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
                    raise DataError(
                        f"METC 标注时间戳无效：{raw_time!r}（{annotation}，第 {index + 1} 条）",
                        hint="时间戳必须是非负有限秒数，或以 ms 结尾的毫秒数。",
                    )
            try:
                label = space.canonicalize(value)
            except Exception as exc:
                raise DataError(
                    f"METC 标签无法映射：{value!r}（{annotation}，第 {index + 1} 条）",
                    hint="公开标签应为 0..6；核对 METC JSON schema 与 dataset.label_space。",
                ) from exc
            sequence.append(label)
            timestamps.append(timestamp)

        clip_id = video.relative_to(root).with_suffix("").as_posix()
        fps = float(rc.dataset.prep.fps)
        valid_timestamps = [stamp for stamp in timestamps if stamp is not None]
        if len(valid_timestamps) >= 2:
            clip_duration = max(valid_timestamps) - min(valid_timestamps)
        else:
            clip_duration = len(sequence) / fps
        placeholder = [
            FrameRecord(
                clip_id=clip_id,
                frame_index=index,
                image_path=_portable_path(video),
                label=label,
                dataset=str(rc.dataset.name),
                split="train",
                timestamp_s=timestamps[index] if timestamps[index] is not None else index / fps,
            )
            for index, label in enumerate(sequence)
        ]
        frames.extend(placeholder)
        clips.append(
            ClipRecord(
                clip_id=clip_id,
                dataset=str(rc.dataset.name),
                split="train",
                video_path=_portable_path(video),
                frame_count=len(sequence),
                fps=fps,
                duration_s=clip_duration,
                label_sequence=tuple(sequence),
                metadata={"annotation": _portable_path(annotation), "label_timestamps_s": tuple(timestamps)},
            )
        )

    if missing_video:
        log.warning("METC 有 %d 个 JSON 找不到同名视频，已跳过", missing_video)
    if not clips:
        raise DataError(
            f"METC 没有解析出任何视频与标注配对（root={root}）",
            hint="检查 Interface_number_* 下视频和 JSON 的文件 stem 是否一致。",
        )
    return frames, clips


def _records_from_video_dirs(rc: ResolvedConfig, root: Path) -> tuple[list[FrameRecord], list[ClipRecord]]:
    """通用兜底适配器：``<root>/<label_name>/<video>.mp4`` 或 ``<root>/<clip_id>.csv``。

    用于 METC / Jurmala / 自采视频这类"目录名即标签"或"逐帧 csv"的结构。
    若某个数据集的结构与众不同，**在这里加一个专门的适配器函数**，
    并在 ``_SOURCE_FACTORIES`` 里注册，不要改主流程。
    """
    from handwash.io.utils import VIDEO_EXTENSIONS, list_videos, read_csv

    space = get_label_space(rc.label_space)
    frames: list[FrameRecord] = []
    clips: dict[str, list[FrameRecord]] = defaultdict(list)

    # 形式 1：目录名即标签
    video_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    handled = False
    for label_dir in video_dirs:
        try:
            label = space.canonicalize(label_dir.name)
        except Exception:
            continue
        if label not in space:
            if label in NON_WASH_STEPS and not rc.dataset.include_non_wash:
                log.info("按配置排除辅助类别目录：%s", label_dir)
                continue
            raise DataError(
                f"标签 {label.value!r} 不在当前标签空间 {space.name!r} 中",
                hint="调整 dataset.label_space / include_non_wash，确保数据标签可被模型表示。",
            )
        for video in list_videos(label_dir):
            clip_id = video.relative_to(root).with_suffix("").as_posix()
            recs = _placeholder_frames(rc, clip_id, video, label)
            frames.extend(recs)
            clips[clip_id].extend(recs)
            handled = True

    # 形式 2：每段视频一个 csv
    for csv_path in sorted(root.rglob("*.csv")):
        rows = read_csv(csv_path)
        if not rows or "label" not in rows[0]:
            continue
        matches = [
            path for path in root.rglob(f"{csv_path.stem}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not matches:
            continue
        if len(matches) > 1:
            raise DataError(
                f"标注 CSV {csv_path} 对应多个同名视频：{matches}",
                hint="请移除歧义文件，或把标注放到能唯一匹配视频的目录结构中。",
            )
        video = matches[0]
        clip_id = video.relative_to(root).with_suffix("").as_posix()
        recs: list[FrameRecord] = []
        for index, row in enumerate(rows):
            try:
                label = space.canonicalize(row["label"])
            except ConfigError as exc:
                raise DataError(
                    f"标注 CSV 标签无法映射：{row.get('label')!r}（{csv_path}，第 {index + 2} 行）",
                    hint="在 core/labels.py 注册数据集标签别名，或修正标注文件；不要静默丢弃未识别标签。",
                ) from exc
            if label not in space:
                if label in NON_WASH_STEPS and not rc.dataset.include_non_wash:
                    continue
                raise DataError(
                    f"CSV 标签 {label.value!r} 不在当前标签空间 {space.name!r} 中：{csv_path}",
                    hint="调整 dataset.label_space / include_non_wash，或修正标注。",
                )
            raw_timestamp = row.get("timestamp_s") or row.get("timestamp")
            timestamp: float | None = None
            if raw_timestamp not in (None, ""):
                try:
                    timestamp = float(raw_timestamp)
                except (TypeError, ValueError) as exc:
                    raise DataError(
                        f"标注 CSV 时间戳无法解析：{raw_timestamp!r}（{csv_path}，第 {index + 2} 行）",
                        hint="timestamp_s / timestamp 必须是秒单位的非负有限数字。",
                    ) from exc
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise DataError(
                        f"标注 CSV 时间戳无效：{raw_timestamp!r}（{csv_path}，第 {index + 2} 行）",
                        hint="timestamp_s / timestamp 必须是秒单位的非负有限数字。",
                    )
            recs.append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_index=int(row.get("frame", index) or index),
                    image_path=_portable_path(video),
                    label=label,
                    dataset=str(rc.dataset.name),
                    split="train",
                    timestamp_s=timestamp,
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
            video_path=next((record.image_path for record in recs if record.image_path), clip_id),
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
            image_path=_portable_path(video),
            label=label,
            dataset=str(rc.dataset.name),
            split="train",
            timestamp_s=0.0,
        )
    ]


def _project_root() -> Path:
    from handwash.paths import PROJECT_ROOT

    return PROJECT_ROOT


def _portable_path(path: Path) -> str:
    """Keep repository files relocatable and external data paths usable."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(_project_root()).as_posix()
    except ValueError:
        return str(resolved)


def _as_external_split(clips: Sequence[ClipRecord]) -> dict[Split, list[ClipRecord]]:
    """Reserve every clip for external evaluation without random partitioning."""
    return {
        "train": [],
        "val": [],
        "test": [],
        "external": [replace(clip, split="external") for clip in clips],
    }


def _label_for_sample(clip: ClipRecord, source_index: int, timestamp_s: float) -> Step | None:
    """Map a decoded source frame to its nearest original per-frame annotation."""
    sequence = clip.label_sequence
    if not sequence:
        return None
    if len(sequence) == 1:
        return sequence[0] if sequence[0] is not Step.UNKNOWN else None

    raw_times = clip.metadata.get("label_timestamps_s", ())
    if isinstance(raw_times, Sequence) and len(raw_times) == len(sequence):
        indexed = sorted((float(value), index) for index, value in enumerate(raw_times) if value is not None)
        if indexed:
            timestamps = [item[0] for item in indexed]
            position = bisect.bisect_left(timestamps, timestamp_s)
            candidates = (max(0, position - 1), min(len(indexed) - 1, position))
            nearest = min(candidates, key=lambda idx: abs(timestamps[idx] - timestamp_s))
            label = sequence[indexed[nearest][1]]
            return None if label is Step.UNKNOWN else label

    if 0 <= source_index < len(sequence):
        label = sequence[source_index]
        return None if label is Step.UNKNOWN else label
    return None


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
    if name == "synthetic" and stage != "all":
        raise DataError(
            "synthetic 数据集是临时生成的，必须一次运行 `--stage all`",
            hint="若要按阶段处理真实数据，请改用 PSKUS、METC 或图像目录数据集。",
        )
    spec = rc.dataset_spec()
    external_only = bool(spec.get("external_only", False))
    out_root = resolve_relative(spec.get("processed_dir") or DATA_PROCESSED_DIRNAME)
    frames_root = resolve_relative(
        spec.get("frames_dir") or f"{DATA_PROCESSED_DIRNAME}/{rc.dataset.name}/frames"
    )
    manifest_path = resolve_relative(spec.get("manifest") or (out_root / "manifest.csv"))
    split_contract_path = out_root / "clip_splits.json"

    log.info(
        "数据准备：dataset=%s，root=%s，stage=%s，label_space=%s",
        name, root, stage, rc.label_space,
    )
    ensure_dir(out_root)

    # --- 1) 取 clip 级清单 -------------------------------------------------
    presplit_images = False
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
    elif name == "kaggle" or (name != "metc" and _looks_like_image_tree(root)):
        frames, clips = _records_from_frames_dir(rc, root)
        split_dirs = {"train", "val", "test", "external"}.intersection(
            p.name.lower() for p in root.iterdir() if p.is_dir()
        )
        if split_dirs:
            presplit_images = True
            split_map = {key: [] for key in ("train", "val", "test", "external")}
            for clip in clips:
                if clip.split not in split_dirs:
                    raise DataError(
                        f"圖像資料集已有 split 目錄，但 clip={clip.clip_id} 的 split={clip.split!r} 不匹配",
                        hint="每張圖應位於 train/val/test/external/<label>/ 下，或刪除 split 目錄後由程序重新劃分。",
                    )
                split_map[clip.split].append(clip)  # type: ignore[index]
        elif external_only:
            split_map = _as_external_split(clips)
        else:
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
        if external_only:
            split_map = _as_external_split(clips)
        else:
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

    current_clips = {
        clip.clip_id: clip for clip_list in split_map.values() for clip in clip_list
    }
    listed_clip_count = sum(len(clip_list) for clip_list in split_map.values())
    if len(current_clips) != listed_clip_count:
        raise DataError("同一 clip_id 被重复分配到多个 split")
    if stage == "frames" and split_contract_path.is_file():
        from handwash.io.utils import read_json

        payload = read_json(split_contract_path)
        if not isinstance(payload, dict) or payload.get("dataset") != name:
            raise DataError(
                f"劃分文件資料集與目前資料集不匹配：{split_contract_path}",
                hint="請對目前資料集重新執行 --stage split，再執行 --stage frames。",
            )
        saved_splits = payload.get("clip_splits", {})
        if not isinstance(saved_splits, dict) or set(saved_splits) != set(current_clips):
            raise DataError(
                f"掃描到的 clip 與已保存的劃分不一致：{split_contract_path}",
                hint="原始數據在 split 後發生變化；請重新執行 --stage split，再執行 --stage frames。",
            )
        if presplit_images:
            mismatches = [
                clip_id for clip_id, clip in current_clips.items()
                if saved_splits[clip_id] != clip.split
            ]
            if mismatches:
                raise DataError(
                    f"預置 split 目錄與劃分文件不一致：{mismatches[:5]}",
                    hint="確認原始 split 目錄未移動樣本；不要只改 manifest 的 split 欄位。",
                )
        else:
            split_map = {key: [] for key in ("train", "val", "test", "external")}
            for clip_id, clip in sorted(current_clips.items()):
                target_split = saved_splits[clip_id]
                if target_split not in split_map:
                    raise DataError(f"劃分文件中的 split 非法：{target_split!r}（clip={clip_id}）")
                split_map[target_split].append(replace(clip, split=target_split))
    elif stage == "frames" and name != "synthetic" and not presplit_images:
        raise DataError(
            "frames 階段找不到已保存的 clip 劃分",
            hint="先執行 `prepare_data.py --stage split`，或直接執行完整的 `--stage all`。",
        )
    elif stage in ("all", "split"):
        write_json(
            split_contract_path,
            {
                "dataset": name,
                "config_hash": rc.config_hash,
                "clip_splits": {
                    clip.clip_id: split_name
                    for split_name, clip_list in sorted(split_map.items())
                    for clip in clip_list
                },
            },
        )

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
    manifest_frames = prepared_frames if stage in ("all", "frames") else []
    if stage in ("all", "frames") and not manifest_frames:
        raise DataError(
            "準備階段沒有產生任何可用帧",
            hint="檢查視頻能否解碼、標注能否映射，並確認資料目錄與 frames_dir 配置正確。",
        )
    if manifest_frames:
        write_manifest(manifest_path, manifest_frames)

    # 区分原始标注/图像记录数与最终 manifest 采样数，避免把 annotation row
    # 数误报成已解码帧数。
    actual_by_split: dict[str, int] = defaultdict(int)
    for record in manifest_frames:
        actual_by_split[record.split] += 1
    actual_clips_by_split: dict[str, set[str]] = defaultdict(set)
    for record in manifest_frames:
        actual_clips_by_split[record.split].add(record.clip_id)
    for split_name, info in report.items():
        written = actual_by_split.get(split_name, 0)
        clips_written = len(actual_clips_by_split.get(split_name, set()))
        info["num_clips_in_manifest"] = clips_written
        info["num_clips_skipped"] = max(0, int(info["num_clips"]) - clips_written)
        info["num_frames_annotated"] = info["num_frames"]
        info["num_frames_in_manifest"] = written
        info["num_frames"] = written
        if written != info["num_frames_annotated"]:
            info["note"] = (
                f"源标注/图像记录 {info['num_frames_annotated']} 条，"
                f"最终写入 manifest {written} 帧（受采样上限或无效标签影响）"
            )
        if info["num_clips_skipped"]:
            log.warning(
                "split=%s 有 %d/%d 段 clip 沒有可用帧，%d 条标注/图像记录最终未写入 manifest",
                split_name,
                info["num_clips_skipped"],
                info["num_clips"],
                max(0, int(info["num_frames_annotated"]) - written),
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
                "num_frames_annotated 是适配器扫描到的标注/图像记录数；"
                "num_frames/num_frames_in_manifest 是最终登记的采样帧数。"
                "训练与评估用的是后者。"
            ),
        },
    )

    result = PrepareResult(
        manifest_path=manifest_path if manifest_frames else None,
        num_clips=sum(len(v) for v in split_map.values()),
        num_frames=len(manifest_frames),
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
        target = clip_split.get(rec.clip_id)
        if target is None:
            raise DataError(
                f"帧记录对应的 clip 没有 split：{rec.clip_id}",
                hint="clip 清单与帧清单必须来自同一次 prepare。",
            )
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
            except Exception as exc:
                log.error("抽帧失败，已跳过：%s（%s）", video_path, exc)
                continue

            if len(frames) < rc.dataset.prep.min_frames_per_clip:
                log.warning(
                    "视频过短（%d 帧 < %d），已跳过：%s",
                    len(frames), rc.dataset.prep.min_frames_per_clip, video_path.name,
                )
                continue

            for local_index, (source_index, stamp, frame) in enumerate(zip(indices, stamps, frames, strict=True)):
                label = _label_for_sample(clip, source_index, stamp)
                if label is None:
                    continue
                # 【关键】image_path 记录成**相对 frames_root** 的路径（`<clip_id>/00000.jpg`），
                # 与 data/dataset.py 的 _image_root_from() 约定一致：那边把 image_root
                # 解析为配置里的 frames_dir，再用 image_root / image_path 打开文件。
                # 早期版本这里写成 `frames/<clip_id>/...`（相对数据集 root），
                # 导致训练时报"帧图像不存在：<dataset.root>/frames/..."，
                # 而实际帧落在 data/processed/<dataset>/frames/ 下。
                image_ext = rc.dataset.prep.image_ext.lower().lstrip(".")
                relative = Path(clip.clip_id) / f"{local_index:05d}.{image_ext}"
                from PIL import Image, ImageOps

                height, width = rc.dataset.prep.resize_hw
                resized = ImageOps.contain(
                    Image.fromarray(frame),
                    (int(width), int(height)),
                    method=Image.Resampling.BILINEAR,
                )
                save_frame(
                    resized,
                    frames_root / clip.clip_id / f"{local_index:05d}.{image_ext}",
                    quality=rc.dataset.prep.jpeg_quality,
                )
                records.append(
                    FrameRecord(
                        clip_id=clip.clip_id,
                        frame_index=source_index,
                        image_path=str(relative).replace("\\", "/"),
                        label=label,
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
