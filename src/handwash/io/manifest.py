"""manifest 的读写与校验：帧清单是数据层与训练层之间的唯一凭据。

manifest 就是一张表，每行一个已落盘的帧图像：

    clip_id, dataset, split, video_path, frame_index, timestamp_s, image_path, label

不变量（由 ``validate_manifest`` 强制，任何一条不满足都直接报错）
------------------------------------------------------------------
  I1  ``image_path`` 去重后数量 == 行数（不允许同一张图重复登记）
  I2  同一 ``clip_id`` 只能属于一个 ``split``（防数据泄漏的**第一道闸**）
  I3  同一 ``clip_id`` 的 ``dataset`` 必须一致
  I4  ``frame_index`` 在 clip 内唯一
  I5  ``label`` 必须是规范 Step 值
  I6  ``split`` 必须是 train/val/test/external 之一

第二道闸在 ``core.split`` 的按视频划分；第三道闸是 config 的 split.guard_leakage。
把校验放在这里而不是训练脚本里，是为了让"错误数据"一定进不了训练。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from handwash.core.labels import Step
from handwash.core.schema import SPLITS, ClipRecord, FrameRecord, Split
from handwash.errors import DataLeakageError, ManifestError
from handwash.io.utils import read_csv, read_jsonl, write_csv
from handwash.logging import get_logger

__all__ = [
    "MANIFEST_COLUMNS",
    "write_manifest",
    "read_manifest",
    "validate_manifest",
    "manifest_summary",
    "clips_from_manifest",
    "filter_manifest",
]

log = get_logger(__name__)

#: 列顺序冻结：任何新增列都必须追加在末尾（否则旧脚本按位读取会错位）
MANIFEST_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "dataset",
    "split",
    "video_path",
    "frame_index",
    "timestamp_s",
    "image_path",
    "label",
)


def write_manifest(path: str | Path, records: Sequence[FrameRecord]) -> Path:
    """写 manifest（CSV）。写之前先校验，避免把坏数据落盘。"""
    validate_manifest(records)
    rows = [rec.to_dict() for rec in records]
    target = Path(path)
    write_csv(target, rows, fieldnames=MANIFEST_COLUMNS)
    log.info("manifest 已写入：%s（%d 行）", target, len(rows))
    return target


def read_manifest(path: str | Path, *, validate: bool = True) -> list[FrameRecord]:
    """读 manifest 并（默认）校验。

    支持 ``.csv`` 与 ``.jsonl`` 两种格式，由扩展名决定。
    """
    target = Path(path)
    if target.suffix.lower() in (".jsonl", ".ndjson"):
        rows = read_jsonl(target)
    else:
        rows = read_csv(target)

    if not rows:
        raise ManifestError(
            f"manifest 为空：{target}",
            hint="抽帧阶段可能全部失败；先检查原始视频是否可解码。",
        )

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing_cols:
        raise ManifestError(
            f"manifest 缺少必需列：{missing_cols}（文件：{target}）",
            hint=f"必需列为：{list(MANIFEST_COLUMNS)}；请用 handwash prepare 重新生成。",
        )

    records: list[FrameRecord] = []
    for lineno, row in enumerate(rows, start=2):  # 第 1 行是表头
        try:
            records.append(FrameRecord.from_dict(row))
        except (KeyError, ValueError) as exc:
            raise ManifestError(f"manifest 第 {lineno} 行无法解析（{target}）：{exc}") from exc

    if validate:
        validate_manifest(records)
    return records


def validate_manifest(records: Sequence[FrameRecord]) -> None:
    """强制上面列出的 I1–I6 六条不变量。"""
    if not records:
        raise ManifestError("manifest 记录数为 0")

    seen_images: set[str] = set()
    clip_split: dict[str, Split] = {}
    clip_dataset: dict[str, str] = {}
    clip_frame_index: dict[str, set[int]] = defaultdict(set)
    duplicated_clips: set[str] = set()

    for rec in records:
        if rec.image_path in seen_images:
            raise ManifestError(
                f"重复登记的帧图像：{rec.image_path}",
                hint="同一张图只能属于一个 clip/标签；通常是抽帧时 clip_id 生成规则有冲突。",
            )
        seen_images.add(rec.image_path)

        if rec.split not in SPLITS:
            raise ManifestError(f"非法 split：{rec.split!r}，允许 {list(SPLITS)}")

        prev = clip_split.get(rec.clip_id)
        if prev is not None and prev != rec.split:
            duplicated_clips.add(rec.clip_id)
        clip_split[rec.clip_id] = rec.split

        ds = clip_dataset.get(rec.clip_id)
        if ds is not None and ds != rec.dataset:
            raise ManifestError(
                f"同一 clip_id={rec.clip_id} 的 dataset 不一致：{ds} vs {rec.dataset}"
            )
        clip_dataset[rec.clip_id] = rec.dataset

        if rec.frame_index in clip_frame_index[rec.clip_id]:
            raise ManifestError(f"clip={rec.clip_id} 的 frame_index={rec.frame_index} 重复")
        clip_frame_index[rec.clip_id].add(rec.frame_index)

    if duplicated_clips:
        # 这是最严重的错误：直接抛专用异常，训练脚本不捕获，进程终止
        raise DataLeakageError(sorted(duplicated_clips))


def manifest_summary(records: Sequence[FrameRecord]) -> dict[str, object]:
    """统计摘要：写进日志与 outputs/<run>/manifest_summary.json，方便写报告。"""
    by_split: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    clips_by_split: dict[str, set[str]] = defaultdict(set)
    datasets: set[str] = set()

    for rec in records:
        by_split[rec.split] += 1
        by_label[rec.label.value] += 1
        clips_by_split[rec.split].add(rec.clip_id)
        datasets.add(rec.dataset)

    return {
        "num_frames": len(records),
        "num_clips": len({r.clip_id for r in records}),
        "datasets": sorted(datasets),
        "frames_per_split": dict(by_split),
        "clips_per_split": {k: len(v) for k, v in clips_by_split.items()},
        "frames_per_label": dict(sorted(by_label.items())),
    }


def clips_from_manifest(records: Sequence[FrameRecord]) -> list[ClipRecord]:
    """从帧级 manifest 反推 clip 级清单（每段视频一条记录）。"""
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.clip_id].append(rec)

    clips: list[ClipRecord] = []
    for clip_id, recs in sorted(grouped.items()):
        recs_sorted = sorted(recs, key=lambda r: r.frame_index)
        fps_guess = _infer_fps(recs_sorted)
        duration = (
            recs_sorted[-1].timestamp_s - recs_sorted[0].timestamp_s if recs_sorted[0].timestamp_s is not None else None
        )
        clips.append(
            ClipRecord(
                clip_id=clip_id,
                dataset=recs_sorted[0].dataset,
                split=recs_sorted[0].split,
                video_path=recs_sorted[0].video_path,
                frame_count=len(recs_sorted),
                fps=fps_guess,
                duration_s=duration,
                label_sequence=tuple(r.label for r in recs_sorted),
                metadata={"image_dir": str(Path(recs_sorted[0].image_path).parent)},
            )
        )
    return clips


def _infer_fps(records: Sequence[FrameRecord]) -> float | None:
    """由 timestamp 序列推断抽帧后的有效帧率（用于时长统计）。"""
    stamps = [r.timestamp_s for r in records if r.timestamp_s is not None]
    if len(stamps) < 2:
        return None
    deltas = [b - a for a, b in zip(stamps[:-1], stamps[1:], strict=True) if b > a]
    if not deltas:
        return None
    return round(1.0 / (sum(deltas) / len(deltas)), 6)


def filter_manifest(
    records: Sequence[FrameRecord],
    *,
    split: str | None = None,
    dataset: str | None = None,
    labels: Iterable[Step] | None = None,
) -> list[FrameRecord]:
    """按条件筛出子集（评估脚本按 split 取数据时用）。"""
    wanted = set(labels) if labels is not None else None
    return [
        rec
        for rec in records
        if (split is None or rec.split == split)
        and (dataset is None or rec.dataset == dataset)
        and (wanted is None or rec.label in wanted)
    ]
