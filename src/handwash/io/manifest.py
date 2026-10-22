"""Read and validate frame manifests while enforcing split integrity."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import median

from handwash.core.labels import Step
from handwash.core.schema import SPLITS, ClipRecord, FrameRecord, Split
from handwash.errors import DataLeakageError, ManifestError
from handwash.io.utils import read_csv, read_jsonl, write_csv
from handwash.logging import get_logger

__all__ = [
    "MANIFEST_COLUMNS",
    "clips_from_manifest",
    "filter_manifest",
    "manifest_summary",
    "read_manifest",
    "validate_manifest",
    "write_manifest",
]

log = get_logger(__name__)


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

    validate_manifest(records)
    rows = [rec.to_dict() for rec in records]
    target = Path(path)
    write_csv(target, rows, fieldnames=MANIFEST_COLUMNS)
    log.info("Manifest written: %s (%d rows)", target, len(rows))
    return target


def read_manifest(path: str | Path, *, validate: bool = True) -> list[FrameRecord]:

    target = Path(path)
    if target.suffix.lower() in (".jsonl", ".ndjson"):
        rows = read_jsonl(target)
    else:
        rows = read_csv(target)

    if not rows:
        raise ManifestError(
            f"Empty manifest: {target}",
            hint="Frame extraction may have failed; check whether the original videos can be decoded.",
        )

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing_cols:
        raise ManifestError(
            f"Manifest is missing required columns: {missing_cols}(file: {target})",
            hint=f"Required columns: {list(MANIFEST_COLUMNS)}; regenerate the prepared data.",
        )

    records: list[FrameRecord] = []
    for lineno, row in enumerate(rows, start=2):
        try:
            records.append(FrameRecord.from_dict(row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Manifest row {lineno} could not be parsed ({target}): {exc}") from exc

    if validate:
        validate_manifest(records)
    return records


def validate_manifest(records: Sequence[FrameRecord]) -> None:

    if not records:
        raise ManifestError("The manifest has no records")

    seen_images: set[str] = set()
    clip_split: dict[str, Split] = {}
    clip_dataset: dict[str, str] = {}
    clip_frame_index: dict[str, set[int]] = defaultdict(set)
    duplicated_clips: set[str] = set()

    for rec in records:
        for field_name, value in (
            ("clip_id", rec.clip_id),
            ("dataset", rec.dataset),
            ("image_path", rec.image_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"In the manifest, {field_name} must not be empty")
            if value != value.strip():
                raise ManifestError(
                    f"In the manifest, {field_name} has leading or trailing whitespace: {value!r}"
                )
        if not isinstance(rec.label, Step):
            raise ManifestError(f"clip={rec.clip_id} label is not a canonical Step: {rec.label!r}")
        if isinstance(rec.frame_index, bool) or not isinstance(rec.frame_index, int) or rec.frame_index < 0:
            raise ManifestError(
                f"clip={rec.clip_id} frame_index must be a non-negative integer: {rec.frame_index!r}"
            )
        if rec.timestamp_s is not None and (
            not isinstance(rec.timestamp_s, (int, float))
            or not math.isfinite(float(rec.timestamp_s))
            or rec.timestamp_s < 0
        ):
            raise ManifestError(
                f"clip={rec.clip_id} timestamp_s must be finite non-negative seconds: {rec.timestamp_s!r}"
            )
        if rec.image_path in seen_images:
            raise ManifestError(
                f"Duplicate frame image record: {rec.image_path}",
                hint="Each image must belong to one clip and label; check clip_id generation for collisions.",
            )
        seen_images.add(rec.image_path)

        if rec.split not in SPLITS:
            raise ManifestError(f"Invalid split: {rec.split!r}; allowed {list(SPLITS)}")

        prev = clip_split.get(rec.clip_id)
        if prev is not None and prev != rec.split:
            duplicated_clips.add(rec.clip_id)
        clip_split[rec.clip_id] = rec.split

        ds = clip_dataset.get(rec.clip_id)
        if ds is not None and ds != rec.dataset:
            raise ManifestError(
                f"Same clip_id={rec.clip_id} has inconsistent dataset values: {ds} vs {rec.dataset}"
            )
        clip_dataset[rec.clip_id] = rec.dataset

        if rec.frame_index in clip_frame_index[rec.clip_id]:
            raise ManifestError(f"clip={rec.clip_id} frame_index={rec.frame_index} duplicate")
        clip_frame_index[rec.clip_id].add(rec.frame_index)

    if duplicated_clips:
        raise DataLeakageError(sorted(duplicated_clips))


def manifest_summary(records: Sequence[FrameRecord]) -> dict[str, object]:

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

    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.clip_id].append(rec)

    clips: list[ClipRecord] = []
    for clip_id, recs in sorted(grouped.items()):
        recs_sorted = sorted(recs, key=lambda r: r.frame_index)
        fps_guess = _infer_fps(recs_sorted)
        valid_times = [float(record.timestamp_s) for record in recs_sorted if record.timestamp_s is not None]
        duration = (
            max(valid_times) - min(valid_times) + (1.0 / fps_guess)
            if len(valid_times) >= 2 and fps_guess
            else None
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

    timed = [
        (position, float(record.timestamp_s))
        for position, record in enumerate(records)
        if record.timestamp_s is not None
    ]
    if len(timed) < 2:
        return None
    seconds_per_frame = [
        (right_time - left_time) / (right_position - left_position)
        for (left_position, left_time), (right_position, right_time) in zip(
            timed[:-1], timed[1:], strict=True
        )
        if right_position > left_position and right_time > left_time
    ]
    if not seconds_per_frame:
        return None
    return round(1.0 / float(median(seconds_per_frame)), 6)


def filter_manifest(
    records: Sequence[FrameRecord],
    *,
    split: str | None = None,
    dataset: str | None = None,
    labels: Iterable[Step] | None = None,
) -> list[FrameRecord]:

    wanted = set(labels) if labels is not None else None
    return [
        rec
        for rec in records
        if (split is None or rec.split == split)
        and (dataset is None or rec.dataset == dataset)
        and (wanted is None or rec.label in wanted)
    ]
