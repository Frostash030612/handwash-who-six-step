#!/usr/bin/env python
"""Build a seven-class Ultralytics image dataset from all PSKUS partitions.

Uses the same video split and two-annotator rule as --full-pskus VideoMAE.
Only stable consensus frames enter training; val/test keep the natural class
distribution. Train keeps every washing-step frame and caps only ``other``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from scripts.train_videomae_pskus import _discover_videos, _preserve_video_splits, _read_csv

CLASSES = (
    "0_other_or_nonstandard",
    "1_palm_to_palm",
    "2_palm_over_dorsum",
    "3_palm_to_palm_interlaced",
    "4_backs_of_fingers",
    "5_thumb_rubbing",
    "6_fingertips_to_palm",
)


def _eligible(video, sample_fps: float, radius: int) -> list[tuple[int, int]]:
    import cv2

    left, left_times = _read_csv(video.primary)
    right, right_times = _read_csv(video.secondary)
    if not left or len(left) != len(right) or len(left_times) != len(right_times):
        return []
    if any(not (a == b and abs(a - b) <= 0.001) for a, b in zip(left_times, right_times)):
        return []
    cap = cv2.VideoCapture(str(video.path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(left))
    cap.release()
    stride = max(1, round(fps / sample_fps))
    max_index = min(len(left), frame_count)
    agreed = [a >= 0 and a == b for a, b in zip(left[:max_index], right[:max_index])]
    result = []
    for i in range(0, max_index, stride):
        lo, hi = max(0, i - radius), min(max_index, i + radius + 1)
        label = left[i]
        if agreed[i] and all(agreed[j] and left[j] == label for j in range(lo, hi)):
            result.append((i, label))
    return result


def build(root: Path, output: Path, sample_fps: float, radius: int,
          other_to_wash_ratio: float, datasets: list[str] | None = None,
          split_reference: Path | None = None) -> dict:
    import cv2

    source_dirs = {p.name for p in root.glob("DataSet*") if p.is_dir()}
    expected = {f"DataSet{i}" for i in range(1, 12)}
    if datasets is None:
        if source_dirs != expected:
            raise SystemExit(f"PSKUS 分片不完整：缺少 {sorted(expected - source_dirs)}；多出 {sorted(source_dirs - expected)}")
        selected_sources = expected
    else:
        selected_sources = set(datasets)
        unknown = selected_sources - expected
        missing = selected_sources - source_dirs
        if unknown or missing:
            raise SystemExit(f"指定分片不可用：未知 {sorted(unknown)}；未解压 {sorted(missing)}")
    videos = [video for video in _discover_videos(root, full_pskus=True)
              if video.source in selected_sources]
    if not videos:
        raise SystemExit("未找到可用双标注视频")
    if split_reference is not None:
        videos = _preserve_video_splits(videos, split_reference)
    summary_path = output / "summary.json"
    marker = output / ".building"
    if summary_path.exists() and not marker.exists():
        raise SystemExit(f"数据集已完成：{output}")
    output.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)

    candidates = {}
    counts_before = {split: Counter() for split in ("train", "val", "test")}
    skipped_alignment = 0
    for number, video in enumerate(videos, 1):
        rows = _eligible(video, sample_fps, radius)
        if not rows:
            skipped_alignment += 1
        key = f"{video.source}/{video.stem}"
        candidates[key] = rows
        counts_before[video.split].update(label for _, label in rows)
        if number % 250 == 0:
            print(f"标注扫描 {number}/{len(videos)}", flush=True)

    wash_count = sum(counts_before["train"][i] for i in range(1, 7))
    other_budget = round(wash_count * other_to_wash_ratio)
    train_others = [
        (hashlib.sha256(f"{video.source}/{video.stem}:{frame_id}".encode()).digest(),
         f"{video.source}/{video.stem}", frame_id)
        for video in videos if video.split == "train"
        for frame_id, label in candidates[f"{video.source}/{video.stem}"] if label == 0
    ]
    train_others.sort(key=lambda row: row[0])
    keep_other = {(key, frame_id) for _, key, frame_id in train_others[:other_budget]}
    del train_others

    estimated_frames = (sum(counts_before["train"][i] for i in range(1, 7))
                        + min(counts_before["train"][0], other_budget)
                        + counts_before["val"].total() + counts_before["test"].total())
    estimated = estimated_frames * 20000
    free = shutil.disk_usage(output).free
    if free < estimated + 2 * 1024**3:
        raise SystemExit(f"剩余空间可能不足：约需 {estimated / 1024**3:.1f} GiB 图像及 2 GiB 余量，"
                         f"当前可用 {free / 1024**3:.1f} GiB")

    counts_after = {split: Counter() for split in ("train", "val", "test")}
    for split in counts_after:
        for name in CLASSES:
            (output / "ultralytics" / split / name).mkdir(parents=True, exist_ok=True)
    split_rows = []
    for number, video in enumerate(videos, 1):
        key = f"{video.source}/{video.stem}"
        chosen = [(frame_id, label) for frame_id, label in candidates[key]
                  if video.split != "train" or label != 0 or (key, frame_id) in keep_other]
        split_rows.append({"video_id": key, "split": video.split,
                           "primary": video.primary.parent.name, "secondary": video.secondary.parent.name,
                           "stable_consensus_frames": len(candidates[key]), "selected_frames": len(chosen)})
        if not chosen:
            continue
        wanted = {frame_id: label for frame_id, label in chosen}
        cap = cv2.VideoCapture(str(video.path))
        index = 0
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            label = wanted.pop(index, None)
            if label is not None:
                # The source mixes 320x240 and 640x480. Keep the same aspect
                # ratio but cap larger frames to 320x240 for predictable disk
                # use and the native scale of the existing exp.pt baseline.
                if frame.shape[1] > 320 or frame.shape[0] > 240:
                    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                name = f"{video.source}_{video.stem}_{index:07d}.jpg"
                target = output / "ultralytics" / video.split / CLASSES[label] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file() or target.stat().st_size == 0:
                    ok_jpg, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok_jpg:
                        raise RuntimeError(f"编码失败：{key} frame {index}")
                    temporary = target.with_suffix(".jpg.part")
                    temporary.write_bytes(encoded.tobytes())
                    temporary.replace(target)
                counts_after[video.split][label] += 1
            index += 1
        cap.release()
        if wanted:
            raise RuntimeError(f"视频提前结束：{key}，缺少 {len(wanted)} 帧")
        if number % 100 == 0:
            print(f"抽帧 {number}/{len(videos)}，当前 {sum(x.total() for x in counts_after.values())} 张", flush=True)

    with (output / "video_splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    summary = {
        "source": "PSKUS Zenodo record 4537209",
        "source_datasets": sorted(selected_sources),
        "videos_with_two_annotations": len(videos),
        "video_split": dict(Counter(v.split for v in videos)),
        "label_rule": f"two aligned annotators agree; +/-{radius} original frames stable; 0/7/non-washing -> other",
        "pair_rule": "DataSet1=2/3; DataSet4=1/2; others=1/2 then first available pair",
        "split_rule": ("preserve referenced video splits; hash split new videos 70/15/15"
                       if split_reference is not None else
                       "preserve prior DataSet1+4 split; hash split remaining videos 70/15/15"),
        "split_reference": str(split_reference) if split_reference is not None else None,
        "split_reference_sha256": (hashlib.sha256(split_reference.read_bytes()).hexdigest()
                                   if split_reference is not None else None),
        "sample_fps": sample_fps,
        "stable_radius_frames": radius,
        "other_to_wash_ratio_train_max": other_to_wash_ratio,
        "skipped_or_no_stable_frames": skipped_alignment,
        "classes": list(CLASSES),
        "counts_before_train_other_cap": {s: dict(c) for s, c in counts_before.items()},
        "selected_counts": {s: {CLASSES[i]: c[i] for i in range(7)} for s, c in counts_after.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker.unlink()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/pskuss_full_yolo")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stable-radius", type=int, default=7)
    parser.add_argument("--other-to-wash-ratio", type=float, default=1.0)
    parser.add_argument("--datasets", default=None,
                        help="可选：仅使用逗号分隔的分片子集，如 DataSet1,DataSet3,DataSet4；默认要求全部 11 个")
    parser.add_argument("--split-reference", type=Path,
                        help="续训时保留既有 video_splits.csv 中所有视频的划分，新视频单独划分")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.stable_radius < 0 or args.other_to_wash_ratio < 0:
        parser.error("采样率须大于零，稳定半径与 other 比例须非负")
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()] if args.datasets else None
    result = build(args.root, args.output, args.sample_fps, args.stable_radius,
                   args.other_to_wash_ratio, datasets=datasets,
                   split_reference=args.split_reference)
    print(f"完成：{result['videos_with_two_annotations']} 段视频；各划分 {result['video_split']}")
    print(result["selected_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
