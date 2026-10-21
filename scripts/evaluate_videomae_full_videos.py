#!/usr/bin/env python
"""Run the trained VideoMAE model over complete PSKUS test videos.

The model predicts overlapping 16-frame windows.  Window probabilities are
smoothed, converted into a continuous action timeline, and evaluated at each
window centre against the two-annotator consensus label.  The original videos
are split before inference using the same deterministic split as training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT
else:
    from ._bootstrap import PROJECT_ROOT

# ``_bootstrap`` puts the repository root ahead of ``scripts/`` on sys.path.
# Some cloud checkouts also contain a legacy top-level
# ``train_videomae_pskus.py`` with an older helper signature, so explicitly
# prefer the companion script beside this evaluator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.train_videomae_pskus import LABELS, VideoInfo, _discover_videos, _read_csv


@dataclass(frozen=True)
class TimelineSegment:
    start_s: float
    end_s: float
    label: str
    confidence: float


def _window_starts(sample_count: int, window: int, stride: int) -> list[int]:
    if sample_count < window:
        return []
    starts = list(range(0, sample_count - window + 1, stride))
    tail = sample_count - window
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def _decode_sampled_frames(video: VideoInfo, *, sample_fps: float) -> tuple[list[np.ndarray], list[int], float]:
    import cv2

    cap = cv2.VideoCapture(str(video.path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_stride = max(1, round(fps / sample_fps))
    frames: list[np.ndarray] = []
    frame_ids: list[int] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % source_stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            frame_ids.append(index)
        index += 1
    cap.release()
    if frame_count and abs(index - frame_count) > 2:
        print(f"警告：{video.stem} 元数据 {frame_count} 帧，实际解码 {index} 帧", flush=True)
    return frames, frame_ids, fps


def _predict_windows(model, frames: list[np.ndarray], starts: list[int], *, num_frames: int,
                     batch_size: int, mean: tuple[float, ...], std: tuple[float, ...], device: str):
    import torch

    outputs: list[np.ndarray] = []
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1)
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset: offset + batch_size]
        arrays = np.stack([
            np.stack(frames[start: start + num_frames], axis=0)
            for start in batch_starts
        ], axis=0)
        values = torch.from_numpy(arrays).permute(0, 1, 4, 2, 3).float().div_(255.0)
        values = (values - mean_t) / std_t
        values = values.to(device, non_blocking=True)
        amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"
        ):
            probs = model(pixel_values=values).logits.softmax(dim=-1)
        outputs.append(probs.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _smooth_probs(probs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return probs
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = np.empty_like(probs)
    for index in range(len(probs)):
        out[index] = probs[max(0, index - radius): min(len(probs), index + radius + 1)].mean(axis=0)
    return out


def _timeline(labels: np.ndarray, confidences: np.ndarray, starts: list[int], frame_ids: list[int],
              *, num_frames: int, fps: float) -> list[TimelineSegment]:
    centres = [frame_ids[start + num_frames // 2] / fps for start in starts]
    left = frame_ids[starts[0]] / fps
    right = (frame_ids[-1] + 1) / fps
    boundaries = [left]
    boundaries.extend((centres[i - 1] + centres[i]) / 2.0 for i in range(1, len(centres)))
    boundaries.append(right)
    segments: list[TimelineSegment] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append(TimelineSegment(
                start_s=round(boundaries[start], 3),
                end_s=round(boundaries[index], 3),
                label=LABELS[int(labels[start])],
                confidence=round(float(confidences[start:index].mean()), 4),
            ))
            start = index
    return segments


def _collapse_steps(labels: np.ndarray) -> list[str]:
    sequence: list[str] = []
    for value in labels.tolist():
        if value == 0:
            continue
        label = LABELS[int(value)]
        if not sequence or sequence[-1] != label:
            sequence.append(label)
    return sequence


def _ground_truth(video: VideoInfo, centre_frame_ids: list[int]) -> tuple[list[int], list[bool]]:
    left, left_times = _read_csv(video.primary)
    right, right_times = _read_csv(video.secondary)
    max_index = min(len(left), len(right), len(left_times), len(right_times))
    targets: list[int] = []
    usable: list[bool] = []
    for frame_id in centre_frame_ids:
        valid = (
            frame_id < max_index
            and abs(left_times[frame_id] - right_times[frame_id]) <= 0.001
            and left[frame_id] >= 0
            and left[frame_id] == right[frame_id]
        )
        usable.append(valid)
        targets.append(left[frame_id] if valid else -1)
    return targets, usable


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "outputs/videomae_pskus_autodl/best_model")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/videomae_full_video_test")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--smooth-windows", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=0, help="仅测试前 N 段视频；0 表示完整测试集")
    parser.add_argument("--full-pskus", action="store_true", help="按完整 11 分片的视频划分评估")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    videos = [video for video in _discover_videos(args.root, full_pskus=args.full_pskus)
              if video.split == args.split]
    if args.limit > 0:
        videos = videos[:args.limit]
    if not videos:
        parser.error("没有发现测试视频")
    args.output.mkdir(parents=True, exist_ok=True)
    timeline_dir = args.output / "timelines"
    timeline_dir.mkdir(exist_ok=True)

    processor = VideoMAEImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(args.model, local_files_only=True).to(device).eval()
    mean = tuple(float(v) for v in processor.image_mean)
    std = tuple(float(v) for v in processor.image_std)

    all_targets: list[int] = []
    all_predictions: list[int] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        video_started = time.perf_counter()
        frames, frame_ids, fps = _decode_sampled_frames(video, sample_fps=args.sample_fps)
        starts = _window_starts(len(frames), args.num_frames, args.window_stride)
        if not starts:
            print(f"跳过过短视频：{video.source}/{video.stem}", flush=True)
            continue
        probs = _predict_windows(
            model, frames, starts, num_frames=args.num_frames, batch_size=args.batch_size,
            mean=mean, std=std, device=device,
        )
        probs = _smooth_probs(probs, args.smooth_windows)
        predictions = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        centre_ids = [frame_ids[start + args.num_frames // 2] for start in starts]
        targets, usable = _ground_truth(video, centre_ids)
        y_true = [target for target, keep in zip(targets, usable, strict=True) if keep]
        y_pred = [int(pred) for pred, keep in zip(predictions, usable, strict=True) if keep]
        all_targets.extend(y_true)
        all_predictions.extend(y_pred)
        accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
        macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0) if y_true else 0.0
        segments = _timeline(predictions, confidences, starts, frame_ids, num_frames=args.num_frames, fps=fps)
        payload = {
            "video_id": f"{video.source}/{video.stem}",
            "video_path": str(video.path),
            "fps": fps,
            "sample_fps": args.sample_fps,
            "window_count": len(starts),
            "evaluated_consensus_centres": len(y_true),
            "accuracy": accuracy,
            "macro_f1_all_7_classes": macro_f1,
            "predicted_step_sequence": _collapse_steps(predictions),
            "ground_truth_step_sequence_at_centres": _collapse_steps(np.asarray(y_true, dtype=np.int64)),
            "timeline": [asdict(segment) for segment in segments],
        }
        (timeline_dir / f"{video.source}__{video.stem}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows.append({
            "video_id": payload["video_id"],
            "windows": len(starts),
            "consensus_centres": len(y_true),
            "accuracy": round(float(accuracy), 6),
            "macro_f1_all_7_classes": round(float(macro_f1), 6),
            "predicted_sequence": " > ".join(payload["predicted_step_sequence"]),
            "ground_truth_sequence": " > ".join(payload["ground_truth_step_sequence_at_centres"]),
            "runtime_s": round(time.perf_counter() - video_started, 3),
        })
        print(
            f"[{number}/{len(videos)}] {payload['video_id']}：{len(starts)} 窗口，"
            f"accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}", flush=True,
        )

    labels = list(range(len(LABELS)))
    summary = {
        "model": str(args.model),
        "split": args.split,
        "test_videos_requested": len(videos),
        "test_videos_evaluated": len(rows),
        "evaluated_window_centres": len(all_targets),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, labels=labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            all_targets, all_predictions, labels=labels, target_names=list(LABELS),
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_targets, all_predictions, labels=labels).tolist(),
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "window_stride_sampled_frames": args.window_stride,
        "probability_smoothing_windows": args.smooth_windows,
        "elapsed_s": time.perf_counter() - started,
        "evaluation_note": "Metrics use each window centre where both annotators agree; windows crossing action boundaries remain included.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(args.output / "per_video.csv", rows)
    print(
        f"完成：{len(rows)} 段视频，{len(all_targets)} 个双人一致中心点，"
        f"accuracy={summary['accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
