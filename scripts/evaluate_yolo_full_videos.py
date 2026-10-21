#!/usr/bin/env python
"""Evaluate exp.pt on the exact original-video split and centre timestamps used by VideoMAE.

Tune short probability smoothing and minimum segment duration on validation videos,
then freeze those settings before reporting the held-out test videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__ in (None, ""):
    from _bootstrap import PROJECT_ROOT
else:
    from ._bootstrap import PROJECT_ROOT
from scripts.evaluate_videomae_full_videos import _ground_truth, _window_starts
from scripts.train_videomae_pskus import LABELS, VideoInfo, _discover_videos


def decode(video: VideoInfo, sample_fps: float) -> tuple[list[np.ndarray], list[int], float]:
    import cv2

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video.path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, round(fps / sample_fps))
    frames, indices = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)  # Ultralytics expects OpenCV BGR arrays.
            indices.append(index)
        index += 1
    cap.release()
    return frames, indices, fps


def predict(model, frames: list[np.ndarray], imgsz: int, device: str, batch: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(frames), batch):
        results = model.predict(frames[start:start + batch], imgsz=imgsz, device=device,
                                batch=batch, verbose=False)
        chunks.append(np.stack([r.probs.data.cpu().numpy() for r in results]))
    return np.concatenate(chunks).astype(np.float32)


def _average_probs(probs: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return probs
    radius = width // 2
    return np.stack([probs[max(0, i - radius):min(len(probs), i + radius + 1)].mean(0)
                     for i in range(len(probs))])


def _suppress_short_runs(labels: np.ndarray, averaged: np.ndarray, min_run: int) -> np.ndarray:
    labels = labels.copy()
    if len(labels) == 0 or min_run <= 1:
        return labels
    for _ in range(3):
        runs = []
        begin = 0
        for end in range(1, len(labels) + 1):
            if end == len(labels) or labels[end] != labels[begin]:
                runs.append((begin, end, int(labels[begin])))
                begin = end
        changed = False
        for i, (begin, end, label) in enumerate(runs):
            if end - begin >= min_run or len(runs) == 1:
                continue
            candidates = []
            if i:
                candidates.append(runs[i - 1][2])
            if i + 1 < len(runs):
                candidates.append(runs[i + 1][2])
            if candidates:
                replacement = max(candidates, key=lambda c: float(averaged[begin:end, c].mean()))
                if replacement != label:
                    labels[begin:end] = replacement
                    changed = True
        if not changed:
            break
    return labels


def smooth(probs: np.ndarray, width: int, min_run: int) -> np.ndarray:
    """Centered probability average, then suppress only transient label islands.

    For offline video, a short island is relabelled using the more likely of its
    neighbours. This allows a confirmed new action to start at its first frame,
    without the permanent delay of a causal majority vote.
    """
    averaged = _average_probs(probs, width)
    labels = averaged.argmax(1).astype(np.int64)
    return _suppress_short_runs(labels, averaged, min_run)


def smooth_hysteresis(probs: np.ndarray, width: int, confirm_frames: int,
                      margin: float, min_run: int = 1) -> np.ndarray:
    """Switch actions only after a confident candidate persists for N windows.

    This is an offline smoother: once a switch is confirmed, its boundary is
    backdated to the candidate's first window, so confirmation adds no timeline
    delay. It never forces WHO order; a real out-of-order action remains visible.
    """
    if len(probs) == 0:
        return np.asarray([], dtype=np.int64)
    if confirm_frames < 1 or margin < 0:
        raise ValueError("confirm_frames must be positive and margin non-negative")
    averaged = _average_probs(probs, width)
    raw = averaged.argmax(1).astype(np.int64)
    labels = np.empty_like(raw)
    state = int(raw[0])
    labels[0] = state
    pending_label: int | None = None
    pending_start = 0

    for index in range(1, len(raw)):
        candidate = int(raw[index])
        if candidate == state:
            labels[index] = state
            pending_label = None
            continue
        if candidate != pending_label:
            pending_label = candidate
            pending_start = index
        labels[index] = state
        count = index - pending_start + 1
        advantage = float(np.mean(averaged[pending_start:index + 1, candidate]
                                  - averaged[pending_start:index + 1, state]))
        if count >= confirm_frames and advantage >= margin:
            labels[pending_start:index + 1] = candidate
            state = candidate
            pending_label = None

    return _suppress_short_runs(labels, averaged, min_run)


def metrics(items: list[dict], width: int, min_run: int, *, method: str = "mean_min_run",
            confirm_frames: int = 2, margin: float = 0.05) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    truth, pred, transition_truth, transition_pred = [], [], [], []
    per_video = []
    for item in items:
        labels = (smooth_hysteresis(item["probs"], width, confirm_frames, margin, min_run)
                  if method == "hysteresis" else smooth(item["probs"], width, min_run))
        centre_labels = labels[item["centres"]]
        keep = item["usable"]
        target = item["targets"][keep]
        prediction = centre_labels[keep]
        truth.extend(target.tolist())
        pred.extend(prediction.tolist())
        # A transition centre is at or one centre (about 0.8 s) from a
        # consensus-label change. Ignore changes across missing annotations.
        near_transition = np.zeros(len(keep), dtype=bool)
        for i in range(1, len(keep)):
            gap_s = (item["frame_ids"][item["centres"][i]] -
                     item["frame_ids"][item["centres"][i - 1]]) / item["fps"]
            if keep[i - 1] and keep[i] and item["targets"][i - 1] != item["targets"][i] and gap_s <= 1.6:
                near_transition[max(0, i - 2):min(len(keep), i + 2)] = True
        boundary_mask = near_transition & keep
        transition_truth.extend(item["targets"][boundary_mask].tolist())
        transition_pred.extend(centre_labels[boundary_mask].tolist())
        per_video.append({"video_id": item["video_id"], "evaluated_centres": len(target),
                          "accuracy": float(accuracy_score(target, prediction)) if len(target) else None,
                          "macro_f1": float(f1_score(target, prediction, labels=list(range(7)),
                                                     average="macro", zero_division=0)) if len(target) else None})
    return {"videos": len(items), "evaluated_centres": len(truth),
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=list(range(7)), average="macro", zero_division=0)),
            "transition_centres": len(transition_truth),
            "transition_accuracy": float(accuracy_score(transition_truth, transition_pred)) if transition_truth else None,
            "transition_macro_f1": float(f1_score(transition_truth, transition_pred, labels=list(range(7)),
                                               average="macro", zero_division=0)) if transition_truth else None,
            "truth": truth, "predictions": pred, "per_video": per_video}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "exp.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/yolo_full_video_test")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full-pskus", action="store_true", help="按完整 11 分片的视频划分评估")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import classification_report, confusion_matrix
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    expected = ("0_other_or_nonstandard", "1_palm_to_palm", "2_palm_over_dorsum",
                "3_palm_to_palm_interlaced", "4_backs_of_fingers", "5_thumb_rubbing",
                "6_fingertips_to_palm")
    names = tuple(model.names[i] for i in range(7))
    if model.task != "classify" or names != expected:
        parser.error(f"权重任务或类别顺序不符：{model.task}, {names}")
    imgsz = int(model.overrides.get("imgsz", 320))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    videos = [v for v in _discover_videos(args.root, full_pskus=args.full_pskus)
              if v.split in ("val", "test")]
    if args.limit:
        videos = [*([v for v in videos if v.split == "val"][:args.limit]),
                  *([v for v in videos if v.split == "test"][:args.limit])]
    items = {"val": [], "test": []}
    started = time.perf_counter()
    for number, video in enumerate(videos, 1):
        frames, frame_ids, fps = decode(video, args.sample_fps)
        starts = _window_starts(len(frames), args.window_frames, args.window_stride)
        if not starts:
            continue
        centres = np.asarray([start + args.window_frames // 2 for start in starts], dtype=np.int64)
        targets, usable = _ground_truth(video, [frame_ids[i] for i in centres])
        probs = predict(model, frames, imgsz, device, args.batch)
        item = {"video_id": f"{video.source}/{video.stem}", "probs": probs, "centres": centres,
                "targets": np.asarray(targets, dtype=np.int64), "usable": np.asarray(usable, dtype=bool),
                "frame_ids": frame_ids, "fps": fps}
        items[video.split].append(item)
        print(f"[{number}/{len(videos)}] {video.split} {item['video_id']}: "
              f"{len(frames)} sampled frames, {sum(usable)} consensus centres", flush=True)
    if not items["val"] or not items["test"]:
        parser.error("验证集或测试集为空")

    grid = []
    for width in (1, 3, 5):
        for min_run in (1, 2, 3):
            result = metrics(items["val"], width, min_run)
            grid.append({"method": "mean_min_run", "width": width, "min_run": min_run,
                         "accuracy": result["accuracy"],
                         "macro_f1": result["macro_f1"]})
        for confirm_frames in (2, 3):
            for margin in (0.0, 0.05):
                result = metrics(items["val"], width, 1, method="hysteresis",
                                 confirm_frames=confirm_frames, margin=margin)
                grid.append({"method": "hysteresis", "width": width, "min_run": 1,
                             "confirm_frames": confirm_frames, "margin": margin,
                             "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    best = max(grid, key=lambda row: (row["macro_f1"], row["accuracy"], -row["width"], -row["min_run"]))
    raw = metrics(items["test"], 1, 1)
    selected = metrics(items["test"], best["width"], best["min_run"], method=best["method"],
                       confirm_frames=best.get("confirm_frames", 2), margin=best.get("margin", 0.05))
    labels = list(range(7))
    summary = {"model": str(args.model), "model_names": names, "image_size": imgsz,
               "sample_fps": args.sample_fps, "comparison_window_frames": args.window_frames,
               "comparison_window_stride": args.window_stride,
               "validation_grid": grid, "chosen_on_validation": best,
               "test_raw": {k: raw[k] for k in ("videos", "evaluated_centres", "accuracy", "macro_f1",
                                                  "transition_centres", "transition_accuracy", "transition_macro_f1")},
               "test_smoothed": {k: selected[k] for k in ("videos", "evaluated_centres", "accuracy", "macro_f1",
                                                       "transition_centres", "transition_accuracy", "transition_macro_f1")},
               "classification_report": classification_report(selected["truth"], selected["predictions"],
                   labels=labels, target_names=LABELS, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(selected["truth"], selected["predictions"], labels=labels).tolist(),
               "elapsed_s": time.perf_counter() - started,
               "evaluation_note": "Identical original-video split and agreed window-centre timestamps to VideoMAE; all boundary-crossing windows retained. YOLO receives each full-resolution sampled frame."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected["per_video"][0].keys())
        writer.writeheader()
        writer.writerows(selected["per_video"])
    print(f"Validation chose {best['method']} width={best['width']}, min_run={best['min_run']}; "
          f"test raw F1={raw['macro_f1']:.4f}, smoothed F1={selected['macro_f1']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
