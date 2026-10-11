"""Shared, split-safe protocol for whole-video action recognition comparisons.

All models predict the same 5 fps window centres. Post-processing is selected
on validation videos and then frozen for the held-out test videos.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.evaluate_yolo_full_videos import smooth, smooth_hysteresis
from scripts.train_videomae_pskus import VideoInfo, _discover_videos


def videos_from_reference(root: Path, reference: Path) -> list[VideoInfo]:
    """Use exactly the original-video membership used to train both models."""
    videos = _discover_videos(root, full_pskus=True)
    discovered = {f"{video.source}/{video.stem}": video for video in videos}
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    recorded: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["video_id"]
        if key in recorded or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid or duplicate split entry: {key}")
        recorded[key] = row
    # Additional PSKUS partitions may be downloaded later. They must not
    # silently enter this locked DataSet1/2/3/4 comparison.
    missing = sorted(set(recorded) - set(discovered))
    if missing:
        raise ValueError(f"Reference videos missing from disk: {missing[:3]}")
    result = []
    for key, video in discovered.items():
        if key not in recorded:
            continue
        row = recorded[key]
        if (row["primary"], row["secondary"]) != (video.primary.parent.name, video.secondary.parent.name):
            raise ValueError(f"Annotator pair changed: {key}")
        result.append(replace(video, split=row["split"]))
    return result


def _sequence(labels: np.ndarray) -> list[int]:
    steps: list[int] = []
    for label in labels:
        value = int(label)
        if value and (not steps or steps[-1] != value):
            steps.append(value)
    return steps


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, start=1):
        current = [index]
        for column, other in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (value != other)))
        previous = current
    return previous[-1]


def _predictions(probs: np.ndarray, setting: dict[str, object]) -> np.ndarray:
    if setting["method"] == "hysteresis":
        return smooth_hysteresis(probs, int(setting["width"]),
                                 int(setting["confirm_frames"]), float(setting["margin"]),
                                 int(setting["min_run"]))
    return smooth(probs, int(setting["width"]), int(setting["min_run"]))


def score(items: list[dict], setting: dict[str, object], *, per_video: bool = False) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    truth: list[int] = []
    predictions: list[int] = []
    boundary_truth: list[int] = []
    boundary_predictions: list[int] = []
    rows: list[dict] = []
    exact = edits = reference_steps = 0
    for item in items:
        labels = _predictions(item["probs"], setting)
        keep = item["usable"]
        target = item["targets"][keep]
        predicted = labels[keep]
        truth.extend(target.tolist())
        predictions.extend(predicted.tolist())
        near = np.zeros(len(keep), dtype=bool)
        for index in range(1, len(keep)):
            gap = (item["centre_frame_ids"][index] - item["centre_frame_ids"][index - 1]) / item["fps"]
            if keep[index - 1] and keep[index] and item["targets"][index - 1] != item["targets"][index] and gap <= 1.6:
                near[max(0, index - 2):min(len(keep), index + 2)] = True
        boundary = near & keep
        boundary_truth.extend(item["targets"][boundary].tolist())
        boundary_predictions.extend(labels[boundary].tolist())
        reference_sequence = _sequence(target)
        predicted_sequence = _sequence(predicted)
        exact += predicted_sequence == reference_sequence
        edits += _edit_distance(predicted_sequence, reference_sequence)
        reference_steps += max(1, len(reference_sequence))
        if per_video:
            rows.append({
                "video_id": item["video_id"], "evaluated_centres": len(target),
                "accuracy": float(accuracy_score(target, predicted)) if len(target) else None,
                "macro_f1": float(f1_score(target, predicted, labels=list(range(7)),
                                           average="macro", zero_division=0)) if len(target) else None,
                "sequence_exact": predicted_sequence == reference_sequence,
                "sequence_edit_distance": _edit_distance(predicted_sequence, reference_sequence),
                "predicted_sequence": ">".join(map(str, predicted_sequence)),
                "reference_sequence": ">".join(map(str, reference_sequence)),
            })
    if not truth:
        raise ValueError("No two-annotator consensus centres to evaluate")
    return {
        "videos": len(items), "evaluated_centres": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "transition_centres": len(boundary_truth),
        "transition_accuracy": float(accuracy_score(boundary_truth, boundary_predictions)) if boundary_truth else None,
        "transition_macro_f1": float(f1_score(boundary_truth, boundary_predictions,
                                               labels=list(range(7)), average="macro", zero_division=0)) if boundary_truth else None,
        "sequence_exact_videos": exact, "sequence_exact_rate": exact / len(items),
        "sequence_edit_distance_per_reference_step": edits / reference_steps,
        "per_video": rows, "truth": truth, "predictions": predictions,
    }


def choose_smoothing(validation: list[dict]) -> tuple[dict[str, object], list[dict]]:
    grid: list[dict] = []
    for width in (1, 3, 5):
        for min_run in (1, 2, 3):
            setting = {"method": "mean_min_run", "width": width, "min_run": min_run}
            result = score(validation, setting)
            grid.append({**setting, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
        for confirm_frames in (2, 3):
            for margin in (0.0, 0.05):
                setting = {"method": "hysteresis", "width": width, "min_run": 1,
                           "confirm_frames": confirm_frames, "margin": margin}
                result = score(validation, setting)
                grid.append({**setting, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    best = max(grid, key=lambda row: (row["macro_f1"], row["accuracy"],
                                      -row["width"], -row["min_run"]))
    return best, grid


RAW_SETTING: dict[str, object] = {"method": "mean_min_run", "width": 1, "min_run": 1}


def public_metrics(result: dict) -> dict:
    return {key: result[key] for key in (
        "videos", "evaluated_centres", "accuracy", "macro_f1", "transition_centres",
        "transition_accuracy", "transition_macro_f1", "sequence_exact_videos",
        "sequence_exact_rate", "sequence_edit_distance_per_reference_step",
    )}
