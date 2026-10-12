#!/usr/bin/env python
"""Compare frozen boundary-GRU candidates on validation videos only.

This script never reads test videos. Each candidate uses the decoder already
saved in its checkpoint. The bootstrap resamples whole videos, not frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.boundary_gru import (
    boundary_targets_from_consensus,
    decode_boundaries,
    load_boundary_gru,
    weak_boundary_intervals_from_consensus,
)
from handwash.pipelines.exp_demo import file_sha256
from scripts.full_video_eval_common import videos_from_reference
from scripts.train_boundary_yolo_gru import _predict_items, _prepare_eval_templates, _score_predictions
from scripts.train_current_yolo_gru import _read_or_extract


def _labels_for_video(item: dict, decoder: dict) -> tuple[np.ndarray, np.ndarray]:
    probabilities = item["frame_probs"]
    if decoder["method"] == "boundary_segment":
        probabilities = decode_boundaries(
            probabilities, item["edge_probs"],
            threshold=float(decoder["threshold"]), min_gap=int(decoder["min_gap"]),
        )
    keep = item["usable"]
    target = np.asarray(item["targets"], dtype=np.int64)[keep]
    predicted = probabilities[item["positions"]].argmax(axis=1)[keep]
    return target, predicted


def _pooled_f1(video_pairs: list[tuple[np.ndarray, np.ndarray]], indices: np.ndarray) -> float:
    truth = np.concatenate([video_pairs[int(index)][0] for index in indices])
    prediction = np.concatenate([video_pairs[int(index)][1] for index in indices])
    return float(f1_score(truth, prediction, labels=list(range(7)), average="macro", zero_division=0))


def _boundary_diagnostics(predictions: list[dict], data: list[dict], *, threshold: float = 0.25) -> dict:
    """Separate exact adjacent boundaries from short uncertain change intervals."""
    totals = {
        "adjacent_true": 0, "adjacent_hit": 0, "known_negative": 0,
        "known_negative_alarm": 0, "short_unknown_change_intervals": 0,
        "short_unknown_interval_hit": 0,
    }
    for prediction, cached in zip(predictions, data, strict=True):
        if prediction["video_id"] != cached["video_id"]:
            raise ValueError("Validation prediction/cache video order mismatch")
        target, mask = boundary_targets_from_consensus(cached["consensus"])
        positive = mask & (target > 0)
        negative = mask & (target == 0)
        detected = prediction["edge_probs"] >= threshold
        totals["adjacent_true"] += int(positive.sum())
        totals["adjacent_hit"] += int((positive & detected).sum())
        totals["known_negative"] += int(negative.sum())
        totals["known_negative_alarm"] += int((negative & detected).sum())
        intervals = weak_boundary_intervals_from_consensus(cached["consensus"], max_gap=3)
        totals["short_unknown_change_intervals"] += len(intervals)
        totals["short_unknown_interval_hit"] += sum(
            bool(detected[left:right].any()) for left, right in intervals
        )
    return {
        "threshold": threshold,
        **totals,
        "adjacent_recall": totals["adjacent_hit"] / max(totals["adjacent_true"], 1),
        "known_negative_alarm_rate": totals["known_negative_alarm"] / max(totals["known_negative"], 1),
        "short_unknown_interval_hit_rate": (
            totals["short_unknown_interval_hit"] / max(totals["short_unknown_change_intervals"], 1)
        ),
    }


def _group_diagnostics(predictions: list[dict], decoder: dict) -> dict[str, dict]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for item in predictions:
        source = item["video_id"].split("/", 1)[0]
        grouped.setdefault(source, []).append(_labels_for_video(item, decoder))
    report = {}
    for source, pairs in sorted(grouped.items()):
        truth = np.concatenate([pair[0] for pair in pairs])
        predicted = np.concatenate([pair[1] for pair in pairs])
        confusion = confusion_matrix(truth, predicted, labels=list(range(7)))
        report[source] = {
            "videos": len(pairs), "centres": len(truth),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=list(range(7)),
                                       average="macro", zero_division=0)),
            "class_support": confusion.sum(axis=1).tolist(),
            "class_recall": (np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)).tolist(),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--split-reference", type=Path,
                        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv")
    parser.add_argument("--yolo", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt")
    parser.add_argument("--cache-root", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--ensemble", action="store_true",
                        help="Also test a fixed 50/50 probability average with the baseline")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 100:
        parser.error("Use at least 100 video-level bootstrap repetitions")
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")

    videos = [video for video in videos_from_reference(args.root, args.split_reference)
              if video.split == "val"]
    if not videos:
        parser.error("No validation videos in locked split")
    model_sha = file_sha256(args.yolo)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = []
    extractor = None
    for video in videos:
        item, extractor = _read_or_extract(video, extractor, args.yolo, model_sha,
                                           args.cache_root, 5.0, 7, device)
        data.append(item)
    templates = _prepare_eval_templates(data, {f"{v.source}/{v.stem}": v for v in videos})

    results = {}
    video_pairs = {}
    prediction_sets = {}
    decoders = {}
    for path in [args.baseline, *args.candidate]:
        head, payload = load_boundary_gru(path)
        if payload.get("exp_sha256") != model_sha:
            raise ValueError(f"YOLO feature model mismatch: {path}")
        decoder = payload["decoder"]
        predictions = _predict_items(head.to(device), templates, device)
        prediction_sets[str(path)] = predictions
        decoders[str(path)] = decoder
        metrics = _score_predictions(predictions, decoder)
        results[str(path)] = {key: metrics[key] for key in (
            "videos", "evaluated_centres", "accuracy", "macro_f1",
            "transition_macro_f1", "sequence_exact_videos",
        )}
        results[str(path)]["boundary_diagnostics_at_common_threshold"] = _boundary_diagnostics(
            predictions, data
        )
        results[str(path)]["by_dataset"] = _group_diagnostics(predictions, decoder)
        video_pairs[str(path)] = [_labels_for_video(item, decoder) for item in predictions]

    ensemble_keys = []
    if args.ensemble:
        for path in args.candidate:
            name = f"ensemble_50_50:{path}"
            mixed = []
            for first, second in zip(prediction_sets[str(args.baseline)],
                                     prediction_sets[str(path)], strict=True):
                if first["video_id"] != second["video_id"]:
                    raise ValueError("Ensemble candidates have different validation videos")
                mixed.append({
                    **first,
                    "frame_probs": 0.5 * (first["frame_probs"] + second["frame_probs"]),
                    "edge_probs": 0.5 * (first["edge_probs"] + second["edge_probs"]),
                })
            decoder = decoders[str(args.baseline)]
            metrics = _score_predictions(mixed, decoder)
            results[name] = {key: metrics[key] for key in (
                "videos", "evaluated_centres", "accuracy", "macro_f1",
                "transition_macro_f1", "sequence_exact_videos",
            )}
            results[name]["by_dataset"] = _group_diagnostics(mixed, decoder)
            video_pairs[name] = [_labels_for_video(item, decoder) for item in mixed]
            ensemble_keys.append(name)

    baseline = video_pairs[str(args.baseline)]
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, len(baseline), size=(args.repetitions, len(baseline)))
    comparisons = {}
    for name in [*[str(path) for path in args.candidate], *ensemble_keys]:
        candidate = video_pairs[name]
        differences = np.asarray([
            _pooled_f1(candidate, sample) - _pooled_f1(baseline, sample)
            for sample in samples
        ])
        comparisons[name] = {
            "macro_f1_delta": results[name]["macro_f1"] - results[str(args.baseline)]["macro_f1"],
            "bootstrap_95pct_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
            "bootstrap_unit": "whole validation video",
        }

    report = {
        "split": "val", "split_reference_sha256": file_sha256(args.split_reference),
        "test_used": False, "bootstrap_repetitions": args.repetitions,
        "results": results, "paired_comparisons_to_baseline": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
