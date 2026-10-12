#!/usr/bin/env python
"""Validation-only Viterbi decoding of the existing boundary-aware YOLO+GRU.

The transition score combines the model's action probabilities with its
boundary probability. Every label order remains possible, including an
incorrect or repeated washing step. No training data or locked test labels
are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.boundary_gru import decode_boundaries, load_boundary_gru
from handwash.pipelines.exp_demo import file_sha256


def decode_viterbi(
    class_probs: np.ndarray, edge_probs: np.ndarray, *, edge_weight: float, change_penalty: float
) -> np.ndarray:
    """Return labels maximizing per-frame class and learned change evidence."""
    probs = np.asarray(class_probs, dtype=np.float64)
    edges = np.asarray(edge_probs, dtype=np.float64).reshape(-1)
    if probs.ndim != 2 or len(probs) != len(edges) or probs.shape[1] != 7:
        raise ValueError("Expected [T,7] class probabilities and [T] edge probabilities")
    if edge_weight < 0 or change_penalty < 0:
        raise ValueError("Decoder weights must be nonnegative")
    if not len(probs):
        return np.empty(0, dtype=np.int64)
    emissions = np.log(np.clip(probs, 1e-8, 1.0))
    edge_logit = np.log(np.clip(edges, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(edges, 1e-6, 1 - 1e-6))
    previous = emissions[0]
    back = np.zeros((len(probs), 7), dtype=np.int8)
    changes = 1.0 - np.eye(7, dtype=np.float64)
    for index in range(1, len(probs)):
        bonus = edge_weight * edge_logit[index] - change_penalty
        candidates = previous[:, None] + changes * bonus
        back[index] = candidates.argmax(axis=0).astype(np.int8)
        previous = candidates.max(axis=0) + emissions[index]
    labels = np.empty(len(probs), dtype=np.int64)
    labels[-1] = int(previous.argmax())
    for index in range(len(probs) - 1, 0, -1):
        labels[index - 1] = back[index, labels[index]]
    return labels


def _scored_items(entries: list[dict], labels_by_video: dict[str, np.ndarray]) -> list[dict]:
    scored = []
    for entry in entries:
        labels = labels_by_video[entry["template"]["video_id"]]
        num_classes = entry["class_probs"].shape[1]
        probs = np.eye(num_classes, dtype=np.float32)[labels[entry["positions"]]]
        scored.append({**entry["template"], "probs": probs})
    return scored


def _fold(video_id: str, count: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(video_id.encode()).digest()[:4], "big") % count


def main() -> int:
    from scripts.compare_full_video_models import _centres, _item
    from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score, videos_from_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument(
        "--split-reference",
        type=Path,
        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv",
    )
    parser.add_argument(
        "--yolo", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"
    )
    parser.add_argument(
        "--head", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt"
    )
    parser.add_argument(
        "--feature-cache", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_viterbi_validation.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    for required in (args.split_reference, args.yolo, args.head, args.feature_cache):
        if not required.exists():
            parser.error(f"Missing required input: {required}")
    videos = [
        video for video in videos_from_reference(args.root, args.split_reference) if video.split == "val"
    ]
    if len(videos) != 92:
        parser.error(f"Expected 92 fixed validation videos, found {len(videos)}")
    head, payload = load_boundary_gru(args.head)
    if payload.get("exp_sha256") != file_sha256(args.yolo):
        parser.error("Boundary head is paired with a different YOLO checkpoint")
    head.eval()
    entries: list[dict] = []
    with torch.inference_mode():
        for number, video in enumerate(videos, start=1):
            path = args.feature_cache / f"{video.source}__{video.stem}.npz"
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["model_sha256"]) != payload["exp_sha256"]:
                    raise ValueError(f"Mismatched feature cache: {path}")
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64).tolist()
                fps = float(cached["source_fps"])
            _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
            logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
            probs = logits[0].softmax(dim=-1).numpy()
            edges = edge_logits[0].sigmoid().numpy()
            template = _item(video, probs[positions], frame_ids, positions, fps)
            entries.append(
                {"template": template, "positions": positions, "class_probs": probs, "edge_probs": edges}
            )
            if number % 20 == 0:
                print(f"Features {number}/{len(videos)}", flush=True)

    decoder = payload["decoder"]
    baseline_labels = {
        entry["template"]["video_id"]: decode_boundaries(
            entry["class_probs"],
            entry["edge_probs"],
            threshold=float(decoder["threshold"]),
            min_gap=int(decoder["min_gap"]),
        ).argmax(axis=1)
        for entry in entries
    }
    baseline_items = _scored_items(entries, baseline_labels)
    baseline_score = score(baseline_items, RAW_SETTING)
    if abs(baseline_score["macro_f1"] - 0.953701893445979) > 1e-5:
        raise RuntimeError("Validation protocol no longer reproduces the archived boundary-GRU score")

    settings = [
        {"edge_weight": edge_weight, "change_penalty": penalty}
        for edge_weight in (0.0, 0.25, 0.5, 1.0, 2.0)
        for penalty in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    decoded = {}
    grid = []
    for setting in settings:
        key = (setting["edge_weight"], setting["change_penalty"])
        labels = {
            entry["template"]["video_id"]: decode_viterbi(
                entry["class_probs"],
                entry["edge_probs"],
                **setting,
            )
            for entry in entries
        }
        decoded[key] = labels
        result = score(_scored_items(entries, labels), RAW_SETTING)
        grid.append(
            {
                **setting,
                **{
                    metric: result[metric]
                    for metric in (
                        "macro_f1",
                        "transition_macro_f1",
                        "sequence_exact_videos",
                        "accuracy",
                    )
                },
            }
        )
    best = max(
        grid,
        key=lambda row: (
            row["macro_f1"],
            row["transition_macro_f1"],
            row["sequence_exact_videos"],
        ),
    )
    print(f"Best in-sample grid: {best}", flush=True)

    # Select settings on four fifths of validation videos and score the held-out fifth.
    # This estimates whether a tiny grid gain survives video-level resampling.
    out_of_fold: list[dict] = []
    selected_folds = []
    for fold in range(5):
        training = [entry for entry in entries if _fold(entry["template"]["video_id"]) != fold]
        held_out = [entry for entry in entries if _fold(entry["template"]["video_id"]) == fold]
        if not training or not held_out:
            raise RuntimeError(f"Empty video fold {fold}")
        chosen = max(
            settings,
            key=lambda setting: score(
                _scored_items(training, decoded[(setting["edge_weight"], setting["change_penalty"])]),
                RAW_SETTING,
            )["macro_f1"],
        )
        selected_folds.append({"fold": fold, "videos": len(held_out), **chosen})
        out_of_fold.extend(
            _scored_items(
                held_out,
                decoded[(chosen["edge_weight"], chosen["change_penalty"])],
            )
        )
    cv_score = score(out_of_fold, RAW_SETTING)

    # Bootstrap by whole video rather than by overlapping window centres.
    best_items = _scored_items(entries, decoded[(best["edge_weight"], best["change_penalty"])])
    arrays = []
    for old, new in zip(baseline_items, best_items, strict=True):
        keep = new["usable"]
        arrays.append(
            (new["targets"][keep], new["probs"][keep].argmax(axis=1), old["probs"][keep].argmax(axis=1))
        )
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        joined = [np.concatenate([arrays[int(index)][part] for index in selected]) for part in range(3)]
        differences.append(
            float(
                f1_score(joined[0], joined[1], labels=list(range(7)), average="macro", zero_division=0)
                - f1_score(joined[0], joined[2], labels=list(range(7)), average="macro", zero_division=0)
            )
        )
    result = {
        "split": "val",
        "videos": len(entries),
        "head_sha256": file_sha256(args.head),
        "split_reference_sha256": file_sha256(args.split_reference),
        "baseline": public_metrics(baseline_score),
        "best_grid": best,
        "grid": grid,
        "video_cv_selected": selected_folds,
        "video_cv_score": public_metrics(cv_score),
        "best_grid_delta_macro_f1": best["macro_f1"] - baseline_score["macro_f1"],
        "paired_video_bootstrap_95pct_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "baseline",
                    "best_grid",
                    "video_cv_score",
                    "best_grid_delta_macro_f1",
                    "paired_video_bootstrap_95pct_interval",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
