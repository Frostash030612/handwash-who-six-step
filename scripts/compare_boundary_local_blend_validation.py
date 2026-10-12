#!/usr/bin/env python
"""Check whether local GRU evidence improves boundary-near predictions on validation.

The existing boundary-GRU checkpoint and decoder remain fixed. Near a detected
change, blend a small part of the unsmoothed class probabilities back into the
segment prediction. The locked test split is never opened.
"""

from __future__ import annotations

import argparse
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
from scripts.compare_boundary_viterbi_validation import _fold
from scripts.compare_full_video_models import _centres, _item
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score, videos_from_reference


def _boundary_positions(edges: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    left = np.r_[-np.inf, edges[:-1]]
    right = np.r_[edges[1:], -np.inf]
    peaks = np.flatnonzero((edges >= threshold) & (edges >= left) & (edges > right))
    chosen: list[int] = []
    for position in sorted(peaks.tolist(), key=lambda index: float(edges[index]), reverse=True):
        if position and all(abs(position - other) >= min_gap for other in chosen):
            chosen.append(position)
    return sorted(chosen)


def blend_near_boundaries(
    class_probs: np.ndarray,
    edge_probs: np.ndarray,
    *,
    threshold: float,
    min_gap: int,
    radius: int,
    local_weight: float,
) -> np.ndarray:
    """Keep segment decisions; restore local evidence only near detected changes."""
    if radius < 0 or not 0 <= local_weight <= 1:
        raise ValueError("radius must be nonnegative and local_weight in [0,1]")
    segment = decode_boundaries(class_probs, edge_probs, threshold=threshold, min_gap=min_gap)
    if radius == 0 or local_weight == 0:
        return segment
    result = segment.copy()
    weights = np.zeros(len(result), dtype=np.float32)
    for boundary in _boundary_positions(np.asarray(edge_probs), threshold, min_gap):
        for index in range(max(0, boundary - radius), min(len(result), boundary + radius)):
            distance = min(abs(index - boundary), abs(index - boundary + 1))
            weights[index] = max(weights[index], local_weight * (1 - distance / radius))
    return (1 - weights[:, None]) * result + weights[:, None] * class_probs


def _score(entries: list[dict], setting: dict) -> dict:
    items = []
    for entry in entries:
        probs = blend_near_boundaries(entry["class_probs"], entry["edge_probs"], **setting)[
            entry["positions"]
        ]
        items.append({**entry["template"], "probs": probs})
    return score(items, RAW_SETTING)


def main() -> int:
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_local_blend_validation.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    for required in (args.split_reference, args.yolo, args.head, args.feature_cache):
        if not required.exists():
            parser.error(f"Missing input: {required}")
    videos = [
        video for video in videos_from_reference(args.root, args.split_reference) if video.split == "val"
    ]
    if len(videos) != 92:
        parser.error(f"Expected 92 validation videos, found {len(videos)}")
    head, payload = load_boundary_gru(args.head)
    if payload.get("exp_sha256") != file_sha256(args.yolo):
        parser.error("Boundary head and YOLO checkpoint mismatch")
    head.eval()
    decoder = payload["decoder"]
    base_setting = {
        "threshold": float(decoder["threshold"]),
        "min_gap": int(decoder["min_gap"]),
        "radius": 0,
        "local_weight": 0.0,
    }
    entries = []
    with torch.inference_mode():
        for video in videos:
            path = args.feature_cache / f"{video.source}__{video.stem}.npz"
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["model_sha256"]) != payload["exp_sha256"]:
                    raise ValueError(f"Feature cache and YOLO checkpoint mismatch: {path}")
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64).tolist()
                fps = float(cached["source_fps"])
            _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
            logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
            class_probs = logits[0].softmax(-1).numpy()
            edge_probs = edge_logits[0].sigmoid().numpy()
            template = _item(video, class_probs[positions], frame_ids, positions, fps)
            entries.append(
                {
                    "template": template,
                    "positions": positions,
                    "class_probs": class_probs,
                    "edge_probs": edge_probs,
                }
            )

    baseline = _score(entries, base_setting)
    if abs(baseline["macro_f1"] - 0.953701893445979) > 1e-5:
        raise RuntimeError("Archived boundary-GRU validation result was not reproduced")
    grid_settings = [
        {**base_setting, "radius": radius, "local_weight": weight}
        for radius in (1, 2, 3)
        for weight in (0.25, 0.5)
    ]
    rows = []
    for setting in grid_settings:
        result = _score(entries, setting)
        rows.append(
            {
                "radius": setting["radius"],
                "local_weight": setting["local_weight"],
                "macro_f1": result["macro_f1"],
                "transition_macro_f1": result["transition_macro_f1"],
                "sequence_exact_videos": result["sequence_exact_videos"],
            }
        )
    best = max(rows, key=lambda row: (row["macro_f1"], row["transition_macro_f1"]))
    best_setting = {**base_setting, "radius": best["radius"], "local_weight": best["local_weight"]}
    held_out_predictions = []
    chosen_by_fold = []
    for fold in range(5):
        train = [entry for entry in entries if _fold(entry["template"]["video_id"]) != fold]
        held_out = [entry for entry in entries if _fold(entry["template"]["video_id"]) == fold]
        if not train or not held_out:
            raise RuntimeError(f"Empty video fold: {fold}")
        chosen = max(grid_settings, key=lambda setting: _score(train, setting)["macro_f1"])
        chosen_by_fold.append({"fold": fold, "videos": len(held_out), **chosen})
        for entry in held_out:
            probs = blend_near_boundaries(entry["class_probs"], entry["edge_probs"], **chosen)[
                entry["positions"]
            ]
            held_out_predictions.append({**entry["template"], "probs": probs})
    cv_result = score(held_out_predictions, RAW_SETTING)
    arrays = []
    for entry in entries:
        keep = entry["template"]["usable"]
        truth = entry["template"]["targets"][keep]
        first = blend_near_boundaries(entry["class_probs"], entry["edge_probs"], **base_setting)[
            entry["positions"]
        ][keep].argmax(1)
        second = blend_near_boundaries(entry["class_probs"], entry["edge_probs"], **best_setting)[
            entry["positions"]
        ][keep].argmax(1)
        arrays.append((truth, first, second))
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(1000):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        joined = [np.concatenate([arrays[int(index)][part] for index in chosen]) for part in range(3)]
        differences.append(
            float(
                f1_score(joined[0], joined[2], labels=list(range(7)), average="macro", zero_division=0)
                - f1_score(joined[0], joined[1], labels=list(range(7)), average="macro", zero_division=0)
            )
        )
    output = {
        "split": "val",
        "videos": len(videos),
        "baseline": public_metrics(baseline),
        "grid": rows,
        "best_grid": best,
        "video_cv_selected": chosen_by_fold,
        "video_cv_score": public_metrics(cv_result),
        "paired_video_bootstrap_95pct_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: output[key] for key in ("baseline", "best_grid", "video_cv_score")}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
