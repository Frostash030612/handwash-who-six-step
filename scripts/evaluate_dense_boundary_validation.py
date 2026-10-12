#!/usr/bin/env python
"""Validation-only dense-centre audit of the archived boundary-aware GRU.

The model already predicts at every 5 fps sampled frame. This script checks
whether short-run handling at that native 0.2 s resolution improves temporal
segments without erasing true short actions. No test data or model weights are
changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.core.segment_metrics import action_segments, segment_counts
from handwash.pipelines.boundary_gru import decode_boundaries, load_boundary_gru
from handwash.pipelines.exp_demo import file_sha256
from scripts.compare_full_video_models import _centres
from scripts.full_video_eval_common import RAW_SETTING, _predictions, public_metrics, score


def segment_f1(items: list[dict], setting: dict) -> tuple[float, dict[str, int]]:
    tp = fp = fn = 0
    for item in items:
        predicted = _predictions(item["probs"], setting)
        usable = item["usable"].astype(bool)
        result = segment_counts(action_segments(item["targets"], usable), action_segments(predicted, usable))
        tp += result[0]
        fp += result[1]
        fn += result[2]
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator if denominator else 0.0), {"tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-cache", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features"
    )
    parser.add_argument(
        "--head", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt"
    )
    parser.add_argument(
        "--comparison", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    summary = json.loads((args.comparison / "boundary_gru/summary.json").read_text(encoding="utf-8"))
    video_ids = summary["video_ids"]["val"]
    if len(video_ids) != 92:
        parser.error("Expected the fixed 92 validation videos")
    head, payload = load_boundary_gru(args.head)
    if payload.get("exp_sha256") != file_sha256(PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"):
        parser.error("Boundary head is paired with a different YOLO checkpoint")
    head = head.eval()
    decoder = payload["decoder"]
    dense: list[dict] = []
    sparse: list[dict] = []
    sparse_positions: list[np.ndarray] = []
    with torch.inference_mode():
        for number, video_id in enumerate(video_ids, start=1):
            filename = video_id.replace("/", "__") + ".npz"
            with np.load(args.feature_cache / filename, allow_pickle=False) as cached:
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64)
                consensus = cached["consensus"].astype(np.int64)
                fps = float(cached["source_fps"])
            if not (len(features) == len(frame_ids) == len(consensus)):
                raise ValueError(f"Misaligned cached arrays: {video_id}")
            logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
            probs = logits[0].softmax(-1).numpy()
            edges = edge_logits[0].sigmoid().numpy()
            if decoder["method"] == "boundary_segment":
                probs = decode_boundaries(
                    probs, edges, threshold=float(decoder["threshold"]), min_gap=int(decoder["min_gap"])
                )
            _starts, positions = _centres(frame_ids.tolist(), num_frames=16, window_stride=4)
            agreed = consensus[positions] >= 0
            with np.load(
                args.comparison / "boundary_gru/raw_probabilities" / filename, allow_pickle=False
            ) as archived:
                if not (
                    np.array_equal(frame_ids[positions], archived["centre_frame_ids"])
                    and np.array_equal(consensus[positions], archived["targets"])
                    and np.array_equal(agreed, archived["usable"])
                    # The archived GPU inference and this CPU audit can differ
                    # by a few 1e-5 in probabilities without changing labels.
                    and np.allclose(probs[positions], archived["probs"], atol=1e-3)
                ):
                    raise ValueError(f"Dense predictions do not reproduce archived centres: {video_id}")
            dense.append(
                {
                    "probs": probs,
                    "centre_frame_ids": frame_ids,
                    "targets": consensus,
                    "usable": consensus >= 0,
                    "fps": fps,
                }
            )
            sparse.append(
                {
                    "probs": probs[positions],
                    "centre_frame_ids": frame_ids[positions],
                    "targets": consensus[positions],
                    "usable": agreed,
                    "fps": fps,
                }
            )
            sparse_positions.append(positions)
            if number % 20 == 0:
                print(f"Validation videos {number}/{len(video_ids)}", flush=True)
    sparse_score = score(sparse, RAW_SETTING)
    if abs(sparse_score["macro_f1"] - 0.953701893445979) > 1e-5:
        raise ValueError("Sparse baseline no longer reproduces the archived validation score")
    settings = [
        RAW_SETTING,
        *({"method": "mean_min_run", "width": 1, "min_run": run} for run in (2, 3, 4)),
        *({"method": "mean_min_run", "width": 3, "min_run": run} for run in (1, 2, 3)),
        *(
            {"method": "hysteresis", "width": 1, "min_run": 1, "confirm_frames": confirm, "margin": margin}
            for confirm in (2, 3)
            for margin in (0.0, 0.05)
        ),
    ]
    grid = []
    for setting in settings:
        metrics = score(dense, setting)
        f1, counts = segment_f1(dense, setting)
        sampled = [
            {**item, "probs": np.eye(7, dtype=np.float32)[_predictions(full["probs"], setting)[positions]]}
            for item, full, positions in zip(sparse, dense, sparse_positions, strict=True)
        ]
        sampled_metrics = score(sampled, RAW_SETTING)
        sampled_f1, sampled_counts = segment_f1(sampled, RAW_SETTING)
        grid.append(
            {
                "setting": setting,
                "metrics": public_metrics(metrics),
                "who_segment_f1_at_50": f1,
                "segment_counts": counts,
                "sparse_after_dense": {
                    "metrics": public_metrics(sampled_metrics),
                    "who_segment_f1_at_50": sampled_f1,
                    "segment_counts": sampled_counts,
                },
            }
        )
    result = {
        "split": "val",
        "videos": len(video_ids),
        "sparse_baseline": public_metrics(sparse_score),
        "dense_grid": grid,
        "note": "Dense 5 fps labels and segments are a different evaluation unit from archived 0.8 s centres. Compare settings only within this dense grid. No test data read.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
