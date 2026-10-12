#!/usr/bin/env python
"""Check whether VideoMAE adds complementary evidence on validation videos.

Reads the archived complete-video predictions for the boundary-GRU and
center-stable VideoMAE. Only validation video IDs are opened. A fixed small
weight grid is reported; it does not change either checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_videomae_fusion_validation.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    baseline_root = args.comparison / "boundary_gru"
    videomae_root = args.comparison / "videomae_center"
    baseline_summary = json.loads((baseline_root / "summary.json").read_text(encoding="utf-8"))
    videomae_summary = json.loads((videomae_root / "summary.json").read_text(encoding="utf-8"))
    ids = baseline_summary["video_ids"]["val"]
    if len(ids) != 92 or ids != videomae_summary["video_ids"]["val"]:
        parser.error("Validation video IDs differ between methods")
    originals = []
    videomae_probs = []
    for video_id in ids:
        name = video_id.replace("/", "__") + ".npz"
        with (
            np.load(baseline_root / "raw_probabilities" / name, allow_pickle=False) as first,
            np.load(videomae_root / "raw_probabilities" / name, allow_pickle=False) as second,
        ):
            for field in ("centre_frame_ids", "targets", "usable"):
                if not np.array_equal(first[field], second[field]):
                    raise ValueError(f"Methods disagree on {field} for {video_id}")
            originals.append(
                {
                    "video_id": video_id,
                    "probs": first["probs"].astype(np.float32),
                    "centre_frame_ids": first["centre_frame_ids"],
                    "targets": first["targets"],
                    "usable": first["usable"],
                    "fps": float(first["fps"]),
                }
            )
            videomae_probs.append(second["probs"].astype(np.float32))
    grid = []
    predictions = {}
    for weight in (0.0, 0.1, 0.2, 0.3, 0.5):
        items = [
            {**item, "probs": (1 - weight) * item["probs"] + weight * extra}
            for item, extra in zip(originals, videomae_probs, strict=True)
        ]
        result = score(items, RAW_SETTING)
        grid.append(
            {
                "videomae_weight": weight,
                **{
                    key: result[key]
                    for key in (
                        "macro_f1",
                        "transition_macro_f1",
                        "accuracy",
                        "sequence_exact_videos",
                    )
                },
            }
        )
        predictions[weight] = [item["probs"].argmax(1) for item in items]
    best = max(
        grid,
        key=lambda row: (
            row["macro_f1"],
            row["transition_macro_f1"],
            row["sequence_exact_videos"],
        ),
    )
    baseline = score(originals, RAW_SETTING)
    pair_arrays = []
    for item, new_pred, old_pred in zip(
        originals, predictions[best["videomae_weight"]], predictions[0.0], strict=True
    ):
        keep = item["usable"]
        pair_arrays.append((item["targets"][keep], new_pred[keep], old_pred[keep]))
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        indices = rng.integers(0, len(pair_arrays), size=len(pair_arrays))
        pooled = [np.concatenate([pair_arrays[int(index)][part] for index in indices]) for part in range(3)]
        deltas.append(
            float(
                f1_score(pooled[0], pooled[1], labels=list(range(7)), average="macro", zero_division=0)
                - f1_score(pooled[0], pooled[2], labels=list(range(7)), average="macro", zero_division=0)
            )
        )
    result = {
        "split": "val",
        "videos": len(ids),
        "baseline": public_metrics(baseline),
        "grid": grid,
        "best_grid": best,
        "best_delta_macro_f1": best["macro_f1"] - baseline["macro_f1"],
        "paired_video_bootstrap_95pct_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
