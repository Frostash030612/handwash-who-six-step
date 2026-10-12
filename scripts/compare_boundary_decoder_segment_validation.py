#!/usr/bin/env python
"""Audit the existing boundary decoder grid against WHO segment F1 on validation."""

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
from handwash.pipelines.boundary_gru import decode_boundaries, load_boundary_gru
from handwash.pipelines.exp_demo import file_sha256
from scripts.compare_full_video_models import _centres
from scripts.evaluate_dense_boundary_validation import segment_f1
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt")
    parser.add_argument("--feature-cache", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--comparison", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    summary = json.loads((args.comparison / "boundary_gru/summary.json").read_text(encoding="utf-8"))
    ids = summary["video_ids"]["val"]
    if len(ids) != 92:
        parser.error("Expected 92 fixed validation videos")
    head, payload = load_boundary_gru(args.head)
    if payload["exp_sha256"] != file_sha256(PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"):
        parser.error("YOLO checkpoint mismatch")
    head.eval()
    entries = []
    baseline = []
    with torch.inference_mode():
        for video_id in ids:
            filename = video_id.replace("/", "__") + ".npz"
            with np.load(args.feature_cache / filename, allow_pickle=False) as cached:
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64)
                consensus = cached["consensus"].astype(np.int64)
                fps = float(cached["source_fps"])
            _starts, positions = _centres(frame_ids.tolist(), num_frames=16, window_stride=4)
            logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
            template = {
                "video_id": video_id, "centre_frame_ids": frame_ids[positions],
                "targets": consensus[positions], "usable": consensus[positions] >= 0, "fps": fps,
            }
            with np.load(args.comparison / "boundary_gru/raw_probabilities" / filename,
                         allow_pickle=False) as archived:
                if not (np.array_equal(template["centre_frame_ids"], archived["centre_frame_ids"])
                        and np.array_equal(template["targets"], archived["targets"])):
                    raise ValueError(f"Misaligned archive: {video_id}")
                baseline.append({**template, "probs": archived["probs"].copy()})
            entries.append({"template": template, "positions": positions,
                            "class_probs": logits[0].softmax(-1).numpy(),
                            "edge_probs": edge_logits[0].sigmoid().numpy()})
    old = score(baseline, RAW_SETTING)
    if abs(old["macro_f1"] - 0.953701893445979) > 1e-6:
        raise ValueError("Archived baseline score not reproduced")
    old_f1, old_counts = segment_f1(baseline, RAW_SETTING)
    grid = []
    for threshold in (0.25, 0.4, 0.55, 0.7, 0.85):
        for min_gap in (2, 3, 4):
            items = [
                {**entry["template"], "probs": decode_boundaries(
                    entry["class_probs"], entry["edge_probs"],
                    threshold=threshold, min_gap=min_gap,
                )[entry["positions"]]}
                for entry in entries
            ]
            result = score(items, RAW_SETTING)
            f1, counts = segment_f1(items, RAW_SETTING)
            grid.append({"threshold": threshold, "min_gap": min_gap,
                         "who_segment_f1_at_50": f1, "segment_counts": counts,
                         "metrics": public_metrics(result)})
    result = {
        "split": "val", "videos": len(ids), "test_used": False,
        "archived_decoder": payload["decoder"],
        "baseline": {**public_metrics(old), "who_segment_f1_at_50": old_f1,
                     "segment_counts": old_counts},
        "grid": grid,
        "best_by_segment_f1": max(grid, key=lambda row: (
            row["who_segment_f1_at_50"], row["metrics"]["macro_f1"],
            row["metrics"]["sequence_exact_videos"],
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("baseline", "best_by_segment_f1")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
