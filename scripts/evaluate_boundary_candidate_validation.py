#!/usr/bin/env python
"""Compare one boundary-GRU candidate with the archived baseline on validation only."""

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
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--comparison", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    summary = json.loads((args.comparison / "boundary_gru/summary.json").read_text(encoding="utf-8"))
    video_ids = summary["video_ids"]["val"]
    if len(video_ids) != 92:
        parser.error("Expected the fixed 92 validation videos")
    head, payload = load_boundary_gru(args.head)
    if payload["exp_sha256"] != file_sha256(PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"):
        parser.error("Candidate and YOLO checkpoint differ")
    head.eval()
    decoder = payload["decoder"]
    proposed: list[dict] = []
    baseline: list[dict] = []
    with torch.inference_mode():
        for video_id in video_ids:
            filename = video_id.replace("/", "__") + ".npz"
            with np.load(args.feature_cache / filename, allow_pickle=False) as cached:
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64)
                consensus = cached["consensus"].astype(np.int64)
                fps = float(cached["source_fps"])
            _starts, positions = _centres(frame_ids.tolist(), num_frames=16, window_stride=4)
            class_logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
            frame_probs = class_logits[0].softmax(-1).numpy()
            if decoder["method"] == "boundary_segment":
                frame_probs = decode_boundaries(
                    frame_probs, edge_logits[0].sigmoid().numpy(),
                    threshold=float(decoder["threshold"]), min_gap=int(decoder["min_gap"]),
                )
            with np.load(args.comparison / "boundary_gru/raw_probabilities" / filename,
                         allow_pickle=False) as archived:
                if not (np.array_equal(frame_ids[positions], archived["centre_frame_ids"])
                        and np.array_equal(consensus[positions], archived["targets"])):
                    raise ValueError(f"Archive and candidate axes differ: {video_id}")
                common = {
                    "video_id": video_id, "centre_frame_ids": frame_ids[positions],
                    "targets": consensus[positions], "usable": consensus[positions] >= 0,
                    "fps": fps,
                }
                baseline.append({**common, "probs": archived["probs"].copy()})
            proposed.append({**common, "probs": frame_probs[positions]})
    original = score(baseline, RAW_SETTING, per_video=True)
    candidate = score(proposed, RAW_SETTING, per_video=True)
    if abs(original["macro_f1"] - 0.953701893445979) > 1e-6:
        raise ValueError("Archived baseline score not reproduced")
    original_segment, original_counts = segment_f1(baseline, RAW_SETTING)
    candidate_segment, candidate_counts = segment_f1(proposed, RAW_SETTING)
    paired = [
        {"video_id": before["video_id"], "baseline_exact": before["sequence_exact"],
         "candidate_exact": after["sequence_exact"],
         "baseline_edit_distance": before["sequence_edit_distance"],
         "candidate_edit_distance": after["sequence_edit_distance"]}
        for before, after in zip(original["per_video"], candidate["per_video"], strict=True)
    ]
    result = {
        "split": "val", "videos": len(video_ids), "test_used": False,
        "candidate_head": str(args.head), "candidate_decoder": decoder,
        "baseline": {**public_metrics(original), "who_segment_f1_at_50": original_segment,
                     "segment_counts": original_counts},
        "candidate": {**public_metrics(candidate), "who_segment_f1_at_50": candidate_segment,
                      "segment_counts": candidate_counts},
        "strict_sequence_gained_videos": sum(not row["baseline_exact"] and row["candidate_exact"] for row in paired),
        "strict_sequence_lost_videos": sum(row["baseline_exact"] and not row["candidate_exact"] for row in paired),
        "per_video": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline", "candidate", "strict_sequence_gained_videos", "strict_sequence_lost_videos"
    )}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
