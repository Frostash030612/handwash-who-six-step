#!/usr/bin/env python3
"""Compare four self-scene models at the same held-out video time centers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import PROJECT_ROOT
else:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401

from handwash.core.action_chain import adaptive_boundary_gru_chain
from handwash.core.boundary_metrics import (
    BoundaryScore,
    predicted_boundaries,
    reference_boundaries,
    score_boundaries,
)

MODELS = ("yolo", "yolo_gru", "boundary_gru", "videomae")
NAMES = {
    "yolo": "YOLO",
    "yolo_gru": "YOLO + GRU",
    "boundary_gru": "YOLO + Adaptive Boundary-Aware GRU",
    "videomae": "VideoMAE",
}
EXPECTED_VIDEOS = {
    "4",
    "8",
    "test_correct_01",
    "test_correct_02",
    "test_correct_03",
    "test_correct_04",
    "test_error_12356",
    "test_error_12564",
    "test_error_32654",
    "test_error_645321",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _order(values: list[int]) -> list[int]:
    order: list[int] = []
    for value in values:
        if value > 0 and (not order or order[-1] != value):
            order.append(value)
    return order


def _classification(y: list[int], prediction: list[int]) -> tuple[float, float]:
    truth = np.asarray(y)
    predicted = np.asarray(prediction)
    accuracy = float(np.mean(truth == predicted))
    f1s = []
    for label in range(7):
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return accuracy, float(np.mean(f1s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-predictions", type=Path, required=True)
    parser.add_argument("--videomae-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame_rows = _read_csv(args.frame_predictions)
    videomae_rows = _read_csv(args.videomae_predictions)
    if len(videomae_rows) != 632:
        raise ValueError("Expected the 632 valid VideoMAE centers after the video 8 correction")
    dense: dict[str, list[dict[str, str]]] = defaultdict(list)
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in frame_rows:
        dense[row["video_id"]].append(row)
        key = (row["video_id"], row["actual_time_s"])
        if key in lookup:
            raise ValueError(f"Duplicate frame time: {key}")
        lookup[key] = row

    aligned: dict[str, list[dict]] = defaultdict(list)
    for clip in videomae_rows:
        clip_id = clip["clip_id"]
        frame_id = {"4": "prior_video4", "8": "prior_video8"}.get(clip_id, clip_id)
        key = (frame_id, f"{float(clip['time_s']):.6f}")
        if key not in lookup:
            raise ValueError(f"Missing exact frame center: {key}")
        frame = lookup[key]
        truth = int(clip["true_label"])
        if frame["valid_for_evaluation"] != "True" or int(frame["ground_truth"]) != truth:
            raise ValueError(f"Ground truth differs at {key}")
        aligned[clip_id].append(
            {
                "time_s": float(clip["time_s"]),
                "truth": truth,
                "yolo": int(frame["yolo"]),
                "yolo_gru": int(frame["yolo_gru"]),
                "boundary_gru": int(frame["boundary_gru"]),
                "videomae": int(clip["predicted_label"]),
            }
        )
    if set(aligned) != EXPECTED_VIDEOS:
        raise ValueError(f"Unexpected test videos: {set(aligned) ^ EXPECTED_VIDEOS}")

    results: dict[str, dict] = {}
    per_video: dict[str, dict] = {}
    for clip_id, rows in aligned.items():
        rows.sort(key=lambda row: row["time_s"])
        truth = [row["truth"] for row in rows]
        times = [row["time_s"] for row in rows]
        ref = reference_boundaries(truth, times)
        true_order = _order(truth)
        video_result = {"centers": len(rows), "true_order": true_order, "models": {}}
        for model in MODELS:
            predicted = [row[model] for row in rows]
            raw_order = _order(predicted)
            boundary, _ = score_boundaries(ref, predicted_boundaries(predicted, times), tolerance_s=0.4)
            final_order = raw_order
            if model == "boundary_gru":
                frame_id = {"4": "prior_video4", "8": "prior_video8"}.get(clip_id, clip_id)
                full = dense[frame_id]
                chain = adaptive_boundary_gru_chain(
                    [int(row[model]) for row in full],
                    fps=5.0,
                    smoothing_window_seconds=1.0,
                    background_class=0,
                    winner_confidences=[float(row[model + "_confidence"]) for row in full],
                    outlier_z=-2.0,
                    minimum_comparable_segments=4,
                )
                final_order = chain["order"]
            video_result["models"][model] = {
                "raw_order": raw_order,
                "raw_order_match": raw_order == true_order,
                "final_order": final_order,
                "final_order_match": final_order == true_order,
                "output_boundary": boundary.as_dict(),
            }
        per_video[clip_id] = video_result

    for model in MODELS:
        y = [row["truth"] for rows in aligned.values() for row in rows]
        predicted = [row[model] for rows in aligned.values() for row in rows]
        accuracy, macro_f1 = _classification(y, predicted)
        boundary = BoundaryScore()
        for video_result in per_video.values():
            detail = video_result["models"][model]["output_boundary"]
            boundary.true_positive += detail["true_positive"]
            boundary.false_positive += detail["false_positive"]
            boundary.false_negative += detail["false_negative"]
        total_timing_error = sum(
            (v["models"][model]["output_boundary"]["mean_abs_error_s"] or 0.0)
            * v["models"][model]["output_boundary"]["true_positive"]
            for v in per_video.values()
        )
        raw_matches = sum(v["models"][model]["raw_order_match"] for v in per_video.values())
        final_matches = sum(v["models"][model]["final_order_match"] for v in per_video.values())
        chain_rate = final_matches / len(per_video)
        results[model] = {
            "name": NAMES[model],
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "output_boundary_f1": boundary.f1,
            "output_boundary_counts": {
                "tp": boundary.true_positive,
                "fp": boundary.false_positive,
                "fn": boundary.false_negative,
            },
            "output_boundary_mean_abs_error_s": (
                total_timing_error / boundary.true_positive if boundary.true_positive else None
            ),
            "raw_order_matches": raw_matches,
            "final_order_matches": final_matches,
            "composite_score": 0.4 * macro_f1 + 0.3 * chain_rate + 0.3 * boundary.f1,
        }

    ranking = sorted(MODELS, key=lambda model: results[model]["composite_score"], reverse=True)
    report = {
        "scope": "retrospective internal comparison on 10 previously reviewed self-recorded videos",
        "common_centers": len(videomae_rows),
        "exact_time_alignment": True,
        "class_labels": list(range(7)),
        "boundary_protocol": "class-output changes on the same centers; one-to-one match; annotated change interval plus 0.4 s",
        "chain_protocol": "raw order for YOLO, YOLO+GRU and VideoMAE; existing adaptive confirmation for boundary GRU only",
        "composite_weights": {"macro_f1": 0.4, "final_chain_exact_match": 0.3, "output_boundary_f1": 0.3},
        "source_sha256": {
            "frame_predictions": hashlib.sha256(args.frame_predictions.read_bytes()).hexdigest(),
            "videomae_predictions": hashlib.sha256(args.videomae_predictions.read_bytes()).hexdigest(),
        },
        "ranking": ranking,
        "models": results,
        "videos": per_video,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Four-model comparison at shared time centers",
        "",
        "Ten repeatedly reviewed self-recorded videos; 632 exactly aligned valid time centers with identical reference labels.",
        "This is a retrospective internal comparison. VideoMAE uses 16-frame windows; the YOLO family uses predictions at the corresponding times.",
        "",
        "| Rank | Model | Accuracy | Macro F1 | Output boundary F1 | Raw chain | Final chain | Composite |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, model in enumerate(ranking, 1):
        r = results[model]
        lines.append(
            f"| {rank} | {r['name']} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | "
            f"{r['output_boundary_f1']:.2%} | {r['raw_order_matches']}/10 | "
            f"{r['final_order_matches']}/10 | {r['composite_score']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Composite: 40% macro F1 + 30% final-chain exact match + 30% output boundary F1.",
            "Output boundary F1 measures class-output changes for all methods; it is not the boundary-branch score.",
            "Only the boundary-aware GRU uses adaptive segment confirmation; the other final chains equal their raw chains.",
            "Classification, boundaries and raw chains use common centers. The boundary model applies its frozen adaptive confirmation to its dense 5 fps predictions.",
            "Labels and decoding were repeatedly reviewed on these videos; the results do not establish independent generalization.",
            "",
            "| Video | Reference chain | YOLO | YOLO + GRU | Adaptive Boundary-Aware GRU | VideoMAE |",
            "|---|---|---|---|---|---|",
        ]
    )
    for clip_id, item in per_video.items():
        cells = [clip_id, "→".join(map(str, item["true_order"]))]
        for model in MODELS:
            output = item["models"][model]
            cells.append(
                "→".join(map(str, output["final_order"])) + (" ✓" if output["final_order_match"] else " ✗")
            )
        lines.append("| " + " | ".join(cells) + " |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ranking": ranking, "models": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
