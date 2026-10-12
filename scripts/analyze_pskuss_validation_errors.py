#!/usr/bin/env python
"""Diagnose the archived YOLO and boundary-GRU predictions on validation only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES


def _near_transition(
    targets: np.ndarray, usable: np.ndarray, frame_ids: np.ndarray, fps: float
) -> np.ndarray:
    near = np.zeros(len(usable), dtype=bool)
    for index in range(1, len(usable)):
        gap = (frame_ids[index] - frame_ids[index - 1]) / fps
        if usable[index - 1] and usable[index] and targets[index - 1] != targets[index] and gap <= 1.6:
            near[max(0, index - 2) : min(len(usable), index + 2)] = True
    return near & usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_validation_error_analysis.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    roots = {name: args.comparison / name for name in ("yolo", "boundary_gru")}
    summaries = {
        name: json.loads((root / "summary.json").read_text(encoding="utf-8")) for name, root in roots.items()
    }
    video_ids = summaries["yolo"]["video_ids"]["val"]
    if len(video_ids) != 92 or summaries["boundary_gru"]["video_ids"]["val"] != video_ids:
        parser.error("Expected identical lists of 92 validation videos")
    if summaries["yolo"]["split_reference_sha256"] != summaries["boundary_gru"]["split_reference_sha256"]:
        parser.error("Split reference differs between methods")

    truth_all: list[np.ndarray] = []
    yolo_all: list[np.ndarray] = []
    boundary_all: list[np.ndarray] = []
    near_all: list[np.ndarray] = []
    source_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    per_video: list[dict] = []
    for video_id in video_ids:
        filename = video_id.replace("/", "__") + ".npz"
        with (
            np.load(roots["yolo"] / "raw_probabilities" / filename, allow_pickle=False) as yolo,
            np.load(roots["boundary_gru"] / "raw_probabilities" / filename, allow_pickle=False) as boundary,
        ):
            for field in ("centre_frame_ids", "targets", "usable", "fps"):
                if not np.array_equal(yolo[field], boundary[field]):
                    raise ValueError(f"Prediction alignment differs for {video_id}: {field}")
            keep = yolo["usable"].astype(bool)
            target = yolo["targets"][keep].astype(np.int64)
            yolo_prob = yolo["probs"][keep]
            first = yolo_prob.argmax(axis=1)
            second = boundary["probs"][keep].argmax(axis=1)
            near = _near_transition(yolo["targets"], keep, yolo["centre_frame_ids"], float(yolo["fps"]))[keep]
            truth_all.append(target)
            yolo_all.append(first)
            boundary_all.append(second)
            near_all.append(near)
            source_all.append(np.repeat(video_id.split("/", 1)[0], len(target)))
            confidence_all.append(yolo_prob.max(axis=1))
            per_video.append(
                {
                    "video_id": video_id,
                    "centres": len(target),
                    "yolo_errors": int(np.count_nonzero(first != target)),
                    "boundary_errors": int(np.count_nonzero(second != target)),
                    "corrected_by_boundary": int(np.count_nonzero((first != target) & (second == target))),
                    "introduced_by_boundary": int(np.count_nonzero((first == target) & (second != target))),
                }
            )

    truth = np.concatenate(truth_all)
    yolo = np.concatenate(yolo_all)
    boundary = np.concatenate(boundary_all)
    near = np.concatenate(near_all)
    sources = np.concatenate(source_all)
    confidence = np.concatenate(confidence_all)
    labels = list(range(7))
    incorrect = yolo != truth
    pairs = Counter((int(t), int(p)) for t, p in zip(truth[incorrect], yolo[incorrect], strict=True))
    source_metrics = {}
    for source in sorted(set(sources)):
        mask = sources == source
        source_metrics[source] = {
            "centres": int(mask.sum()),
            "yolo_errors": int(np.count_nonzero(incorrect & mask)),
            "boundary_errors": int(np.count_nonzero((boundary != truth) & mask)),
        }
    result = {
        "split": "val",
        "videos": len(video_ids),
        "centres": len(truth),
        "split_reference_sha256": summaries["yolo"]["split_reference_sha256"],
        "classes": list(EXP_CLASS_NAMES),
        "yolo_per_class": classification_report(
            truth, yolo, labels=labels, output_dict=True, zero_division=0
        ),
        "boundary_per_class": classification_report(
            truth, boundary, labels=labels, output_dict=True, zero_division=0
        ),
        "yolo_confusion": confusion_matrix(truth, yolo, labels=labels).tolist(),
        "boundary_confusion": confusion_matrix(truth, boundary, labels=labels).tolist(),
        "top_yolo_confusions": [
            {"true": true, "predicted": predicted, "count": count}
            for (true, predicted), count in pairs.most_common(12)
        ],
        "regions": {
            name: {
                "centres": int(mask.sum()),
                "yolo_errors": int(np.count_nonzero(incorrect & mask)),
                "boundary_errors": int(np.count_nonzero((boundary != truth) & mask)),
                "corrected_by_boundary": int(np.count_nonzero(incorrect & (boundary == truth) & mask)),
                "introduced_by_boundary": int(np.count_nonzero((yolo == truth) & (boundary != truth) & mask)),
            }
            for name, mask in (("transition", near), ("away_from_transition", ~near))
        },
        "sources": source_metrics,
        "yolo_error_confidence": {
            "median": float(np.median(confidence[incorrect])),
            "at_least_0_8": int(np.count_nonzero(incorrect & (confidence >= 0.8))),
            "at_least_0_9": int(np.count_nonzero(incorrect & (confidence >= 0.9))),
        },
        "videos_with_most_yolo_errors": sorted(per_video, key=lambda row: row["yolo_errors"], reverse=True)[
            :10
        ],
        "test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("regions", "sources", "top_yolo_confusions")}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
