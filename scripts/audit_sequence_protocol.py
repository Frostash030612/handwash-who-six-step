#!/usr/bin/env python
"""Audit temporal action segments on the archived, aligned PSKUS videos.

The existing strict sequence exact-match is preserved. Segmental F1@50 is
reported separately: it evaluates each WHO action instance by its overlap with
one human-labelled instance, so a single insertion does not invalidate every
other correctly identified step in the clip.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.core.segment_metrics import action_segments, segment_counts
from scripts.full_video_eval_common import _predictions, _sequence, score

MODEL_NAMES = ("yolo", "videomae_center", "yolo_gru", "boundary_gru")


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def evaluate_model(comparison: Path, model_name: str, baseline: dict, split: str) -> dict:
    root = comparison / model_name
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["split_reference_sha256"] != baseline["split_reference_sha256"]:
        raise ValueError(f"Split reference differs for {model_name}")
    if summary["video_ids"][split] != baseline["video_ids"][split]:
        raise ValueError(f"Video IDs differ for {model_name}/{split}")
    setting = summary["chosen_on_validation"]
    aggregate = Counter()
    by_class = {label: Counter() for label in range(1, 7)}
    items: list[dict] = []
    per_video: list[dict] = []
    for video_id in summary["video_ids"][split]:
        filename = video_id.replace("/", "__") + ".npz"
        with (
            np.load(root / "raw_probabilities" / filename, allow_pickle=False) as saved,
            np.load(comparison / "yolo" / "raw_probabilities" / filename, allow_pickle=False) as anchor,
        ):
            item = {
                key: saved[key].copy() for key in ("probs", "centre_frame_ids", "targets", "usable", "fps")
            }
            for key in ("centre_frame_ids", "targets", "usable"):
                if not np.array_equal(item[key], anchor[key]):
                    raise ValueError(f"Unaligned {key}: {model_name}/{video_id}")
        labels = _predictions(item["probs"], setting)
        keep = item["usable"].astype(bool)
        truth_sequence = _sequence(item["targets"][keep])
        predicted_sequence = _sequence(labels[keep])
        true_segments = action_segments(item["targets"], keep)
        predicted_segments = action_segments(labels, keep)
        tp, fp, fn = segment_counts(true_segments, predicted_segments)
        aggregate.update({"tp": tp, "fp": fp, "fn": fn})
        aggregate["videos"] += 1
        aggregate["centres"] += int(keep.sum())
        if truth_sequence:
            aggregate["videos_with_who_reference"] += 1
            aggregate["nonempty_sequence_exact"] += truth_sequence == predicted_sequence
        else:
            aggregate["videos_without_who_reference"] += 1
            aggregate["no_who_false_positive"] += bool(predicted_sequence)
        for label in range(1, 7):
            result = segment_counts(
                [part for part in true_segments if part.label == label],
                [part for part in predicted_segments if part.label == label],
            )
            by_class[label].update(dict(zip(("tp", "fp", "fn"), result, strict=True)))
        per_video.append(
            {
                "video_id": video_id,
                "reference_sequence": truth_sequence,
                "predicted_sequence": predicted_sequence,
                "sequence_exact": truth_sequence == predicted_sequence,
                "segment_tp": tp,
                "segment_fp": fp,
                "segment_fn": fn,
            }
        )
        items.append(item)
    historical = score(items, setting)
    if (
        split == "test"
        and historical["sequence_exact_videos"] != summary["test_smoothed"]["sequence_exact_videos"]
    ):
        raise ValueError(f"Historical exact-match changed for {model_name}")
    return {
        "model": model_name,
        "split": split,
        "chosen_on_validation": setting,
        "historical": {
            key: historical[key]
            for key in ("macro_f1", "transition_macro_f1", "sequence_exact_videos", "sequence_exact_rate")
        },
        "who_segment_f1_at_50": _f1(aggregate["tp"], aggregate["fp"], aggregate["fn"]),
        "segment_counts": {key: aggregate[key] for key in ("tp", "fp", "fn")},
        "per_class_segment_f1_at_50": {
            str(label): _f1(by_class[label]["tp"], by_class[label]["fp"], by_class[label]["fn"])
            for label in range(1, 7)
        },
        "nonempty_sequence_exact": aggregate["nonempty_sequence_exact"],
        "videos_with_who_reference": aggregate["videos_with_who_reference"],
        "no_who_false_positive": aggregate["no_who_false_positive"],
        "videos_without_who_reference": aggregate["videos_without_who_reference"],
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    anchor = json.loads((args.comparison / "yolo" / "summary.json").read_text(encoding="utf-8"))
    results = {
        split: {name: evaluate_model(args.comparison, name, anchor, split) for name in MODEL_NAMES}
        for split in ("val", "test")
    }
    result = {
        "definition": {
            "unit": "WHO action instances on aligned two-annotator-consensus video centres",
            "temporal_overlap": "centre-index IoU >= 0.50, one-to-one, same class",
            "other_and_unannotated": "Other (0) and unusable centres break segments; neither is a WHO segment",
            "model_selection": "No parameters selected from test; historical validation-selected decoders are unchanged",
            "clinical_compliance": "Not a complete WHO six-step compliance accuracy",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                split: {
                    name: {
                        key: result["results"][split][name][key]
                        for key in (
                            "who_segment_f1_at_50",
                            "segment_counts",
                            "nonempty_sequence_exact",
                            "videos_with_who_reference",
                            "no_who_false_positive",
                            "videos_without_who_reference",
                        )
                    }
                    for name in MODEL_NAMES
                }
                for split in ("val", "test")
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
