#!/usr/bin/env python
"""Train-only hard-negative mining for a short-action proposal verifier.

The frozen boundary-GRU proposes WHO segments. A small classifier learns which
one- or two-centre proposals resemble true annotated segments. Its rejection
threshold is chosen by video-fold out-of-fold predictions on TRAIN, then used
once on the fixed validation split. Test videos are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.core.segment_metrics import ActionSegment, action_segments
from handwash.pipelines.boundary_gru import decode_boundaries, load_boundary_gru
from handwash.pipelines.exp_demo import file_sha256
from scripts.compare_full_video_models import _centres
from scripts.evaluate_dense_boundary_validation import segment_f1
from scripts.full_video_eval_common import RAW_SETTING, public_metrics, score


def _iou(left: ActionSegment, right: ActionSegment) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


def _proposal_features(
    segment: ActionSegment, labels: np.ndarray, raw_probs: np.ndarray,
    decoded_probs: np.ndarray, edges: np.ndarray,
) -> np.ndarray:
    start, end, label = segment.start, segment.end, segment.label
    local = raw_probs[start:end]
    chosen = local[:, label]
    second = np.partition(local, -2, axis=1)[:, -2]
    left = int(labels[start - 1]) if start else 0
    right = int(labels[end]) if end < len(labels) else 0
    values = [
        float(end - start), float(chosen.mean()), float(chosen.min()),
        float(chosen.std()), float((chosen - second).mean()),
        float(decoded_probs[start:end, label].mean()),
        float(edges[start:end].mean()), float(edges[start]),
        float(edges[end]) if end < len(edges) else 0.0,
        float(left == right and left > 0), float(left == label), float(right == label),
        float((start + end) / (2 * len(labels))),
    ]
    values.extend(float(label == step) for step in range(1, 7))
    return np.asarray(values, dtype=np.float32)


def _proposal_truth(segment: ActionSegment, targets: np.ndarray) -> int | None:
    known = targets[segment.start:segment.end] >= 0
    if known.mean() < 0.5:
        return None
    actual = action_segments(targets, targets >= 0)
    return int(any(reference.label == segment.label and _iou(segment, reference) >= 0.5
                   for reference in actual))


def _apply_rejections(item: dict, probabilities: np.ndarray, threshold: float) -> dict:
    original = item["labels"]
    updated = original.copy()
    for proposal, probability in zip(item["proposals"], probabilities, strict=True):
        if probability >= threshold:
            continue
        left = int(original[proposal.start - 1]) if proposal.start else 0
        right = int(original[proposal.end]) if proposal.end < len(original) else 0
        replacement = left if left == right and left > 0 else 0
        updated[proposal.start:proposal.end] = replacement
    return {**item["template"], "probs": np.eye(7, dtype=np.float32)[updated]}


def _score_items(items: list[dict], probabilities: list[np.ndarray], threshold: float) -> dict:
    evaluated = [_apply_rejections(item, probs, threshold)
                 for item, probs in zip(items, probabilities, strict=True)]
    metrics = score(evaluated, RAW_SETTING)
    f1, counts = segment_f1(evaluated, RAW_SETTING)
    return {**public_metrics(metrics), "who_segment_f1_at_50": f1, "segment_counts": counts}


def _fold(video_id: str) -> int:
    return int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % 5


def _load_items(args: argparse.Namespace, head: torch.nn.Module, decoder: dict,
                ids: list[str], *, oof_root: Path | None = None) -> list[dict]:
    items = []
    with torch.inference_mode():
        for number, video_id in enumerate(ids, start=1):
            filename = video_id.replace("/", "__") + ".npz"
            with np.load(args.feature_cache / filename, allow_pickle=False) as cached:
                features = cached["features"].astype(np.float32)
                frame_ids = cached["frame_ids"].astype(np.int64)
                consensus = cached["consensus"].astype(np.int64)
                fps = float(cached["source_fps"])
            _starts, positions = _centres(frame_ids.tolist(), num_frames=16, window_stride=4)
            if oof_root is None:
                logits, edge_logits, _state = head(torch.from_numpy(features).unsqueeze(0))
                raw = logits[0].softmax(-1).numpy()
                edges = edge_logits[0].sigmoid().numpy()
            else:
                with np.load(oof_root / "predictions" / filename, allow_pickle=False) as saved:
                    if (str(saved["video_id"]) != video_id or int(saved["fold"]) != _fold(video_id)
                            or not np.array_equal(saved["frame_ids"], frame_ids)
                            or str(saved["split_sha256"]) != file_sha256(args.split_reference)):
                        raise ValueError(f"OOF prediction alignment or fold mismatch: {video_id}")
                    raw = saved["class_probs"].astype(np.float32)
                    edges = saved["edge_probs"].astype(np.float32)
                if raw.shape != (len(frame_ids), 7) or edges.shape != (len(frame_ids),):
                    raise ValueError(f"OOF prediction shape mismatch: {video_id}")
            decoded = decode_boundaries(raw, edges, threshold=float(decoder["threshold"]),
                                        min_gap=int(decoder["min_gap"]))
            labels = decoded[positions].argmax(axis=1)
            raw_centres, edge_centres = raw[positions], edges[positions]
            targets = consensus[positions]
            proposals = [part for part in action_segments(labels, np.ones(len(labels), dtype=bool))
                         if part.end - part.start <= 2]
            proposal_features = [
                _proposal_features(part, labels, raw_centres, decoded[positions], edge_centres)
                for part in proposals
            ]
            raw_labels = raw_centres.argmax(axis=1)
            existing = {(part.label, part.start, part.end) for part in proposals}
            hard_proposals = [
                part for part in action_segments(raw_labels, np.ones(len(raw_labels), dtype=bool))
                if part.end - part.start <= 2 and (part.label, part.start, part.end) not in existing
            ]
            hard_features = [
                _proposal_features(part, raw_labels, raw_centres, raw_centres, edge_centres)
                for part in hard_proposals
            ]
            template = {
                "video_id": video_id, "probs": decoded[positions],
                "centre_frame_ids": frame_ids[positions], "targets": targets,
                "usable": targets >= 0, "fps": fps,
            }
            items.append({"video_id": video_id, "template": template, "labels": labels,
                          "proposals": proposals, "features": proposal_features,
                          "training_truth": [_proposal_truth(part, targets) for part in proposals],
                          "hard_features": hard_features,
                          "hard_truth": [_proposal_truth(part, targets) for part in hard_proposals]})
            if number % 100 == 0:
                print(f"Prepared {number}/{len(ids)} videos", flush=True)
    return items


def _train_rows(items: list[dict], *, include_raw_hard_negatives: bool) -> tuple[np.ndarray, np.ndarray, list[int]]:
    features = []
    targets = []
    item_indices = []
    for index, item in enumerate(items):
        sources = [(item["features"], item["training_truth"])]
        if include_raw_hard_negatives:
            sources.append((item["hard_features"], item["hard_truth"]))
        for feature_list, truth_list in sources:
            for feature, truth in zip(feature_list, truth_list, strict=True):
                if truth is not None:
                    features.append(feature)
                    targets.append(truth)
                    item_indices.append(index)
    if not features:
        raise ValueError("No short-segment proposals with usable annotations")
    return np.stack(features), np.asarray(targets, dtype=np.int64), item_indices


def _model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=500, C=0.3, random_state=42,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru_20261012/head.pt")
    parser.add_argument("--split-reference", type=Path,
                        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv")
    parser.add_argument("--feature-cache", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--disable-raw-hard-negatives", action="store_true",
                        help="Reproduce the initial decoded-proposals-only pilot")
    parser.add_argument("--oof-root", type=Path,
                        help="Use fold-excluded temporal-head predictions for training proposals")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output: {args.output}")
    head, payload = load_boundary_gru(args.head)
    if payload["exp_sha256"] != file_sha256(PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"):
        parser.error("YOLO checkpoint mismatch")
    head.eval()
    with args.split_reference.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_split = {split: [row["video_id"] for row in rows if row["split"] == split]
                for split in ("train", "val", "test")}
    if [len(by_split[key]) for key in ("train", "val", "test")] != [430, 92, 93]:
        parser.error("Locked 430/92/93 original-video split is required")
    if args.oof_root is not None:
        for fold in range(5):
            meta_path = args.oof_root / f"fold_{fold}" / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta["fold"] != fold or not meta["temporal_head_fold_excluded"]
                    or meta["split_sha256"] != file_sha256(args.split_reference)):
                parser.error(f"Invalid OOF fold metadata: {meta_path}")
    train = _load_items(args, head, payload["decoder"], by_split["train"], oof_root=args.oof_root)
    val = _load_items(args, head, payload["decoder"], by_split["val"])
    x, y, indices = _train_rows(
        train, include_raw_hard_negatives=not args.disable_raw_hard_negatives
    )
    print(f"Short proposals: {len(x)} annotated; true={int(y.sum())}; false={int((1-y).sum())}",
          flush=True)
    if min(int(y.sum()), int((1 - y).sum())) < 10:
        raise ValueError("Too few positive or negative short proposals for a verifier")
    oof = [np.ones(len(item["proposals"]), dtype=np.float32) for item in train]
    fold_rows = []
    for fold in range(5):
        fit = np.asarray([_fold(train[index]["video_id"]) != fold for index in indices])
        held = ~fit
        if not fit.any() or not held.any() or len(np.unique(y[fit])) < 2:
            raise ValueError(f"Fold {fold} lacks both proposal classes")
        model = _model()
        model.fit(x[fit], y[fit])
        for item_index, item in enumerate(train):
            if _fold(item["video_id"]) == fold and item["features"]:
                oof[item_index] = model.predict_proba(np.stack(item["features"]))[:, 1]
        fold_rows.append({"fold": fold, "train_rows": int(fit.sum()), "held_rows": int(held.sum())})
    baseline_train = _score_items(train, oof, 0.0)
    thresholds = [0.0, *(round(float(value), 2) for value in np.arange(0.1, 0.91, 0.1))]
    grid = [{"threshold": threshold, "metrics": _score_items(train, oof, threshold)}
            for threshold in thresholds]
    eligible = [row for row in grid if row["metrics"]["macro_f1"]
                >= baseline_train["macro_f1"] - 0.002]
    chosen = max(eligible, key=lambda row: (
        row["metrics"]["who_segment_f1_at_50"], row["metrics"]["macro_f1"],
        row["metrics"]["sequence_exact_videos"], -row["threshold"],
    ))
    model = _model()
    model.fit(x, y)
    val_probs = [model.predict_proba(np.stack(item["features"]))[:, 1]
                 if item["features"] else np.zeros(0, dtype=np.float32) for item in val]
    baseline_val = _score_items(val, val_probs, 0.0)
    if abs(baseline_val["macro_f1"] - 0.953701893445979) > 1e-5:
        raise ValueError("Validation baseline does not reproduce the archived boundary-GRU score")
    candidate_val = _score_items(val, val_probs, chosen["threshold"])
    result = {
        "method": "frozen boundary-GRU plus logistic short-segment verifier",
        "selection_split": "train video-fold out-of-fold", "evaluation_split": "val",
        "test_used": False, "split_sha256": file_sha256(args.split_reference),
        "base_head_sha256": file_sha256(args.head),
        "training_prediction_source": str(args.oof_root) if args.oof_root else "in_sample_full_head",
        "raw_hard_negatives_included": not args.disable_raw_hard_negatives,
        "train_proposals": {"annotated": len(x), "true": int(y.sum()),
                            "false": int((1-y).sum())},
        "folds": fold_rows, "train_baseline": baseline_train,
        "train_threshold_grid": grid, "chosen_threshold": chosen["threshold"],
        "val_baseline": baseline_val, "val_candidate": candidate_val,
        "promote": (candidate_val["who_segment_f1_at_50"] > baseline_val["who_segment_f1_at_50"]
                    and candidate_val["macro_f1"] >= baseline_val["macro_f1"] - 0.002
                    and candidate_val["sequence_exact_videos"] >= baseline_val["sequence_exact_videos"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    joblib.dump(model, args.output / "verifier.joblib")
    print(json.dumps({key: result[key] for key in (
        "train_proposals", "chosen_threshold", "train_baseline", "val_baseline",
        "val_candidate", "promote",
    )}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
