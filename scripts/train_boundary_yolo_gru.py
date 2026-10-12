#!/usr/bin/env python
"""Fine-tune the existing YOLO+GRU head with dual-annotator boundary supervision.

This is a new checkpoint and never overwrites the YOLO or GRU baselines. Model
selection and boundary-decoder tuning use train/validation videos only; the
locked test annotations are not scored by this trainer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:  # Direct script execution and ``python -m scripts.train_boundary_yolo_gru``.
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover - exercised by module-style imports
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.boundary_gru import (
    BOUNDARY_HEAD_FORMAT,
    BoundaryAwareGRU,
    boundary_targets_from_consensus,
    decode_boundaries,
    weak_boundary_intervals_from_consensus,
    weak_interval_loss,
)
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, ExpFeatureExtractor, file_sha256
from scripts.compare_full_video_models import _centres, _item
from scripts.full_video_eval_common import RAW_SETTING, score, videos_from_reference
from scripts.train_current_yolo_gru import _read_or_extract


def _crop_batch(
    batch: list[dict], rng: random.Random, *, min_length: int = 64, max_length: int = 160,
    transition_supervision_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    crops: list[tuple[dict, int, int]] = []
    for item in batch:
        length = len(item["features"])
        size = min(length, rng.randint(min_length, max_length))
        stable_positions = np.flatnonzero(item["stable"] > 0)
        boundary_positions = np.flatnonzero(item["boundary_targets"] > 0)
        weak_intervals = item.get("weak_boundary_intervals", [])
        selection = rng.random()
        if weak_intervals and selection < 0.25:
            interval = rng.choice(weak_intervals)
            focus = rng.randint(interval[0], interval[1] - 1)
            start = min(max(0, focus - rng.randint(0, size - 1)), length - size)
        elif len(boundary_positions) and selection < 0.5:
            focus = int(rng.choice(boundary_positions.tolist()))
            start = min(max(0, focus - rng.randint(0, size - 1)), length - size)
        elif len(stable_positions) and rng.random() < 0.75:
            focus = int(rng.choice(stable_positions.tolist()))
            start = min(max(0, focus - rng.randint(0, size - 1)), length - size)
        else:
            start = rng.randint(0, length - size)
        crops.append((item, start, size))

    longest = max(size for _item, _start, size in crops)
    feature_dim = int(batch[0]["features"].shape[1])
    features = torch.zeros(len(batch), longest, feature_dim)
    class_targets = torch.full((len(batch), longest), -1, dtype=torch.long)
    class_supervision = torch.zeros(len(batch), longest, dtype=torch.float32)
    edge_targets = torch.zeros(len(batch), longest, dtype=torch.float32)
    edge_mask = torch.zeros(len(batch), longest, dtype=torch.bool)
    weak_intervals_in_crop: list[tuple[int, int, int]] = []
    for row, (item, start, size) in enumerate(crops):
        end = start + size
        features[row, :size] = torch.from_numpy(item["features"][start:end].astype(np.float32))
        stable = item["stable"][start:end].astype(np.int64)
        consensus = item["consensus"][start:end].astype(np.int64)
        class_targets[row, :size] = torch.from_numpy(
            np.where(stable >= 0, stable, consensus if transition_supervision_weight > 0 else -1)
        )
        class_supervision[row, :size] = torch.from_numpy(
            np.where(stable >= 0, 1.0,
                     np.where(consensus >= 0, transition_supervision_weight, 0.0)).astype(np.float32)
        )
        edge_targets[row, :size] = torch.from_numpy(item["boundary_targets"][start:end].astype(np.float32))
        edge_mask[row, :size] = torch.from_numpy(item["boundary_mask"][start:end])
        # The start of a random crop is not necessarily a true sequence start.
        edge_mask[row, 0] = False
        for left, right in item.get("weak_boundary_intervals", []):
            if start < left and right <= end:
                weak_intervals_in_crop.append((row, left - start, right - start))
    return features, class_targets, class_supervision, edge_targets, edge_mask, weak_intervals_in_crop


def _prepare_eval_templates(data: list[dict], video_by_id: dict[str, object]) -> list[dict]:
    templates: list[dict] = []
    for item in data:
        key = item["video_id"]
        video = video_by_id[key]
        frame_ids = np.asarray(item["frame_ids"], dtype=np.int64).tolist()
        _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
        if not len(positions):
            continue
        template = _item(video, np.zeros((len(positions), len(EXP_CLASS_NAMES)), dtype=np.float32),
                         frame_ids, positions, float(item["source_fps"]))
        templates.append({"cache_item": item, "positions": positions, "template": template})
    return templates


@torch.inference_mode()
def _predict_items(head: BoundaryAwareGRU, templates: list[dict], device: str) -> list[dict]:
    head.eval()
    predicted: list[dict] = []
    for entry in templates:
        item = entry["cache_item"]
        features = torch.from_numpy(item["features"].astype(np.float32)).unsqueeze(0).to(device)
        class_logits, edge_logits, _state = head(features)
        frame_probs = class_logits[0].softmax(-1).cpu().numpy()
        edge_probs = edge_logits[0].sigmoid().cpu().numpy()
        template = entry["template"]
        predicted.append({
            **template,
            "frame_probs": frame_probs,
            "edge_probs": edge_probs,
            "positions": entry["positions"],
        })
    return predicted


def _score_predictions(predictions: list[dict], decoder: dict) -> dict:
    evaluated = []
    for item in predictions:
        frame_probs = item["frame_probs"]
        if decoder["method"] == "boundary_segment":
            frame_probs = decode_boundaries(
                frame_probs, item["edge_probs"], threshold=float(decoder["threshold"]),
                min_gap=int(decoder["min_gap"]),
            )
        centre_probs = frame_probs[item["positions"]]
        evaluated.append({**item, "probs": centre_probs})
    return score(evaluated, RAW_SETTING)


def _validation_grid(predictions: list[dict]) -> tuple[dict, list[dict]]:
    settings = [{"method": "raw", "threshold": None, "min_gap": None}]
    settings.extend(
        {"method": "boundary_segment", "threshold": threshold, "min_gap": gap}
        for threshold in (0.25, 0.4, 0.55, 0.7, 0.85)
        for gap in (2, 3, 4)
    )
    grid: list[dict] = []
    for setting in settings:
        score_setting = (setting if setting["method"] == "boundary_segment"
                         else {"method": "raw"})
        result = _score_predictions(predictions, score_setting)
        grid.append({**setting, **{key: result[key] for key in (
            "accuracy", "macro_f1", "transition_macro_f1", "sequence_exact_rate",
        )}})
    best = max(grid, key=lambda row: (
        row["macro_f1"], row["transition_macro_f1"] or 0.0, row["accuracy"],
        row["method"] == "raw",
    ))
    return best, grid


def _load_baseline_state(baseline_path: Path) -> tuple[dict, dict]:
    payload = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES:
        raise ValueError("Baseline GRU class order does not match the seven-class protocol")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Missing GRU state_dict: {baseline_path}")
    return payload.get("hparams", {}), state


def _copy_baseline_state(head: BoundaryAwareGRU, state: dict) -> None:
    if head.hparams["bidirectional"]:
        initialized = head.state_dict()
        for key, value in state.items():
            if key == "classifier.weight":
                if initialized[key].shape[1] != 2 * value.shape[1]:
                    raise ValueError("Bidirectional classifier does not match the baseline hidden size")
                initialized[key][:, : value.shape[1]] = value
                initialized[key][:, value.shape[1] :] = 0
            elif key in initialized and initialized[key].shape == value.shape:
                initialized[key] = value.clone()
            else:
                raise ValueError(f"Unexpected baseline GRU parameter: {key}")
        for suffix in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            initialized[f"rnn.{suffix}_reverse"] = state[f"rnn.{suffix}"].clone()
        head.load_state_dict(initialized, strict=True)
        return
    missing, unexpected = head.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {
        "boundary_classifier.0.weight", "boundary_classifier.0.bias",
        "boundary_classifier.3.weight", "boundary_classifier.3.bias",
    }:
        raise ValueError(f"Unexpected baseline GRU checkpoint structure: missing={missing}, unexpected={unexpected}")


def _segment_consistency_loss(class_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reduce within-action flicker, without coupling across unknown gaps or boundaries.

    Only contiguous, two-annotator-agreed stable runs are regularized. Short
    runs are left intact, so the loss cannot encode a minimum action duration.
    """
    probabilities = class_logits.softmax(dim=-1)
    total = class_logits.new_zeros(())
    count = 0
    for row in range(targets.shape[0]):
        start = 0
        while start < targets.shape[1]:
            label = int(targets[row, start])
            end = start + 1
            while end < targets.shape[1] and int(targets[row, end]) == label:
                end += 1
            if label >= 0 and end - start >= 2:
                run = probabilities[row, start:end]
                centre = run.mean(dim=0, keepdim=True).detach()
                total = total + (run - centre).square().sum()
                count += end - start
            start = end
    return total / max(count, 1)


def _train(
    data: list[dict], val_templates: list[dict], *, epochs: int, batch_size: int,
    patience: int, seed: int, device: str, boundary_weight: float,
    boundary_positive_weight: float | None,
    bidirectional: bool,
    gap_weak_loss_weight: float,
    repeat_minority_sources: bool,
    temporal_consistency_weight: float,
    segment_consistency_weight: float,
    transition_supervision_weight: float,
    baseline_hparams: dict, baseline_state: dict,
) -> tuple[BoundaryAwareGRU, list[dict], int, float]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    feature_dim = int(data[0]["features"].shape[1])
    head = BoundaryAwareGRU(**{**baseline_hparams, "feature_dim": feature_dim,
                               "bidirectional": bidirectional})
    _copy_baseline_state(head, baseline_state)
    counts = Counter(int(label) for item in data for label in item["stable"] if label >= 0)
    class_weights = torch.tensor(
        [1 / max(counts.get(index, 0), 1) ** 0.5 for index in range(7)], dtype=torch.float32
    )
    class_weights /= class_weights.mean()
    positives = sum(int(item["boundary_targets"].sum()) for item in data)
    known_edges = sum(int(item["boundary_mask"].sum()) for item in data)
    negatives = max(known_edges - positives, 1)
    edge_pos_weight = (boundary_positive_weight if boundary_positive_weight is not None
                       else min(20.0, max(1.0, float(np.sqrt(negatives / max(positives, 1))))))

    head = head.to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(head.encoder.parameters()) + list(head.rnn.parameters()) + list(head.classifier.parameters()),
         "lr": 2e-4},
        {"params": head.boundary_classifier.parameters(), "lr": 8e-4},
    ], weight_decay=1e-2)
    history: list[dict] = []
    best_state = None
    best_epoch = 0
    best_f1 = -1.0
    stale = 0
    source_counts = Counter(item["video_id"].split("/", 1)[0] for item in data)
    majority_source = source_counts.most_common(1)[0][0]

    for epoch in range(1, epochs + 1):
        order = list(data)
        if repeat_minority_sources:
            order.extend(item for item in data
                         if item["video_id"].split("/", 1)[0] != majority_source)
        rng.shuffle(order)
        head.train()
        losses: list[float] = []
        class_losses: list[float] = []
        boundary_losses: list[float] = []
        weak_losses: list[float] = []
        consistency_losses: list[float] = []
        segment_losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            features, class_target, class_supervision, edge_target, edge_mask, weak_intervals = _crop_batch(
                order[offset:offset + batch_size], rng,
                transition_supervision_weight=transition_supervision_weight,
            )
            features = features.to(device)
            class_target = class_target.to(device)
            class_supervision = class_supervision.to(device)
            edge_target = edge_target.to(device)
            edge_mask = edge_mask.to(device)
            class_logits, edge_logits, _state = head(features)

            valid_class = class_target >= 0
            if not valid_class.any():
                continue
            class_loss_values = nn.functional.cross_entropy(
                class_logits.reshape(-1, 7), class_target.clamp_min(0).reshape(-1),
                weight=class_weights, reduction="none",
            ).reshape_as(class_target)
            class_loss = (class_loss_values * class_supervision).sum() / class_supervision.sum().clamp_min(1)
            edge_loss_values = nn.functional.binary_cross_entropy_with_logits(
                edge_logits, edge_target,
                pos_weight=torch.tensor(edge_pos_weight, dtype=torch.float32, device=device),
                reduction="none",
            )
            edge_loss = ((edge_loss_values * edge_mask).sum() / edge_mask.sum().clamp_min(1))
            weak_loss = weak_interval_loss(edge_logits, weak_intervals)
            class_probabilities = class_logits.softmax(dim=-1)
            same_known = edge_mask[:, 1:] & (edge_target[:, 1:] == 0)
            consistency_values = (
                (class_probabilities[:, 1:] - class_probabilities[:, :-1]).square().sum(dim=-1)
            )
            consistency_loss = (
                (consistency_values * same_known).sum() / same_known.sum().clamp_min(1)
            )
            segment_loss = _segment_consistency_loss(class_logits, class_target)
            loss = (class_loss + boundary_weight * (edge_loss + gap_weak_loss_weight * weak_loss)
                    + temporal_consistency_weight * consistency_loss
                    + segment_consistency_weight * segment_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            class_losses.append(float(class_loss.detach().item()))
            boundary_losses.append(float(edge_loss.detach().item()))
            weak_losses.append(float(weak_loss.detach().item()))
            consistency_losses.append(float(consistency_loss.detach().item()))
            segment_losses.append(float(segment_loss.detach().item()))

        validation = _score_predictions(_predict_items(head, val_templates, device), {"method": "raw"})
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "class_loss": float(np.mean(class_losses)) if class_losses else None,
            "boundary_loss": float(np.mean(boundary_losses)) if boundary_losses else None,
            "weak_boundary_loss": float(np.mean(weak_losses)) if weak_losses else None,
            "temporal_consistency_loss": (float(np.mean(consistency_losses)) if consistency_losses else None),
            "segment_consistency_loss": (float(np.mean(segment_losses)) if segment_losses else None),
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_transition_macro_f1": validation["transition_macro_f1"],
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if entry["val_macro_f1"] > best_f1 + 1e-5:
            best_f1, best_epoch, stale = entry["val_macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("Boundary-GRU training did not produce a validation checkpoint")
    head.load_state_dict(best_state)
    return head.cpu().eval(), history, best_epoch, edge_pos_weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--split-reference", type=Path,
                        default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo/video_splits.csv")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt")
    parser.add_argument("--baseline-head", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/head.pt")
    parser.add_argument("--cache-root", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_boundary_gru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--stable-radius", type=int, default=7)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--boundary-positive-weight", type=float, default=None,
                        help="Override positive boundary loss weight; default is sqrt(negative/positive)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use future context for offline full-video classification")
    parser.add_argument("--gap-max", type=int, default=3,
                        help="Maximum number of unknown samples in a weak boundary interval")
    parser.add_argument("--gap-weak-loss-weight", type=float, default=0.0,
                        help="Additional multiple-instance loss weight for short uncertain gaps")
    parser.add_argument("--repeat-minority-sources", action="store_true",
                        help="Repeat non-majority source videos once per epoch without dropping any video")
    parser.add_argument("--temporal-consistency-weight", type=float, default=0.0,
                        help="Penalize output changes only between adjacent, agreed same-class labels")
    parser.add_argument("--segment-consistency-weight", type=float, default=0.0,
                        help="Penalize within-action probability flicker on agreed stable runs")
    parser.add_argument("--transition-supervision-weight", type=float, default=0.0,
                        help="Give agreed but boundary-near class labels this relative training weight")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch-size must be positive")
    if not 0 < args.boundary_weight <= 2:
        parser.error("boundary-weight must be in (0, 2]")
    if args.boundary_positive_weight is not None and args.boundary_positive_weight <= 0:
        parser.error("boundary-positive-weight must be positive")
    if (args.gap_max < 1 or args.gap_weak_loss_weight < 0 or args.temporal_consistency_weight < 0
            or args.segment_consistency_weight < 0):
        parser.error("gap-max must be positive; weak and consistency weights must be nonnegative")
    if not 0 <= args.transition_supervision_weight <= 1:
        parser.error("transition-supervision-weight must be in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {args.output}")
    for required in (args.split_reference, args.model, args.baseline_head):
        if not required.is_file():
            parser.error(f"Required file is missing: {required}")

    all_videos = videos_from_reference(args.root, args.split_reference)
    videos = [video for video in all_videos if video.split in {"train", "val"}]
    counts = {split: sum(video.split == split for video in videos) for split in ("train", "val")}
    if min(counts.values()) == 0:
        parser.error(f"Training or validation videos are missing: {counts}")
    print(f"Locked video split (test remains untouched): {counts}", flush=True)

    model_sha = file_sha256(args.model)
    feature_rows: list[dict] = []
    extractor: ExpFeatureExtractor | None = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        item, extractor = _read_or_extract(
            video, extractor, args.model, model_sha, args.cache_root,
            args.sample_fps, args.stable_radius, device,
        )
        boundary_targets, boundary_mask = boundary_targets_from_consensus(item["consensus"])
        item["boundary_targets"] = boundary_targets
        item["boundary_mask"] = boundary_mask
        item["weak_boundary_intervals"] = (
            weak_boundary_intervals_from_consensus(item["consensus"], max_gap=args.gap_max)
            if args.gap_weak_loss_weight else []
        )
        feature_rows.append(item)
        print(f"Feature {number}/{len(videos)} {item['split']} {item['video_id']} "
              f"samples={len(item['frame_ids'])} stable={int((item['stable'] >= 0).sum())} "
              f"boundaries={int(boundary_targets.sum())}", flush=True)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_data = [item for item in feature_rows if item["split"] == "train"]
    val_data = [item for item in feature_rows if item["split"] == "val"]
    video_by_id = {f"{video.source}/{video.stem}": video for video in videos}
    val_templates = _prepare_eval_templates(val_data, video_by_id)

    baseline_hparams, baseline_state = _load_baseline_state(args.baseline_head)
    if int(baseline_hparams.get("feature_dim", -1)) != int(train_data[0]["features"].shape[1]):
        parser.error("Baseline GRU feature dimension does not match cached YOLO features")

    # Kept separate from the long-lived baseline checkpoint; the helper enforces a fresh output.
    args.output.mkdir(parents=True, exist_ok=True)
    boundary_head, history, best_epoch, edge_pos_weight = _train(
        data=train_data, val_templates=val_templates, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience, seed=args.seed,
        device=device, boundary_weight=args.boundary_weight,
        boundary_positive_weight=args.boundary_positive_weight,
        bidirectional=args.bidirectional,
        gap_weak_loss_weight=args.gap_weak_loss_weight,
        repeat_minority_sources=args.repeat_minority_sources,
        temporal_consistency_weight=args.temporal_consistency_weight,
        segment_consistency_weight=args.segment_consistency_weight,
        transition_supervision_weight=args.transition_supervision_weight,
        baseline_hparams=baseline_hparams, baseline_state=baseline_state,
    )
    val_predictions = _predict_items(boundary_head.to(device), val_templates, device)
    decoder, grid = _validation_grid(val_predictions)
    raw_validation = _score_predictions(val_predictions, {"method": "raw"})
    selected_validation = _score_predictions(val_predictions, decoder)
    metadata = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1_raw": raw_validation["macro_f1"],
        "selected_validation": {key: selected_validation[key] for key in (
            "videos", "evaluated_centres", "accuracy", "macro_f1", "transition_centres",
            "transition_accuracy", "transition_macro_f1", "sequence_exact_videos", "sequence_exact_rate",
        )},
        "decoder": decoder,
        "split_reference": str(args.split_reference),
        "split_reference_sha256": file_sha256(args.split_reference),
        "video_counts": counts,
        "sample_fps": args.sample_fps,
        "stable_radius": args.stable_radius,
        "boundary_loss_weight": args.boundary_weight,
        "boundary_positive_weight": edge_pos_weight,
        "bidirectional": args.bidirectional,
        "gap_max": args.gap_max,
        "gap_weak_loss_weight": args.gap_weak_loss_weight,
        "weak_boundary_interval_count_train": sum(len(item["weak_boundary_intervals"]) for item in train_data),
        "repeat_minority_sources": args.repeat_minority_sources,
        "temporal_consistency_weight": args.temporal_consistency_weight,
        "segment_consistency_weight": args.segment_consistency_weight,
        "transition_supervision_weight": args.transition_supervision_weight,
        "boundary_transition_count_train": sum(int(item["boundary_targets"].sum()) for item in train_data),
        "yolo_model_sha256": model_sha,
        "baseline_gru": str(args.baseline_head),
        "feature_cache_root": str(args.cache_root),
        "elapsed_s": time.perf_counter() - started,
        "test_used_for_selection": False,
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "validation.json").write_text(json.dumps({
        "raw": {key: raw_validation[key] for key in (
            "videos", "evaluated_centres", "accuracy", "macro_f1", "transition_centres",
            "transition_accuracy", "transition_macro_f1", "sequence_exact_videos", "sequence_exact_rate",
        )},
        "selected": metadata["selected_validation"],
        "decoder_grid": grid,
    }, indent=2), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save({
        "format": BOUNDARY_HEAD_FORMAT,
        "class_names": list(EXP_CLASS_NAMES),
        "hparams": dict(boundary_head.hparams),
        "state_dict": {key: value.detach().cpu() for key, value in boundary_head.state_dict().items()},
        "exp_sha256": model_sha,
        "decoder": decoder,
        "metrics": metadata,
    }, args.output / "head.pt")
    print(f"Saved boundary-aware GRU: {args.output / 'head.pt'}; best epoch={best_epoch}; "
          f"val Macro-F1={selected_validation['macro_f1']:.4f} "
          f"(raw={raw_validation['macro_f1']:.4f}); decoder={decoder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
