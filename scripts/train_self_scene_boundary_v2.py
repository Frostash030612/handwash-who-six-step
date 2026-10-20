#!/usr/bin/env python
"""Retrain GRU and boundary-aware GRU heads on the self-recorded scene.

Stages (``--stage``, default ``all``):

* ``features``: frozen-YOLO features for every listed clip, extracted once.
* ``train``: every seed x {gru, joint (A), detached (B)}. Only the fit clips train
  heads; the best validation checkpoint is saved together with the decoder chosen
  for it on the same validation clips. Nothing is refit with validation clips.
* ``evaluate``: validation and regression report. The 2026-10-19 heads were refit
  on validation clips, so they are compared on the regression videos only.

Regression videos are never read by training or selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn

try:  # Direct script execution and ``python -m scripts.train_self_scene_boundary_v2``.
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover - exercised by module-style imports
    from ._bootstrap import PROJECT_ROOT
from handwash.core.boundary_decoding import apply_decoder, boundary_peaks
from handwash.core.boundary_metrics import (
    BoundaryScore,
    interval_offset_s,
    predicted_boundaries,
    reference_boundaries,
    score_boundaries,
)
from handwash.core.seeding import seed_everything
from handwash.core.temporal_selection import composite_score, replacement_decision
from handwash.pipelines.action_chain import sustained_action_chain
from handwash.pipelines.boundary_gru import (
    BOUNDARY_HEAD_FORMAT,
    BoundaryAwareGRU,
    boundary_targets_from_consensus,
    weak_boundary_intervals_from_consensus,
    weak_interval_loss,
)
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, ExpTemporalHead, file_sha256, save_temporal_head

CLASS_COUNT = len(EXP_CLASS_NAMES)
CANONICAL_ORDER = [1, 2, 3, 4, 5, 6]
CODES = {str(code) for code in range(CLASS_COUNT)}
IGNORE = -100
SUMMARY_KEYS = (
    "accuracy", "macro_f1", "raw_accuracy", "raw_macro_f1",
    "continuous_chain_match_rate", "isolated_chain_match_rate",
    "boundary_f1", "boundary_precision", "boundary_recall",
    "boundary_mean_abs_error_s", "boundary_mean_signed_error_s",
    "boundary_f1_after_chain_smoothing", "composite",
)


# --------------------------------------------------------------------------- io helpers
def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _plain(value: Any) -> Any:
    """Plain Python values, safe for JSON files and ``weights_only`` checkpoints."""
    return json.loads(json.dumps(value, default=_json_default))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                    encoding="utf-8")


def read_labels(path: Path) -> dict[str, Any]:
    """Per-sample labels (``-100`` = unlabelled transition), times and image names.

    Clip 1 uses an older export (``movement_code_train``, ``frame_time_ms``).
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No samples in {path}")
    labels, times, images = [], [], []
    for row in rows:
        code = (row.get("movement_code_train") or row.get("movement_code") or "").strip()
        valid = (row.get("valid_for_training") or "").strip().lower() == "true" and code in CODES
        labels.append(int(code) if valid else IGNORE)
        if row.get("actual_time_s"):
            times.append(float(row["actual_time_s"]))
        elif row.get("frame_time_ms"):
            times.append(float(row["frame_time_ms"]) / 1000.0)
        else:
            times.append(float(row["target_time_s"]))
        images.append(row["image"])
    return {"labels": np.asarray(labels, dtype=np.int64), "times": np.asarray(times, dtype=np.float64),
            "images": images}


def _label_path(scene_root: Path, clip_id: int) -> Path:
    return scene_root / "frames" / f"{clip_id}_frames" / "frame_labels.csv"


def _all_clip_ids(cfg: dict) -> list[int]:
    splits = cfg["splits"]
    ids = [*splits["fit"], *splits["validation"], *splits["regression"]]
    if len(set(ids)) != len(ids):
        raise ValueError("A clip appears in more than one split")
    return [int(clip_id) for clip_id in ids]


def collapse(labels: Any) -> list[int]:
    """WHO step order: drop other/unlabelled samples, merge adjacent repeats."""
    order: list[int] = []
    for value in labels:
        value = int(value)
        if value > 0 and (not order or order[-1] != value):
            order.append(value)
    return order


# --------------------------------------------------------------------------- features
def extract_features(scene_root: Path, cfg: dict, output: Path, device: str) -> None:
    import cv2
    import ultralytics

    from handwash.pipelines.exp_demo import ExpFeatureExtractor

    settings = cfg["features"]
    yolo_path = scene_root / settings["yolo_checkpoint"]
    yolo_sha = file_sha256(yolo_path)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    extractor = None
    started = time.perf_counter()
    inputs: dict[int, list[int]] = {}
    for clip_id in _all_clip_ids(cfg):
        labels = read_labels(_label_path(scene_root, clip_id))
        target = feature_dir / f"clip_{clip_id:02d}.npz"
        if target.exists():
            with np.load(target) as cached:
                if str(cached["yolo_sha256"]) != yolo_sha or cached["images"].tolist() != labels["images"]:
                    raise RuntimeError(f"Stale feature cache {target}; delete it to re-extract")
                inputs[clip_id] = cached["network_input"].tolist()
            continue
        if extractor is None:
            extractor = ExpFeatureExtractor(yolo_path, device=device, cache_dir=output / "ultralytics")
        features, probabilities = [], []
        batch = int(settings["batch_size"])
        for start in range(0, len(labels["images"]), batch):
            frames = [cv2.imread(str(scene_root / settings["frames_dir"] / f"{clip_id}_frames" / name))
                      for name in labels["images"][start:start + batch]]
            if any(frame is None for frame in frames):
                raise RuntimeError(f"Cannot decode a frame of clip {clip_id}")
            feature, probability = extractor.forward_bgr(frames)
            features.append(feature)
            probabilities.append(probability)
        network_input = [int(size) for size in np.atleast_1d(
            getattr(extractor.model.predictor, "imgsz", extractor.image_size))]
        inputs[clip_id] = network_input
        np.savez_compressed(
            target, features=np.concatenate(features).astype(np.float32),
            yolo_probs=np.concatenate(probabilities).astype(np.float32),
            images=np.asarray(labels["images"]), yolo_sha256=np.asarray(yolo_sha),
            network_input=np.asarray(network_input),
        )
        print(f"features clip {clip_id:02d}: {len(labels['images'])} samples, network input {network_input}",
              flush=True)
    _write_json(feature_dir / "extraction.json", {
        "yolo_checkpoint": str(yolo_path), "yolo_sha256": yolo_sha,
        "requested_image_size": None if extractor is None else extractor.image_size,
        "network_input_by_clip": inputs, "frames_dir": settings["frames_dir"],
        "batch_size": settings["batch_size"], "ultralytics": ultralytics.__version__,
        "elapsed_s_this_call": time.perf_counter() - started,
    })


def load_clips(scene_root: Path, cfg: dict, output: Path) -> dict[int, dict]:
    manifest = json.loads((scene_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    kinds = {int(clip["id"]): str(clip["kind"]) for clip in manifest["clips"]}
    reviewed = {int(key): value for key, value in (cfg["splits"].get("reviewed_labels") or {}).items()}
    if not set(reviewed) <= set(cfg["splits"]["regression"]):
        raise ValueError("Reviewed labels are only allowed for regression videos")
    clips: dict[int, dict] = {}
    for clip_id in _all_clip_ids(cfg):
        base = read_labels(_label_path(scene_root, clip_id))
        clip = {"id": clip_id, "kind": kinds[clip_id], "times": base["times"], "images": base["images"],
                "labels": base["labels"], "original_labels": base["labels"], "label_source": "original"}
        if clip_id in reviewed:
            review = read_labels(scene_root / reviewed[clip_id])
            if review["images"] != base["images"] or not np.array_equal(review["labels"] < 0, base["labels"] < 0):
                raise ValueError(f"Reviewed labels of clip {clip_id} do not align with its samples")
            clip["labels"], clip["label_source"] = review["labels"], "visual_review"
        with np.load(output / "features" / f"clip_{clip_id:02d}.npz") as cached:
            if cached["images"].tolist() != base["images"]:
                raise ValueError(f"Feature cache of clip {clip_id} does not match its label file")
            clip["features"] = cached["features"].astype(np.float32)
            clip["yolo_probs"] = cached["yolo_probs"].astype(np.float32)
            clip["yolo_sha256"] = str(cached["yolo_sha256"])
            clip["network_input"] = cached["network_input"].tolist()
        clips[clip_id] = clip
    return clips


# --------------------------------------------------------------------------- evaluation
def decoder_grid(cfg: dict) -> list[dict]:
    settings = cfg["decoder"]
    base = {"method": settings["method"], "protect_min_seconds": float(settings["protect_min_seconds"]),
            "protect_confidence": float(settings["protect_confidence"]),
            "sample_fps": float(cfg["features"]["sample_fps"])}
    decoders: list[dict] = []
    for weight in settings["grid"]["raw_weight"]:
        if float(weight) == 1.0:  # control: no segment averaging at all
            decoders.append({"method": "raw", "raw_weight": 1.0})
            continue
        decoders.extend({**base, "threshold": float(threshold), "min_gap": int(gap), "raw_weight": float(weight)}
                        for threshold in settings["grid"]["threshold"] for gap in settings["grid"]["min_gap"])
    return decoders


def _selection_key(option: tuple[dict, dict]) -> tuple:
    """Higher score first; ties prefer less averaging, then fewer boundary cuts."""
    decoder, metrics = option
    simplest = ((1.0, 1.0, 10**6) if decoder["method"] == "raw"
                else (decoder["raw_weight"], decoder["threshold"], decoder["min_gap"]))
    return (round(metrics["composite"], 12), round(metrics["macro_f1"], 12),
            round(metrics["boundary_f1"], 12), *simplest)


def _inside_label(labels: np.ndarray, index: int) -> int | None:
    """GT label on both sides of a change placed before ``index``; None across a real change."""
    valid = np.flatnonzero(labels >= 0)
    left, right = valid[valid < index], valid[valid >= index]
    if not len(left) or not len(right) or labels[left[-1]] != labels[right[0]]:
        return None
    return int(labels[left[-1]])


def _error_intervals(labels: np.ndarray, pred: np.ndarray, decoded: np.ndarray, raw: np.ndarray,
                     times: np.ndarray, fps: float) -> list[dict]:
    """Index-contiguous wrong runs with one truth/prediction pair; unlabelled samples break runs."""
    runs: list[dict] = []
    for index in np.flatnonzero((labels >= 0) & (labels != pred)):
        key = (int(labels[index]), int(pred[index]))
        if runs and runs[-1]["last_index"] + 1 == index and (runs[-1]["truth"], runs[-1]["prediction"]) == key:
            runs[-1]["last_index"] = int(index)
        else:
            runs.append({"truth": key[0], "prediction": key[1], "first_index": int(index),
                         "last_index": int(index)})
    for run in runs:
        span = slice(run["first_index"], run["last_index"] + 1)
        run.update(
            start_s=float(times[run["first_index"]]), last_sample_s=float(times[run["last_index"]]),
            sampled_frames=run["last_index"] - run["first_index"] + 1,
            duration_s=(run["last_index"] - run["first_index"] + 1) / fps,
            mean_wrong_class_probability=float(decoded[span, run["prediction"]].mean()),
            mean_true_class_probability=float(decoded[span, run["truth"]].mean()),
            mean_raw_wrong_class_probability=float(raw[span, run["prediction"]].mean()),
        )
    return runs


def evaluate_clip(clip: dict, probs: np.ndarray, edges: np.ndarray | None, decoder: dict, cfg: dict, *,
                  labels: np.ndarray, detail: bool) -> dict:
    times = clip["times"]
    chain_cfg = cfg["action_chain"]
    tolerance = float(cfg["evaluation"]["boundary_tolerance_s"])
    decoded = apply_decoder(probs, edges, decoder)
    raw_pred, pred = probs.argmax(axis=1), decoded.argmax(axis=1)
    chain = sustained_action_chain(pred.tolist(), fps=float(chain_cfg["sample_fps"]),
                                   min_action_seconds=float(chain_cfg["min_action_seconds"]),
                                   smoothing_window_seconds=float(chain_cfg["smoothing_window_seconds"]),
                                   background_class=int(chain_cfg["background_class"]))
    truth_order = collapse(labels)
    reference = reference_boundaries(labels, times)
    predicted = predicted_boundaries(pred, times)
    boundary, matches = score_boundaries(reference, predicted, tolerance_s=tolerance)
    smoothed, _ = score_boundaries(reference, predicted_boundaries(chain["smoothed_labels"], times),
                                   tolerance_s=tolerance)
    valid = labels >= 0
    record = {"clip": clip["id"], "kind": clip["kind"], "y": labels[valid], "pred": pred[valid],
              "raw_pred": raw_pred[valid], "chain_match": chain["order"] == truth_order,
              "boundary": boundary, "smoothed_boundary": smoothed}
    if not detail:
        return record

    fps = float(chain_cfg["sample_fps"])
    hand_switch = set(cfg["evaluation"]["hand_switch_steps"])
    matched_reference = {match.reference for match in matches}
    matched_predicted = {match.predicted for match in matches}
    false_boundaries = []
    for number, proposal in enumerate(predicted):
        if number in matched_predicted:
            continue
        inside = _inside_label(labels, proposal.index)
        false_boundaries.append({"time_s": proposal.time_s, "from": proposal.before, "to": proposal.after,
                                 "inside_gt_label": inside,
                                 "possible_hand_switch": inside in hand_switch})
    head_alarms = []
    if edges is not None:
        threshold = float(decoder.get("threshold", 0.5))
        gap = int(decoder.get("min_gap", 3))
        for index in boundary_peaks(edges, threshold=threshold, min_gap=gap):
            moment = float((times[index - 1] + times[index]) / 2)
            if any(abs(interval_offset_s(moment, actual)) <= tolerance for actual in reference):
                continue
            inside = _inside_label(labels, index)
            head_alarms.append({"time_s": moment, "boundary_probability": float(edges[index]),
                                "inside_gt_label": inside, "possible_hand_switch": inside in hand_switch})
    errors = _error_intervals(labels, pred, decoded, probs, times, fps)
    brief = [run for run in errors
             if run["duration_s"] <= float(cfg["evaluation"]["brief_error_max_seconds"]) + 1e-9
             and run["mean_wrong_class_probability"] >= float(cfg["evaluation"]["brief_error_min_confidence"])]
    valid_index = np.flatnonzero(valid)
    steps = [index for index in valid_index if labels[index] > 0]
    edges_of_washing = {}
    if steps:
        for name, indices in (("opening_others", [i for i in valid_index if i < steps[0] and labels[i] == 0]),
                              ("closing_others", [i for i in valid_index if i > steps[-1] and labels[i] == 0])):
            edges_of_washing[name] = {
                "gt_frames": len(indices),
                "decoded_correct": int(np.count_nonzero(pred[indices] == 0)) if indices else 0,
                "raw_correct": int(np.count_nonzero(raw_pred[indices] == 0)) if indices else 0,
            }
    confirmed = [segment for segment in chain["segments"] if segment["status"] == "confirmed"]
    for segment in chain["segments"]:
        begin, end = segment["start_index"], segment["end_index_exclusive"]
        segment.update(start_s=float(times[begin]), last_sample_s=float(times[end - 1]),
                       mean_class_probability=float(decoded[begin:end, segment["class"]].mean()))
    record.update(
        label_source=clip["label_source"] if labels is clip["labels"] else "original",
        truth_order=truth_order, raw_order=collapse(raw_pred), decoded_order=collapse(pred),
        chain_order=chain["order"], non_canonical_truth=truth_order != CANONICAL_ORDER and clip["kind"] == "continuous",
        chain_segments=chain["segments"],
        washing_start_s={"gt": float(times[steps[0]]) if steps else None,
                         "chain": confirmed[0]["start_s"] if confirmed else None},
        washing_last_sample_s={"gt": float(times[steps[-1]]) if steps else None,
                               "chain": confirmed[-1]["last_sample_s"] if confirmed else None},
        accuracy=float(np.mean(labels[valid] == pred[valid])),
        raw_accuracy=float(np.mean(labels[valid] == raw_pred[valid])),
        missed_boundaries=[{"start_s": actual.start_s, "end_s": actual.end_s, "from": actual.before,
                            "to": actual.after}
                           for number, actual in enumerate(reference) if number not in matched_reference],
        false_boundaries=false_boundaries, boundary_head_alarms_inside_steps=head_alarms,
        error_intervals=errors, brief_high_confidence_errors=brief, washing_edges=edges_of_washing,
        arrays={"labels": labels, "probs": probs, "decoded": decoded, "edges": edges,
                "smoothed": np.asarray(chain["smoothed_labels"])},
    )
    return record


def summarize(records: list[dict], cfg: dict) -> dict:
    y = np.concatenate([record["y"] for record in records])
    pred = np.concatenate([record["pred"] for record in records])
    raw = np.concatenate([record["raw_pred"] for record in records])
    labels = list(range(CLASS_COUNT))
    continuous = [record for record in records if record["kind"] == "continuous"]
    isolated = [record for record in records if record["kind"] != "continuous"]
    pooled, smoothed = BoundaryScore(), BoundaryScore()
    for record in records:
        pooled.add(record["boundary"])
        smoothed.add(record["smoothed_boundary"])
    summary: dict[str, Any] = {
        "clips": len(records), "frames": len(y),
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "raw_accuracy": float(np.mean(y == raw)),
        "raw_macro_f1": float(f1_score(y, raw, labels=labels, average="macro", zero_division=0)),
        "continuous_chain_matches": f"{sum(r['chain_match'] for r in continuous)}/{len(continuous)}",
        "continuous_chain_match_rate": (float(np.mean([r["chain_match"] for r in continuous]))
                                        if continuous else None),
        "isolated_chain_matches": f"{sum(r['chain_match'] for r in isolated)}/{len(isolated)}",
        "isolated_chain_match_rate": (float(np.mean([r["chain_match"] for r in isolated]))
                                      if isolated else None),
        "boundary_f1": pooled.f1, "boundary_precision": pooled.precision, "boundary_recall": pooled.recall,
        "boundary_mean_abs_error_s": pooled.mean_abs_error_s,
        "boundary_mean_signed_error_s": pooled.mean_signed_error_s,
        "boundary_counts": pooled.as_dict(),
        "boundary_f1_after_chain_smoothing": smoothed.f1,
    }
    summary["composite"] = (composite_score(summary, cfg["evaluation"]["selection_weights"])
                            if continuous else None)
    return summary


# --------------------------------------------------------------------------- training
def _build_head(spec: dict, cfg: dict, feature_dim: int) -> nn.Module:
    settings = cfg["training"]
    common = {"feature_dim": feature_dim, "hidden": int(settings["hidden"]), "num_classes": CLASS_COUNT,
              "dropout": float(settings["dropout"])}
    if spec["model"] == "gru":
        return ExpTemporalHead(**common)
    if spec["model"] == "boundary":
        return BoundaryAwareGRU(**common, detach_boundary_input=bool(spec.get("detach_boundary_input", False)))
    raise ValueError(f"Unknown head model {spec['model']!r}")


@torch.inference_mode()
def predict(head: nn.Module, clip: dict, device: str) -> tuple[np.ndarray, np.ndarray | None]:
    head.eval()
    outputs = head(torch.from_numpy(clip["features"]).unsqueeze(0).to(device))
    probs = outputs[0][0].float().softmax(-1).cpu().numpy()
    edges = outputs[1][0].float().sigmoid().cpu().numpy() if isinstance(head, BoundaryAwareGRU) else None
    return probs, edges


def _validation_options(clips: dict, ids: list[int], predictions: dict, decoders: list[dict],
                        cfg: dict) -> list[tuple[dict, dict]]:
    options = []
    for decoder in decoders:
        records = [evaluate_clip(clips[clip_id], *predictions[clip_id], decoder, cfg,
                                 labels=clips[clip_id]["labels"], detail=False) for clip_id in ids]
        options.append((decoder, summarize(records, cfg)))
    return options


def train_run(kind: str, seed: int, clips: dict, cfg: dict, output: Path, device: str,
              config_sha: str) -> dict:
    run_dir = output / "runs" / f"{kind}_seed{seed}"
    if (run_dir / "head.pt").exists():
        selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if selection.get("config_sha256") != config_sha:
            raise RuntimeError(f"{run_dir} was trained with another config; use a new --output")
        print(f"keep existing run {run_dir.name}", flush=True)
        return selection
    spec = cfg["training"]["candidates"][kind]
    settings, boundary_cfg = cfg["training"], cfg["training"]["boundary"]
    is_boundary = spec["model"] == "boundary"
    fit, validation = [int(i) for i in cfg["splits"]["fit"]], [int(i) for i in cfg["splits"]["validation"]]
    seed_everything(seed, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    head = _build_head(spec, cfg, int(clips[fit[0]]["features"].shape[1]))
    fit_features = np.concatenate([clips[clip_id]["features"] for clip_id in fit])
    head.feature_mean.copy_(torch.from_numpy(fit_features.mean(axis=0)))
    head.feature_std.copy_(torch.from_numpy(fit_features.std(axis=0).clip(min=1e-4)))
    head.to(device)

    items, positives, negatives = [], 0, 0
    for clip_id in fit:
        labels = clips[clip_id]["labels"]
        consensus = np.where(labels >= 0, labels, -1)
        targets, mask = boundary_targets_from_consensus(consensus)
        intervals = weak_boundary_intervals_from_consensus(consensus, max_gap=int(boundary_cfg["interval_max_gap"]))
        positives += int(targets.sum())
        negatives += int(mask.sum() - targets.sum())
        items.append({"x": torch.from_numpy(clips[clip_id]["features"]).unsqueeze(0).to(device),
                      "y": torch.from_numpy(labels).unsqueeze(0).to(device),
                      "targets": torch.from_numpy(targets).to(device), "mask": torch.from_numpy(mask).to(device),
                      "intervals": [(0, left, right) for left, right in intervals]})
    positive_weight = min(float(boundary_cfg["positive_weight_cap"]),
                          max(1.0, math.sqrt(negatives / max(positives, 1))))
    positive_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    warmup = int(boundary_cfg["warmup_epochs"]) if is_boundary else 0
    decoders = decoder_grid(cfg) if is_boundary else [{"method": "raw"}]
    rng = random.Random(seed)
    order = list(range(len(items)))
    history: list[dict] = []
    best: dict | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        rng.shuffle(order)
        head.train()
        boundary_on = is_boundary and epoch > warmup
        totals: Counter[str] = Counter()
        for index in order:
            item = items[index]
            optimizer.zero_grad(set_to_none=True)
            outputs = head(item["x"])
            class_loss = nn.functional.cross_entropy(outputs[0].reshape(-1, CLASS_COUNT), item["y"].reshape(-1),
                                                     ignore_index=IGNORE)
            loss = class_loss
            if boundary_on:
                edges, mask = outputs[1], item["mask"]
                direct = (nn.functional.binary_cross_entropy_with_logits(
                    edges[0][mask], item["targets"][mask], pos_weight=positive_weight_tensor)
                    if bool(mask.any()) else edges.sum() * 0.0)
                interval = weak_interval_loss(edges, item["intervals"])
                loss = (loss + float(boundary_cfg["direct_loss_weight"]) * direct
                        + float(boundary_cfg["interval_loss_weight"]) * interval)
                totals["direct_boundary_loss"] += float(direct.detach())
                totals["interval_boundary_loss"] += float(interval.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["class_loss"] += float(class_loss.detach())
        predictions = {clip_id: predict(head, clips[clip_id], device) for clip_id in validation}
        decoder, metrics = max(_validation_options(clips, validation, predictions, decoders, cfg),
                               key=_selection_key)
        eligible = epoch > warmup
        history.append({"epoch": epoch, "eligible": eligible,
                        **{key: value / len(order) for key, value in totals.items()},
                        "decoder": decoder, "validation": metrics})
        print(json.dumps({"run": run_dir.name, "epoch": epoch, "eligible": eligible,
                          "composite": round(metrics["composite"], 4), "macro_f1": round(metrics["macro_f1"], 4),
                          "chain": metrics["continuous_chain_matches"],
                          "boundary_f1": round(metrics["boundary_f1"], 4), "decoder": decoder}), flush=True)
        if not eligible:
            continue
        if best is None or metrics["composite"] > best["validation"]["composite"] + 1e-9:
            best = {"epoch": epoch, "decoder": decoder, "validation": metrics,
                    "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
    if best is None:
        raise RuntimeError(f"{run_dir.name}: no eligible validation checkpoint")
    head.load_state_dict(best["state"])
    predictions = {clip_id: predict(head, clips[clip_id], device) for clip_id in validation}
    grid = _validation_options(clips, validation, predictions, decoders, cfg)
    replayed = next(metrics for decoder, metrics in grid if decoder == best["decoder"])
    if abs(replayed["composite"] - best["validation"]["composite"]) > 1e-6:
        raise RuntimeError(f"{run_dir.name}: restored checkpoint does not reproduce its validation score")

    yolo_sha = clips[fit[0]]["yolo_sha256"]
    fps = float(cfg["features"]["sample_fps"])
    selection = _plain({
        "run": run_dir.name, "kind": kind, "seed": seed, "candidate": spec, "config_sha256": config_sha,
        "selected_epoch": best["epoch"], "epochs_run": len(history), "decoder": best["decoder"],
        "validation": best["validation"],
        "selection_score": "0.4*macro_f1 + 0.3*continuous_chain_match_rate + 0.3*boundary_f1 (validation)",
        "validation_grid_at_selected_epoch": [{"decoder": d, **{k: m[k] for k in SUMMARY_KEYS}} for d, m in grid],
        "fit_clips": fit, "validation_clips": validation,
        "normalization": "mean/std of every sampled frame of the fit clips",
        "fit_boundary_counts": {"direct_positive": positives, "direct_negative": negatives,
                                "weak_intervals": sum(len(item["intervals"]) for item in items)},
        "boundary_positive_weight": positive_weight if is_boundary else None,
        "training": settings, "decoder_grid": cfg["decoder"], "action_chain": cfg["action_chain"],
        "yolo_sha256": yolo_sha, "network_input": clips[fit[0]]["network_input"], "sample_fps": fps,
        "device": device, "torch": torch.__version__, "elapsed_s": time.perf_counter() - started,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    head = head.cpu().eval()
    if is_boundary:
        torch.save({"format": BOUNDARY_HEAD_FORMAT, "class_names": list(EXP_CLASS_NAMES),
                    "hparams": dict(head.hparams),
                    "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                    "exp_sha256": yolo_sha, "fps": fps, "decoder": selection["decoder"],
                    "action_chain": selection["action_chain"], "metrics": selection}, run_dir / "head.pt")
    else:
        save_temporal_head(run_dir / "head.pt", head, exp_sha256=yolo_sha, fps=fps, metrics=selection)
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "selection.json", selection)
    print(f"saved {run_dir.name}: epoch {best['epoch']}, decoder {best['decoder']}, "
          f"validation composite {best['validation']['composite']:.4f}", flush=True)
    return selection


# --------------------------------------------------------------------------- reporting
def _load_head(path: Path, device: str) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES:
        raise ValueError(f"Class order of {path} does not match the seven-class protocol")
    boundary = "boundary" in str(payload.get("format"))
    head = (BoundaryAwareGRU if boundary else ExpTemporalHead)(**payload["hparams"])
    head.load_state_dict(payload["state_dict"], strict=True)
    decoder = payload.get("decoder") or payload.get("metrics", {}).get("decoder") or {"method": "raw"}
    if "method" not in decoder:  # 2026-10-19 checkpoints stored only threshold/min_gap
        decoder = {"method": "boundary_segment", "threshold": decoder["threshold"], "min_gap": decoder["min_gap"]}
    return head.to(device).eval(), {"decoder": decoder, "exp_sha256": payload.get("exp_sha256")}


def _write_predictions(path: Path, clip: dict, record: dict) -> None:
    arrays = record["arrays"]
    status = [""] * len(arrays["labels"])
    for segment in record["chain_segments"]:
        for index in range(segment["start_index"], segment["end_index_exclusive"]):
            status[index] = segment["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "time_s", "label", "label_original", "raw_label", "raw_confidence",
                         "decoded_label", "decoded_confidence", "boundary_probability",
                         "chain_smoothed_label", "chain_status"])
        for index, image in enumerate(clip["images"]):
            label, original = int(arrays["labels"][index]), int(clip["original_labels"][index])
            raw, decoded = arrays["probs"][index], arrays["decoded"][index]
            writer.writerow([
                image, f"{clip['times'][index]:.6f}", "ignore" if label < 0 else label,
                "ignore" if original < 0 else original, int(raw.argmax()), f"{raw.max():.4f}",
                int(decoded.argmax()), f"{decoded.max():.4f}",
                "" if arrays["edges"] is None else f"{arrays['edges'][index]:.4f}",
                int(arrays["smoothed"][index]), status[index],
            ])


def _mean_std(rows: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
        out[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else None
    return out


def evaluate(clips: dict, cfg: dict, output: Path, device: str) -> dict:
    splits = cfg["splits"]
    sets = {"validation": [int(i) for i in splits["validation"]], "regression": [int(i) for i in splits["regression"]]}
    report_dir = output / "report"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    yolo_sha = clips[sets["validation"][0]]["yolo_sha256"]
    models: list[dict] = []
    for kind in cfg["training"]["candidates"]:
        for seed in cfg["training"]["seeds"]:
            head, info = _load_head(output / "runs" / f"{kind}_seed{seed}" / "head.pt", device)
            models.append({"name": kind, "seed": int(seed), "head": head, "decoder": info["decoder"],
                           "sets": ("validation", "regression", "regression_original_labels")})
    models.append({"name": "yolo_frame", "seed": None, "head": None, "decoder": {"method": "raw"},
                   "sets": ("validation", "regression", "regression_original_labels")})
    scene_root = Path(cfg["_scene_root"])
    for name, relative in (cfg.get("legacy_checkpoints") or {}).items():
        head, info = _load_head(scene_root / relative, device)
        if info["exp_sha256"] != yolo_sha:
            raise RuntimeError(f"Legacy {name} was trained on another YOLO checkpoint")
        # Refit on fit + validation clips on 2026-10-19: only the regression videos are unseen.
        models.append({"name": f"legacy_{name}", "seed": None, "head": head, "decoder": info["decoder"],
                       "sets": ("regression", "regression_original_labels")})

    results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    per_video: list[dict] = []
    for model in models:
        tag = model["name"] if model["seed"] is None else f"{model['name']}_seed{model['seed']}"
        for set_name in model["sets"]:
            ids = sets["validation" if set_name == "validation" else "regression"]
            records = []
            for clip_id in ids:
                clip = clips[clip_id]
                if model["head"] is None:
                    probs, edges = clip["yolo_probs"], None
                else:
                    probs, edges = predict(model["head"], clip, device)
                labels = clip["original_labels"] if set_name == "regression_original_labels" else clip["labels"]
                record = evaluate_clip(clip, probs, edges, model["decoder"], cfg, labels=labels, detail=True)
                records.append(record)
                if set_name != "regression_original_labels":
                    _write_predictions(report_dir / "predictions" / tag / f"video{clip_id:02d}.csv", clip, record)
                per_video.append({"model": model["name"], "seed": model["seed"], "set": set_name,
                                  **{key: value for key, value in record.items()
                                     if key not in {"y", "pred", "raw_pred", "arrays", "boundary", "smoothed_boundary"}},
                                  "boundary": record["boundary"].as_dict(),
                                  "boundary_after_chain_smoothing": record["smoothed_boundary"].as_dict()})
            summary = summarize(records, cfg)
            results[model["name"]][set_name].append({"seed": model["seed"], "decoder": model["decoder"], **summary})

    aggregate = {name: {set_name: _mean_std(rows) for set_name, rows in by_set.items()}
                 for name, by_set in results.items()}
    candidates = [kind for kind, spec in cfg["training"]["candidates"].items() if spec["model"] == "boundary"]
    selected = max(candidates, key=lambda kind: (aggregate[kind]["validation"]["composite"],
                                                 aggregate[kind]["validation"]["macro_f1"]))
    decisions = {kind: {set_name: replacement_decision(aggregate[kind][set_name], aggregate["gru"][set_name])
                        for set_name in ("validation", "regression")} for kind in candidates}
    recommend = all(decision["recommend_replacement"] for decision in decisions[selected].values())
    summary = _plain({
        "format": "self-scene-boundary-v2-report/1", "config": {k: v for k, v in cfg.items() if k != "_scene_root"},
        "scene_root": str(scene_root), "yolo_sha256": yolo_sha,
        "network_input": clips[sets["validation"][0]]["network_input"],
        "per_seed": results, "seed_mean": aggregate,
        "selected_boundary_candidate": selected,
        "selection_rule": "highest mean validation composite over seeds (0.4 macro-F1 + 0.3 chain + 0.3 boundary F1)",
        "replacement_decisions": decisions,
        "recommend_replacing_gru": recommend,
        "notes": [
            "Validation clips selected epochs and decoders, so validation numbers are optimistic.",
            "Regression videos 4 and 8 were inspected repeatedly in earlier rounds; they are not an independent test.",
            "Video 8 is scored with visually reviewed labels; regression_original_labels keeps the original labels.",
            "legacy_* heads were refit on fit + validation clips and are compared on regression videos only.",
        ],
    })
    _write_json(report_dir / "summary.json", summary)
    _write_json(report_dir / "per_video.json", _plain(per_video))
    (report_dir / "report.md").write_text(render_markdown(summary, _plain(per_video), cfg), encoding="utf-8")
    print(json.dumps({"selected": selected, "recommend_replacing_gru": recommend,
                      "decisions": {k: {s: d["recommend_replacement"] for s, d in v.items()}
                                    for k, v in decisions.items()}}, ensure_ascii=False), flush=True)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pm(row: dict, key: str, digits: int = 4) -> str:
    mean, std = row.get(key), row.get(f"{key}_std")
    if mean is None:
        return "—"
    return f"{mean:.{digits}f}" + ("" if std is None else f" ± {std:.{digits}f}")


MODEL_LABELS = {
    "gru": "普通 GRU（重训）", "joint": "候选 A：联合训练", "detached": "候选 B：边界不回传",
    "yolo_frame": "YOLO 逐帧（参考）", "legacy_yolo_gru": "旧 GRU（10-19，含验证集重训）",
    "legacy_boundary_gru": "旧边界 GRU（10-19，含验证集重训）",
}
SET_LABELS = {"validation": "验证集（8 段，选模用）", "regression": "回归视频 4、8（视频 8 用复核标签）",
              "regression_original_labels": "回归视频 4、8（视频 8 原标签，敏感性）"}


def _decoder_text(decoder: dict) -> str:
    if decoder.get("method") == "raw":
        return "原始概率" + ("（融合权重 1.0）" if decoder.get("raw_weight") == 1.0 else "")
    if decoder.get("method") == "boundary_segment":
        return f"旧段平均 阈值{decoder['threshold']} 间隔{decoder['min_gap']}"
    return f"阈值{decoder['threshold']} 间隔{decoder['min_gap']} 原始权重{decoder['raw_weight']}"


def render_markdown(summary: dict, per_video: list[dict], cfg: dict) -> str:
    means, per_seed = summary["seed_mean"], summary["per_seed"]
    lines = ["# 自采场景边界感知模型优化报告（v2）", ""]
    lines += [f"- YOLO 特征提取器 sha256 `{summary['yolo_sha256'][:16]}…`，实际网络输入 {summary['network_input']}。",
              f"- 种子：{cfg['training']['seeds']}；只用 14 段训练，验证集 8 段选检查点与解码参数，回归视频不参与选择。",
              "- 选模分数 = 0.4×宏平均 F1 + 0.3×连续视频动作链匹配率 + 0.3×边界 F1（容差：标注过渡区间外加 "
              f"{cfg['evaluation']['boundary_tolerance_s']} 秒，一对一匹配）。", ""]
    columns = ["accuracy", "macro_f1", "continuous_chain_match_rate", "isolated_chain_match_rate", "boundary_f1",
               "boundary_mean_abs_error_s", "boundary_f1_after_chain_smoothing", "composite"]
    header = ("| 模型 | 准确率 | 宏平均 F1 | 连续视频动作链 | 单动作视频动作链 | 边界 F1 | 边界时间误差(秒) | "
              "1 秒平滑后边界 F1 | 综合分 |")
    for set_name in ("validation", "regression", "regression_original_labels"):
        lines += [f"## {SET_LABELS[set_name]}（三种子均值 ± 标准差）", "", header, "|" + " --- |" * 9]
        for name, by_set in means.items():
            if set_name in by_set:
                row = by_set[set_name]
                lines.append(f"| {MODEL_LABELS.get(name, name)} | "
                             + " | ".join(_pm(row, key, 3 if "error" in key else 4) for key in columns) + " |")
        lines.append("")
    lines += ["## 每个种子的选择结果", "", "| 模型 | 种子 | 解码设置 | 验证综合分 | 验证宏 F1 | 验证边界 F1 | "
              "回归准确率 | 回归宏 F1 | 回归动作链 | 回归边界 F1 |", "|" + " --- |" * 10]
    for name in cfg["training"]["candidates"]:
        regression = {row["seed"]: row for row in per_seed[name]["regression"]}
        for row in per_seed[name]["validation"]:
            other = regression[row["seed"]]
            lines.append(f"| {MODEL_LABELS[name]} | {row['seed']} | {_decoder_text(row['decoder'])} | "
                         f"{_fmt(row['composite'])} | {_fmt(row['macro_f1'])} | {_fmt(row['boundary_f1'])} | "
                         f"{_fmt(other['accuracy'])} | {_fmt(other['macro_f1'])} | {other['continuous_chain_matches']} | "
                         f"{_fmt(other['boundary_f1'])} |")
    lines.append("")
    selected = summary["selected_boundary_candidate"]
    lines += ["## 替换判定", "", f"按验证集三种子平均综合分，入选的边界候选是 **{MODEL_LABELS[selected]}**。", ""]
    for kind, decisions in summary["replacement_decisions"].items():
        for set_name, decision in decisions.items():
            verdict = "通过" if decision["recommend_replacement"] else "未通过"
            details = "；".join(
                (f"{item['metric']} {_fmt(item['candidate'])} vs GRU {_fmt(item['baseline'])}"
                 if item["rule"] != "at_least_one_improves" else "没有任何一项高于 GRU")
                for item in decision["failures"]) or "无"
            improved = "、".join(decision["improved"]) or "无"
            lines.append(f"- {MODEL_LABELS[kind]} · {SET_LABELS[set_name]}：**{verdict}**；提高项：{improved}；未达标：{details}")
    lines += ["", "**结论：" + ("推荐用入选的边界候选替换普通 GRU。" if summary["recommend_replacing_gru"]
                              else "不推荐替换，继续以普通 GRU 为主方案。") + "**", ""]
    lines += _error_section(per_video, cfg)
    lines += _chain_section(per_video)
    lines += ["## 说明", ""] + [f"- {note}" for note in summary["notes"]] + [""]
    return "\n".join(lines)


def _error_section(per_video: list[dict], cfg: dict) -> list[str]:
    lines = ["## 错误检查（三种子合计）", "",
             "| 模型 | 数据 | 漏检边界 | 误检边界 | 其中步骤内部 | 其中可能是换手（步骤 2/4/5/6） | 边界头步骤内报警 | "
             "短暂高置信度误判 | 开头 others 正确帧 | 结尾 others 正确帧 | 乱序视频动作链匹配 |", "|" + " --- |" * 11]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_video:
        groups[(row["model"], row["set"])].append(row)
    for (model, set_name), rows in groups.items():
        if set_name == "regression_original_labels":
            continue
        false = [item for row in rows for item in row["false_boundaries"]]
        inside = [item for item in false if item["inside_gt_label"] is not None]
        opening = [row["washing_edges"].get("opening_others", {}) for row in rows]
        closing = [row["washing_edges"].get("closing_others", {}) for row in rows]
        shuffled = [row for row in rows if row["non_canonical_truth"]]
        lines.append(
            f"| {MODEL_LABELS.get(model, model)} | {'验证' if set_name == 'validation' else '回归'} | "
            f"{sum(len(row['missed_boundaries']) for row in rows)} | {len(false)} | {len(inside)} | "
            f"{sum(item['possible_hand_switch'] for item in inside)} | "
            f"{sum(len(row['boundary_head_alarms_inside_steps']) for row in rows)} | "
            f"{sum(len(row['brief_high_confidence_errors']) for row in rows)} | "
            f"{sum(item.get('decoded_correct', 0) for item in opening)}/{sum(item.get('gt_frames', 0) for item in opening)} | "
            f"{sum(item.get('decoded_correct', 0) for item in closing)}/{sum(item.get('gt_frames', 0) for item in closing)} | "
            f"{sum(row['chain_match'] for row in shuffled)}/{len(shuffled)} |")
    lines += ["", "重复步骤：现有标注中没有任何视频出现重复步骤，只能由单元测试覆盖（`[4, 3, 4]` 等）。", ""]
    lines += ["## 回归视频的错误时间段（解码后，含置信度）", ""]
    for row in per_video:
        if row["set"] != "regression" or not row["error_intervals"]:
            continue
        name = MODEL_LABELS.get(row["model"], row["model"]) + ("" if row["seed"] is None else f" 种子{row['seed']}")
        spans = "；".join(
            f"{run['start_s']:.1f}–{run['last_sample_s']:.1f}s 真{run['truth']}→判{run['prediction']} "
            f"({run['sampled_frames']}帧, 错类 {run['mean_wrong_class_probability']:.2f})"
            for run in row["error_intervals"])
        lines.append(f"- 视频 {row['clip']} · {name}：{spans}")
    lines.append("")
    return lines


def _chain_section(per_video: list[dict]) -> list[str]:
    lines = ["## 逐视频确认动作链", "", "| 数据 | 视频 | 标注顺序 | 模型 | 确认动作链（各种子） | 匹配 |", "|" + " --- |" * 6]
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in per_video:
        if row["set"] != "regression_original_labels":
            groups[(row["set"], row["clip"], row["model"])].append(row)
    for (set_name, clip_id, model), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        chains = sorted({tuple(row["chain_order"]) for row in rows})
        text = " / ".join("".join(map(str, chain)) or "（空）" for chain in chains)
        lines.append(f"| {'验证' if set_name == 'validation' else '回归'} | {clip_id} | "
                     f"{''.join(map(str, rows[0]['truth_order']))} | {MODEL_LABELS.get(model, model)} | {text} | "
                     f"{sum(row['chain_match'] for row in rows)}/{len(rows)} |")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- entry point
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-root", type=Path, required=True,
                        help="Self-scene working directory (frames/, prepared/, models/, dataset_manifest.json)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research/self_scene_boundary_v2.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("features", "train", "evaluate", "all"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # Must be set before the first cuBLAS call for deterministic GRU training on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_text = args.config.read_text(encoding="utf-8")
    config_sha = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    cfg = yaml.safe_load(config_text)
    if cfg.get("format") != "self-scene-boundary-v2/1":
        parser.error(f"Unexpected config format in {args.config}")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "config.yaml"
    if frozen.exists() and frozen.read_text(encoding="utf-8") != config_text:
        parser.error(f"{args.output} was created with a different config; choose a new --output")
    frozen.write_text(config_text, encoding="utf-8")
    cfg["_scene_root"] = str(args.scene_root)

    if args.stage in ("features", "all"):
        extract_features(args.scene_root, cfg, args.output, args.device)
    if args.stage == "features":
        return 0
    clips = load_clips(args.scene_root, cfg, args.output)
    if args.stage in ("train", "all"):
        for seed in cfg["training"]["seeds"]:
            for kind in cfg["training"]["candidates"]:
                train_run(kind, int(seed), clips, cfg, args.output, args.device, config_sha)
    if args.stage in ("evaluate", "all"):
        evaluate(clips, cfg, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
