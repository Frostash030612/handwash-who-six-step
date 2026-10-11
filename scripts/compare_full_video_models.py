#!/usr/bin/env python
"""Evaluate YOLO, VideoMAE, and an optional YOLO+GRU on identical PSKUS videos.

The original training split CSV is mandatory. Each method predicts the same
16-frame window centres; the same smoothing search is run on validation and
the selected setting is frozen before test scoring. Complete video timelines
are decoded, including action transitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from scripts.evaluate_videomae_full_videos import (
    _decode_sampled_frames, _ground_truth, _predict_windows, _window_starts,
)
from scripts.evaluate_yolo_full_videos import decode as decode_yolo, predict as predict_yolo
from scripts.full_video_eval_common import (
    RAW_SETTING, _predictions, choose_smoothing, public_metrics, score, videos_from_reference,
)
from scripts.train_videomae_pskus import LABELS, VideoInfo


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centres(frame_ids: list[int], *, num_frames: int, window_stride: int) -> tuple[list[int], np.ndarray]:
    starts = _window_starts(len(frame_ids), num_frames, window_stride)
    if not starts:
        return [], np.asarray([], dtype=np.int64)
    positions = np.asarray([start + num_frames // 2 for start in starts], dtype=np.int64)
    return starts, positions


def _item(video: VideoInfo, probs: np.ndarray, frame_ids: list[int], positions: np.ndarray,
          fps: float) -> dict:
    centre_frame_ids = np.asarray([frame_ids[int(index)] for index in positions], dtype=np.int64)
    targets, usable = _ground_truth(video, centre_frame_ids.tolist())
    if probs.shape != (len(positions), len(LABELS)):
        raise ValueError(f"Unexpected probabilities for {video.source}/{video.stem}: {probs.shape}")
    return {
        "video_id": f"{video.source}/{video.stem}", "split": video.split,
        "probs": probs.astype(np.float32),
        "centre_frame_ids": centre_frame_ids,
        "targets": np.asarray(targets, dtype=np.int64),
        "usable": np.asarray(usable, dtype=bool), "fps": fps,
    }


def _save_raw(cache: Path, item: dict) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    name = item["video_id"].replace("/", "__") + ".npz"
    np.savez_compressed(cache / name, **{
        key: item[key] for key in ("probs", "centre_frame_ids", "targets", "usable", "fps")
    })


def _evaluate(name: str, items: dict[str, list[dict]], output: Path, *, model: Path,
              reference: Path, elapsed_s: float) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    best, grid = choose_smoothing(items["val"])
    raw = score(items["test"], RAW_SETTING)
    selected = score(items["test"], best, per_video=True)
    split_ids = {split: [item["video_id"] for item in group] for split, group in items.items()}
    summary = {
        "model_kind": name, "model": str(model),
        "model_sha256": _sha256(model if model.is_file() else model / "model.safetensors"),
        "split_reference": str(reference), "split_reference_sha256": _sha256(reference),
        "video_counts": {split: len(group) for split, group in items.items()},
        "video_ids": split_ids,
        "sample_fps": 5.0, "num_frames": 16, "window_stride_sampled_frames": 4,
        "smoothing_unit": "aligned window centres (about 0.8 seconds each)",
        "validation_grid": grid, "chosen_on_validation": best,
        "test_raw": public_metrics(raw), "test_smoothed": public_metrics(selected),
        "classification_report": classification_report(
            selected["truth"], selected["predictions"], labels=list(range(7)),
            target_names=LABELS, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(selected["truth"], selected["predictions"],
                                               labels=list(range(7))).tolist(),
        "elapsed_s": elapsed_s,
        "evaluation_note": "The model processes each entire video. Scores use identical two-annotator consensus centres, including action transitions. Sequence metrics use the collapsed centre-label sequences, not manual whole-video compliance judgments.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected["per_video"][0]))
        writer.writeheader()
        writer.writerows(selected["per_video"])
    print(f"{name}: test raw F1={raw['macro_f1']:.4f}, selected F1={selected['macro_f1']:.4f}, "
          f"sequence exact={selected['sequence_exact_videos']}/{selected['videos']}", flush=True)
    return summary


def _run_yolo(videos: list[VideoInfo], model_path: Path, output: Path, *, batch: int,
              reference: Path) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    expected = ("0_other_or_nonstandard", "1_palm_to_palm", "2_palm_over_dorsum",
                "3_palm_to_palm_interlaced", "4_backs_of_fingers", "5_thumb_rubbing",
                "6_fingertips_to_palm")
    if model.task != "classify" or tuple(model.names[index] for index in range(7)) != expected:
        raise ValueError("YOLO checkpoint task or class order differs from the seven-class protocol")
    imgsz = int(model.overrides.get("imgsz", 320))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    items: dict[str, list[dict]] = {"val": [], "test": []}
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        frames, frame_ids, fps = decode_yolo(video, 5.0)
        _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
        if not len(positions):
            raise ValueError(f"Too short for common centres: {video.source}/{video.stem}")
        frame_probs = predict_yolo(model, frames, imgsz, device, batch)
        item = _item(video, frame_probs[positions], frame_ids, positions, fps)
        items[video.split].append(item)
        _save_raw(output / "raw_probabilities", item)
        print(f"YOLO {number}/{len(videos)} {video.split} {item['video_id']} "
              f"centres={len(positions)} agreed={int(item['usable'].sum())}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _evaluate("yolo", items, output, model=model_path, reference=reference,
                     elapsed_s=time.perf_counter() - started)


def _run_videomae(videos: list[VideoInfo], model_path: Path, output: Path, *, name: str,
                  batch: int, reference: Path) -> dict:
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = VideoMAEImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    items: dict[str, list[dict]] = {"val": [], "test": []}
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        frames, frame_ids, fps = _decode_sampled_frames(video, sample_fps=5.0)
        starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
        if not starts:
            raise ValueError(f"Too short for common centres: {video.source}/{video.stem}")
        probs = _predict_windows(model, frames, starts, num_frames=16, batch_size=batch,
                                 mean=mean, std=std, device=device)
        item = _item(video, probs, frame_ids, positions, fps)
        items[video.split].append(item)
        _save_raw(output / "raw_probabilities", item)
        print(f"{name} {number}/{len(videos)} {video.split} {item['video_id']} "
              f"centres={len(positions)} agreed={int(item['usable'].sum())}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _evaluate(name, items, output, model=model_path, reference=reference,
                     elapsed_s=time.perf_counter() - started)


def _run_gru(videos: list[VideoInfo], head_path: Path, yolo_path: Path,
             feature_cache: Path, output: Path, *, reference: Path) -> dict:
    import torch
    from handwash.pipelines.exp_demo import load_temporal_head

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    head = load_temporal_head(head_path, exp_checkpoint=yolo_path, fps=5.0).to(device)
    checkpoint_sha = _sha256(yolo_path)
    items: dict[str, list[dict]] = {"val": [], "test": []}
    started = time.perf_counter()
    for number, video in enumerate(videos, start=1):
        path = feature_cache / f"{video.source}__{video.stem}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing YOLO feature cache for GRU: {path}")
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["model_sha256"]) != checkpoint_sha or float(cached["sample_fps"]) != 5.0:
                raise ValueError(f"GRU feature cache does not match YOLO checkpoint: {path}")
            features = cached["features"].astype(np.float32)
            frame_ids = cached["frame_ids"].astype(np.int64).tolist()
            fps = float(cached["source_fps"])
        _starts, positions = _centres(frame_ids, num_frames=16, window_stride=4)
        if not len(positions):
            raise ValueError(f"Too short for common centres: {video.source}/{video.stem}")
        with torch.inference_mode():
            values = torch.from_numpy(features).unsqueeze(0).to(device)
            probs = head(values)[0][0].softmax(dim=-1).cpu().numpy()
        item = _item(video, probs[positions], frame_ids, positions, fps)
        items[video.split].append(item)
        _save_raw(output / "raw_probabilities", item)
        print(f"YOLO+GRU {number}/{len(videos)} {video.split} {item['video_id']} "
              f"centres={len(positions)} agreed={int(item['usable'].sum())}", flush=True)
    return _evaluate("yolo_gru", items, output, model=head_path, reference=reference,
                     elapsed_s=time.perf_counter() - started)


def _compare_existing(output: Path, reference: Path) -> None:
    names = ("yolo", "videomae", "videomae_center", "yolo_gru")
    summaries = {}
    for name in names:
        path = output / name / "summary.json"
        if path.is_file():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    if len(summaries) < 2:
        return
    baseline = next(iter(summaries))
    for name, summary in summaries.items():
        if summary["split_reference_sha256"] != _sha256(reference):
            raise RuntimeError(f"Split reference changed for {name}")
        for split in ("val", "test"):
            if summary["video_ids"][split] != summaries[baseline]["video_ids"][split]:
                raise RuntimeError(f"Model video lists differ on {split}: {baseline}, {name}")
            for video_id in summary["video_ids"][split]:
                filename = video_id.replace("/", "__") + ".npz"
                with np.load(output / baseline / "raw_probabilities" / filename) as first, \
                     np.load(output / name / "raw_probabilities" / filename) as second:
                    for key in ("centre_frame_ids", "targets", "usable"):
                        if not np.array_equal(first[key], second[key]):
                            raise RuntimeError(f"Different {key} for {video_id}: {baseline}, {name}")
    from sklearn.metrics import f1_score

    def paired_delta(first_name: str, second_name: str) -> dict:
        # Resample *videos*, never correlated overlapping windows as if they
        # were independent observations. This is descriptive, not a subject-
        # or hospital-level generalisation interval.
        arrays = []
        for video_id in summaries[baseline]["video_ids"]["test"]:
            filename = video_id.replace("/", "__") + ".npz"
            predictions = {}
            truth = None
            for name in (first_name, second_name):
                with np.load(output / name / "raw_probabilities" / filename) as saved:
                    usable = saved["usable"]
                    if truth is None:
                        truth = saved["targets"][usable]
                    predictions[name] = _predictions(
                        saved["probs"], summaries[name]["chosen_on_validation"]
                    )[usable]
            arrays.append((truth, predictions[first_name], predictions[second_name]))
        rng = np.random.default_rng(42)
        deltas = np.empty(1000, dtype=np.float64)
        for sample in range(len(deltas)):
            choices = rng.integers(0, len(arrays), size=len(arrays))
            joined = [np.concatenate([arrays[int(index)][part] for index in choices]) for part in range(3)]
            deltas[sample] = f1_score(joined[0], joined[1], labels=list(range(7)),
                                      average="macro", zero_division=0) - f1_score(
                                          joined[0], joined[2], labels=list(range(7)),
                                          average="macro", zero_division=0)
        return {
            "first": first_name, "second": second_name,
            "observed_delta_macro_f1": (summaries[first_name]["test_smoothed"]["macro_f1"]
                                        - summaries[second_name]["test_smoothed"]["macro_f1"]),
            "video_bootstrap_95pct_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "bootstrap_replicates": len(deltas),
        }

    pairs = [("yolo", "videomae"), ("yolo_gru", "yolo"),
             ("videomae_center", "videomae"), ("videomae_center", "yolo_gru")]
    videomae_candidates = [name for name in ("videomae", "videomae_center") if name in summaries]
    selected_videomae = (max(videomae_candidates,
                            key=lambda name: summaries[name]["chosen_on_validation"]["macro_f1"])
                         if videomae_candidates else None)
    comparison = {
        "split_reference_sha256": _sha256(reference),
        "common_val_videos": len(summaries[baseline]["video_ids"]["val"]),
        "common_test_videos": len(summaries[baseline]["video_ids"]["test"]),
        "common_test_centres": summaries[baseline]["test_raw"]["evaluated_centres"],
        "metrics": {name: {"raw": summary["test_raw"], "selected": summary["test_smoothed"]}
                    for name, summary in summaries.items()},
        "paired_test_macro_f1_deltas": [paired_delta(left, right) for left, right in pairs
                                        if left in summaries and right in summaries],
        "selected_videomae_by_validation": selected_videomae,
        "videomae_validation_macro_f1": {
            name: summaries[name]["chosen_on_validation"]["macro_f1"]
            for name in videomae_candidates
        },
        "note": "Same videos, centre times, consensus labels, smoothing candidate grid and validation-only selection. YOLO sees one frame, YOLO+GRU sees sampled-frame YOLO features, and both VideoMAE variants see 16-frame windows. The centre-stable variant is a training-policy change, not an architectural ablation.",
    }
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"Paired comparison completed: {', '.join(summaries)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/pskuss/extracted")
    parser.add_argument("--split-reference", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt")
    parser.add_argument("--videomae-model", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_videomae/best_model")
    parser.add_argument("--videomae-center-model", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_videomae_center_stable/best_model")
    parser.add_argument("--gru-head", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/head.pt")
    parser.add_argument("--gru-feature-cache", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_gru/features")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_fair_video_comparison")
    parser.add_argument("--models", default="yolo,videomae", help="Comma-separated subset of yolo,videomae,videomae_center,yolo_gru")
    parser.add_argument("--yolo-batch", type=int, default=32)
    parser.add_argument("--videomae-batch", type=int, default=4)
    parser.add_argument("--compare-only", action="store_true",
                        help="Rebuild paired comparison from existing per-model summaries and cached probabilities")
    args = parser.parse_args()
    if args.compare_only:
        _compare_existing(args.output, args.split_reference)
        return 0
    chosen = [name.strip() for name in args.models.split(",") if name.strip()]
    if not chosen or set(chosen) - {"yolo", "videomae", "videomae_center", "yolo_gru"} or len(set(chosen)) != len(chosen):
        parser.error("--models must be yolo, videomae, videomae_center, or yolo_gru without duplicates")
    videos = [video for video in videos_from_reference(args.root, args.split_reference)
              if video.split in {"val", "test"}]
    counts = {split: sum(video.split == split for video in videos) for split in ("val", "test")}
    if min(counts.values()) == 0:
        parser.error(f"Validation or test videos missing from training split: {counts}")
    print(f"Strict original-video split: {counts}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in chosen:
        output = args.output / name
        if (output / "summary.json").exists():
            raise ValueError(f"Refusing to overwrite existing evaluation: {output}")
        if name == "yolo":
            summaries[name] = _run_yolo(videos, args.yolo_model, output,
                                        batch=args.yolo_batch, reference=args.split_reference)
        elif name in {"videomae", "videomae_center"}:
            model_path = args.videomae_model if name == "videomae" else args.videomae_center_model
            summaries[name] = _run_videomae(videos, model_path, output, name=name,
                                            batch=args.videomae_batch, reference=args.split_reference)
        else:
            summaries[name] = _run_gru(videos, args.gru_head, args.yolo_model,
                                       args.gru_feature_cache, output, reference=args.split_reference)
    _compare_existing(args.output, args.split_reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
