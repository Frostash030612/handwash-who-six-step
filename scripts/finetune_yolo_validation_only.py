#!/usr/bin/env python
"""Continue the existing YOLO26m classifier on the same locked PSKUS split.

This controlled candidate uses lower learning rate and less random erasing of
small hand details. Ultralytics selects its checkpoint on the existing val
images. The locked test images and videos are never evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import PROJECT_ROOT
except ImportError:  # pragma: no cover
    from ._bootstrap import PROJECT_ROOT
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo/weights/best.pt"
    )
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_low_lr_candidate"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--erasing", type=float, default=0.15)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error("epochs, patience, batch and imgsz must be positive")
    if not 0 < args.lr < 0.01 or not 0 <= args.erasing <= 1:
        parser.error("lr must be in (0, .01), erasing in [0, 1]")
    for required in (args.model, args.data / "summary.json", args.data / "video_splits.csv"):
        if not required.is_file():
            parser.error(f"Missing required input: {required}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite a non-empty output: {args.output}")
    summary = json.loads((args.data / "summary.json").read_text(encoding="utf-8"))
    if tuple(summary["classes"]) != EXP_CLASS_NAMES or summary["video_split"] != {
        "train": 430,
        "val": 92,
        "test": 93,
    }:
        parser.error("Data does not match the locked DataSet1-4 seven-class protocol")

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        parser.error("CUDA is required for this experiment")
    yolo = YOLO(str(args.model))
    if yolo.task != "classify" or yolo.model.yaml.get("scale") != "m":
        parser.error("Expected the current YOLO26m classification checkpoint")
    names = tuple(str(yolo.names.get(index, "")) for index in range(len(yolo.names)))
    if names != EXP_CLASS_NAMES:
        parser.error(f"Class order changed: {names}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_checkpoint": str(args.model),
        "source_sha256": file_sha256(args.model),
        "data": str(args.data),
        "split_reference_sha256": file_sha256(args.data / "video_splits.csv"),
        "video_split": summary["video_split"],
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": "AdamW",
        "lr0": args.lr,
        "lrf": 0.1,
        "weight_decay": 5e-4,
        "warmup_epochs": 0.0,
        "warmup_bias_lr": 0.0,
        "auto_augment": "randaugment",
        "erasing": args.erasing,
        "test_used_for_selection": False,
    }
    (args.output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yolo.train(
        data=str(args.data / "ultralytics"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=0,
        workers=6,
        amp=True,
        cache=False,
        optimizer="AdamW",
        lr0=args.lr,
        lrf=0.1,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        weight_decay=5e-4,
        auto_augment="randaugment",
        erasing=args.erasing,
        project=str(args.output.resolve().parent),
        name=args.output.name,
        exist_ok=True,
        seed=42,
        plots=False,
    )
    best = args.output / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training ended without a best checkpoint: {best}")
    metadata["best_checkpoint"] = str(best)
    metadata["best_sha256"] = file_sha256(best)
    (args.output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "best": str(best), "sha256": metadata["best_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
