#!/usr/bin/env python
"""Train YOLO26l-cls on the locked PSKUS DataSet1-4 training split.

This model-size ablation selects a checkpoint using validation images only.
The existing YOLO26m baseline and the locked test split are untouched.
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
    parser.add_argument("--model", default="yolo26l-cls.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/pskuss_ds1234_yolo26l_candidate"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch, args.imgsz) < 1:
        parser.error("epochs, patience, batch and imgsz must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    for required in (args.data / "summary.json", args.data / "video_splits.csv"):
        if not required.is_file():
            parser.error(f"Missing locked data input: {required}")
    dataset = json.loads((args.data / "summary.json").read_text(encoding="utf-8"))
    if tuple(dataset["classes"]) != EXP_CLASS_NAMES or dataset["video_split"] != {
        "train": 430,
        "val": 92,
        "test": 93,
    }:
        parser.error("Data does not match the locked seven-class DataSet1-4 protocol")

    import torch
    from ultralytics import YOLO, __version__

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for this experiment")
    yolo = YOLO(args.model)
    if yolo.task != "classify" or yolo.model.yaml.get("scale") != "l":
        parser.error("Expected a pretrained YOLO26l classification model")
    source = Path(yolo.ckpt_path)
    metadata = {
        "source_model": args.model,
        "source_sha256": file_sha256(source) if source.is_file() else None,
        "ultralytics_version": __version__,
        "data": str(args.data),
        "split_reference_sha256": file_sha256(args.data / "video_splits.csv"),
        "video_split": dataset["video_split"],
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": "auto",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "seed": 42,
        "test_used_for_selection": False,
        "comparison_note": "YOLO26m baseline was continued from a prior DataSet1/3/4 checkpoint; this YOLO26l experiment starts from the official pretrained classifier, so architecture and initialization both differ.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yolo.train(
        data=str(args.data / "ultralytics"),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=6,
        amp=True,
        cache=False,
        optimizer="auto",
        auto_augment="randaugment",
        erasing=0.4,
        project=str(args.output.resolve().parent),
        name=args.output.name,
        exist_ok=True,
        seed=42,
        plots=False,
    )
    best = args.output / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training ended without best checkpoint: {best}")
    trained = YOLO(str(best))
    names = tuple(str(trained.names.get(index, "")) for index in range(7))
    if trained.model.yaml.get("scale") != "l" or names != EXP_CLASS_NAMES:
        raise RuntimeError(f"Trained model structure or seven-class order changed: {names}")
    metadata["best_checkpoint"] = str(best)
    metadata["best_sha256"] = file_sha256(best)
    (args.output / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "best": str(best), "sha256": metadata["best_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
