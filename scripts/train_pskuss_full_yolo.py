#!/usr/bin/env python
"""Fine-tune exp.pt on all PSKUS partitions using Ultralytics classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from build_pskuss_full_yolo_dataset import CLASSES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "exp.pt")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "data/processed/pskuss_full_yolo")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/pskuss_full_yolo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        parser.error("全量 YOLO 训练需要 CUDA")
    if not args.model.is_file():
        parser.error(f"找不到起始权重：{args.model}")
    summary_path = args.data / "summary.json"
    if not summary_path.is_file() or (args.data / ".building").exists():
        parser.error("七类数据集尚未完整构建并校验")
    data_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tuple(data_summary["classes"]) != CLASSES:
        parser.error("训练数据类别顺序不符")
    run_dir = args.output.resolve()
    if (run_dir / "weights/best.pt").exists():
        parser.error(f"已有最佳权重，为防止覆盖请换 --output：{run_dir}")

    model = YOLO(str(args.model))
    model_names = tuple(model.names[i] for i in range(7))
    if model.task != "classify" or model_names != CLASSES:
        parser.error(f"exp.pt 类别或任务不符：{model.task}, {model_names}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"source_checkpoint": str(args.model), "source_sha256": _sha256(args.model),
                "data": str(args.data), "data_summary": data_summary,
                "epochs_max": args.epochs, "patience": args.patience,
                "batch": args.batch, "imgsz": args.imgsz}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "handwash_training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model.train(data=str(args.data / "ultralytics"), imgsz=args.imgsz,
                epochs=args.epochs, patience=args.patience, batch=args.batch,
                device=0, workers=6, amp=True, cache=False,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                seed=42, plots=True)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"训练结束但缺少最佳权重：{best}")
    test_model = YOLO(str(best))
    metrics = test_model.val(data=str(args.data / "ultralytics"), split="test",
                             imgsz=args.imgsz, batch=args.batch, device=0, workers=6)
    payload = {"best_weights": str(best), "best_sha256": _sha256(best),
               "top1_accuracy": float(metrics.top1), "top5_accuracy": float(metrics.top5),
               "split": "test", "dataset_videos": data_summary["video_split"],
               "note": "Test images are stable dual-annotator frames; full-video transitions are separately evaluated."}
    (run_dir / "test_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"YOLO 全量训练完成：{best}; test top1={payload['top1_accuracy']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
