#!/usr/bin/env python
"""Queue DataSet1/2/3/4 VideoMAE fine-tuning after the YOLO run."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT

WORK = PROJECT_ROOT / "outputs/pskuss_ds1234_videomae_queue"
YOLO_STATUS = PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_queue/status.json"
DATA = PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo"
CHECKPOINT = PROJECT_ROOT / "outputs/pskuss_ds134_videomae_stable/best_model"
OUTPUT = PROJECT_ROOT / "outputs/pskuss_ds1234_videomae"


def status(stage: str, detail: str = "") -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "detail": detail,
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    (WORK / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{payload['updated_utc']} {stage}: {detail}", flush=True)


def screen_running(name: str) -> bool:
    result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
    return any(name in line for line in result.stdout.splitlines())


def yolo_running() -> bool:
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True)
    return any("scripts/train_pskuss_full_yolo.py" in line and
               "python" in line.split()[0]
               for line in result.stdout.splitlines() if line.split())


def main() -> int:
    if (OUTPUT / "test_metrics.json").is_file():
        status("complete", "已有测试指标，未重复训练")
        return 0
    if not (CHECKPOINT / "config.json").is_file():
        status("failed", f"找不到起始权重：{CHECKPOINT}")
        return 1
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    expected = [f"DataSet{i}" for i in range(1, 5)]
    if summary.get("source_datasets") != expected or not (DATA / "video_splits.csv").is_file():
        status("failed", "DataSet1/2/3/4 的数据摘要或视频划分缺失")
        return 1

    status("waiting_for_yolo", str(YOLO_STATUS))
    while True:
        stage = (json.loads(YOLO_STATUS.read_text(encoding="utf-8")).get("stage")
                 if YOLO_STATUS.is_file() else None)
        if stage in {"complete", "failed"}:
            break
        if not screen_running("pskus_ds1234_yolo"):
            status("failed", "YOLO 队列消失但未写出完成状态，需人工检查 GPU 是否空闲")
            return 1
        time.sleep(20)
    while yolo_running():
        time.sleep(15)

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        status("failed", f"输出目录已有未完成内容，为防止覆盖请先检查：{OUTPUT}")
        return 1
    command = [sys.executable, "scripts/train_videomae_pskus.py", "--full-pskus",
               "--datasets", ",".join(expected),
               "--split-reference", str(DATA / "video_splits.csv"),
               "--checkpoint", str(CHECKPOINT), "--output", str(OUTPUT),
               "--cache-root", "/root/pskuss_ds134_videomae_cache",
               "--other-to-wash-ratio", "1.0",
               "--epochs", "20", "--batch-size", "1", "--grad-accum", "4"]
    status("training", " ".join(command))
    with (WORK / "train.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode or not (OUTPUT / "test_metrics.json").is_file():
        status("failed", f"exit={result.returncode}; log={WORK / 'train.log'}")
        return 1
    status("complete", str(OUTPUT / "test_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
