#!/usr/bin/env python
"""Queue DataSet1/2/3/4 YOLO fine-tuning after data build and VideoMAE finish."""

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

WORK = PROJECT_ROOT / "outputs/pskuss_ds1234_yolo_queue"
DATA = PROJECT_ROOT / "data/processed/pskuss_ds1234_yolo"
MODEL = PROJECT_ROOT / "outputs/pskuss_ds134_yolo/weights/best.pt"
OUTPUT = PROJECT_ROOT / "outputs/pskuss_ds1234_yolo"


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


def videomae_running() -> bool:
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True)
    return any("scripts/train_videomae_pskus.py" in line and
               "python" in line.split()[0]
               for line in result.stdout.splitlines() if line.split())


def main() -> int:
    if (OUTPUT / "test_metrics.json").is_file():
        status("complete", "已有测试指标，未重复训练")
        return 0
    if not MODEL.is_file():
        status("failed", f"找不到起始权重：{MODEL}")
        return 1
    status("waiting_for_dataset", str(DATA))
    while not (DATA / "summary.json").is_file() or (DATA / ".building").exists():
        if not screen_running("pskus_ds1234_build"):
            status("failed", f"数据构建进程已结束，但缺少完整产物；检查 {PROJECT_ROOT / 'outputs/pskuss_ds1234_build.log'}")
            return 1
        time.sleep(15)
    status("waiting_for_gpu", "等待当前 VideoMAE 训练结束，不同时抢占 GPU")
    while videomae_running():
        time.sleep(15)
    command = [sys.executable, "scripts/train_pskuss_full_yolo.py",
               "--model", str(MODEL), "--data", str(DATA), "--output", str(OUTPUT),
               "--epochs", "30", "--patience", "5"]
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
