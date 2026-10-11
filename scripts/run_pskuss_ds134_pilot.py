#!/usr/bin/env python
"""After DataSet3 uploads, train isolated YOLO and VideoMAE pilot models on 1/3/4."""

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

WORK = PROJECT_ROOT / "outputs/pskuss_ds134_pilot"
STATUS = WORK / "status.json"
RAW = PROJECT_ROOT / "data/raw/pskuss"
PYTHON = sys.executable


def update(stage: str, state: str, detail: str = "") -> None:
    payload = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"history": []}
    payload.update(stage=stage, state=state, updated_utc=datetime.now(timezone.utc).isoformat(), detail=detail)
    payload["history"].append({"stage": stage, "state": state,
                               "time_utc": payload["updated_utc"], "detail": detail})
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{payload['updated_utc']}] {stage}: {state} {detail}", flush=True)


def run(stage: str, command: list[str], done: Path | None = None) -> None:
    if done is not None and done.exists():
        update(stage, "skipped", str(done))
        return
    update(stage, "running", " ".join(command))
    log = WORK / f"{stage}.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {datetime.now(timezone.utc).isoformat()} ===\n")
        handle.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        update(stage, "failed", f"exit={result.returncode}; log={log}")
        raise SystemExit(result.returncode)
    if done is not None and not done.exists():
        update(stage, "failed", f"命令退出 0，但缺少产物：{done}")
        raise SystemExit(1)
    update(stage, "complete", f"log={log}")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    archive = RAW / "DataSet3.zip"
    update("wait_for_dataset3_upload", "running", str(archive))
    while not archive.is_file() or list(RAW.glob(".DataSet3.zip.*")):
        time.sleep(15)

    run("verify_and_extract_dataset3", [PYTHON, "scripts/download_data.py", "--dataset", "pskuss",
                                       "--files", "DataSet3.zip", "--extract"],
        done=RAW / "DataSet3.zip.extracted")
    run("build_yolo_data_1_3_4", [PYTHON, "scripts/build_pskuss_full_yolo_dataset.py",
                                  "--datasets", "DataSet1,DataSet3,DataSet4",
                                  "--output", "data/processed/pskuss_ds134_yolo"],
        done=PROJECT_ROOT / "data/processed/pskuss_ds134_yolo/summary.json")
    run("train_yolo_1_3_4", [PYTHON, "scripts/train_pskuss_full_yolo.py",
                             "--data", "data/processed/pskuss_ds134_yolo",
                             "--output", "outputs/pskuss_ds134_yolo",
                             "--epochs", "30", "--patience", "5"],
        done=PROJECT_ROOT / "outputs/pskuss_ds134_yolo/test_metrics.json")
    run("train_videomae_1_3_4", [PYTHON, "scripts/train_videomae_pskus.py", "--full-pskus",
                                 "--checkpoint", "outputs/videomae_pskus/best_model",
                                 "--output", "outputs/pskuss_ds134_videomae_stable",
                                 "--cache-root", "/root/pskuss_ds134_videomae_cache",
                                 "--epochs", "20", "--batch-size", "1", "--grad-accum", "4"],
        done=PROJECT_ROOT / "outputs/pskuss_ds134_videomae_stable/test_metrics.json")
    update("pilot_complete", "complete", "YOLO 与 VideoMAE 的 DataSet1/3/4 试训完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
