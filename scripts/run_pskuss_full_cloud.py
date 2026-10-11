#!/usr/bin/env python
"""Resume full PSKUS download, then train YOLO and VideoMAE on AutoDL.

Run inside ``screen``. Each stage has its own log and a machine-readable
status file. The four ``pskus_dl*`` screens may already be downloading shards;
this driver waits for them, verifies all files, extracts, and continues.
"""

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

WORK = PROJECT_ROOT / "outputs/pskuss_full_pipeline"
STATUS = WORK / "status.json"
PYTHON = sys.executable


def update(stage: str, state: str, *, detail: str = "") -> None:
    payload = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"history": []}
    payload["stage"] = stage
    payload["state"] = state
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    payload["detail"] = detail
    payload["history"].append({"stage": stage, "state": state,
                               "time_utc": payload["updated_utc"], "detail": detail})
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{payload['updated_utc']}] {stage}: {state} {detail}", flush=True)


def run(stage: str, command: list[str], *, done: Path | None = None,
        retries: int = 1) -> None:
    if done is not None and done.exists():
        update(stage, "skipped", detail=f"已有产物：{done}")
        return
    update(stage, "running", detail=" ".join(command))
    log = WORK / f"{stage}.log"
    for attempt in range(1, retries + 1):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {datetime.now(timezone.utc).isoformat()} attempt {attempt}/{retries} ===\n")
            handle.flush()
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            break
        if attempt < retries:
            update(stage, "retrying", detail=f"exit={result.returncode}; attempt={attempt}/{retries}; log={log}")
            time.sleep(60)
    if result.returncode:
        update(stage, "failed", detail=f"exit={result.returncode}; log={log}")
        raise SystemExit(result.returncode)
    if done is not None and not done.exists():
        update(stage, "failed", detail=f"命令退出 0，但缺少产物：{done}")
        raise SystemExit(1)
    update(stage, "complete", detail=f"log={log}")


def wait_for_shards() -> None:
    while True:
        result = subprocess.run(["screen", "-ls"], capture_output=True, text=True, check=False)
        # ``pskus_dl_handoff`` is a long-lived placeholder screen and can be
        # idle; it must not block the resumable downloader below forever.
        active = [line.strip() for line in result.stdout.splitlines()
                  if ".pskus_dl" in line and ".pskus_dl_handoff" not in line]
        if not active:
            return
        update("parallel_downloads", "running", detail="; ".join(active))
        time.sleep(180)


def check_extracted() -> None:
    root = PROJECT_ROOT / "data/raw/pskuss"
    ledger = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    for number in range(1, 12):
        name = f"DataSet{number}.zip"
        if not ledger["files"][name]["verified"] or not (root / name).is_file():
            raise SystemExit(f"未通过 MD5：{name}")
        if not (root / f"{name}.extracted").is_file():
            raise SystemExit(f"未完成解压：{name}")
        dataset = root / "extracted" / f"DataSet{number}"
        if not (dataset / "Videos").is_dir() or not (dataset / "Annotations").is_dir():
            raise SystemExit(f"解压目录不完整：{dataset}")
    update("source_audit", "complete", detail="11 个视频分片 MD5 与解压目录均已确认")


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    wait_for_shards()
    run("download_verify_extract", [PYTHON, "scripts/download_data.py", "--dataset", "pskuss", "--all", "--extract"],
        retries=3)
    check_extracted()
    run("build_yolo_images", [PYTHON, "scripts/build_pskuss_full_yolo_dataset.py"],
        done=PROJECT_ROOT / "data/processed/pskuss_full_yolo/summary.json")
    run("train_yolo", [PYTHON, "scripts/train_pskuss_full_yolo.py",
                       "--epochs", "30", "--patience", "5"],
        done=PROJECT_ROOT / "outputs/pskuss_full_yolo/test_metrics.json")
    run("test_yolo_full_video", [PYTHON, "scripts/evaluate_yolo_full_videos.py",
                                 "--full-pskus", "--model", "outputs/pskuss_full_yolo/weights/best.pt",
                                 "--output", "outputs/pskuss_full_yolo_video_test"],
        done=PROJECT_ROOT / "outputs/pskuss_full_yolo_video_test/summary.json")
    run("train_videomae", [PYTHON, "scripts/train_videomae_pskus.py", "--full-pskus",
                           "--checkpoint", "outputs/videomae_pskus/best_model",
                           "--output", "outputs/pskuss_full_videomae",
                           "--cache-root", "/root/pskuss_full_videomae_cache",
                           "--epochs", "20", "--batch-size", "1", "--grad-accum", "4"],
        done=PROJECT_ROOT / "outputs/pskuss_full_videomae/test_metrics.json")
    run("test_videomae_full_video", [PYTHON, "scripts/evaluate_videomae_full_videos.py",
                                     "--full-pskus", "--model", "outputs/pskuss_full_videomae/best_model",
                                     "--output", "outputs/pskuss_full_videomae_video_test",
                                     "--smooth-windows", "3", "--batch-size", "4"],
        done=PROJECT_ROOT / "outputs/pskuss_full_videomae_video_test/summary.json")
    update("all", "complete", detail="YOLO 和 VideoMAE 全量训练及测试完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
