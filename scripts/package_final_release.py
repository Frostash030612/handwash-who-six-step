#!/usr/bin/env python3
"""Archive the frozen self-scene technical release without training or inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

RELEASE = "handwash-final-20261022"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_text(path: Path, value: str) -> None:
    path.write_text(textwrap.dedent(value).lstrip(), encoding="utf-8")


def package(args: argparse.Namespace) -> None:
    repo, out = args.repo.resolve(), args.output.resolve()
    if out.exists() or out.with_suffix(".zip").exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {out}")
    out.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    for name in filter(None, tracked):
        # Old public-data binaries and first-presentation personal material are not the final system.
        if (name.startswith(("models/", "outputs/", "deliverables/", ".git"))
                or Path(name).suffix.lower() in {".pt", ".pth", ".safetensors"}):
            continue
        copy_file(repo / name, out / "source" / name)
    write_text(out / "source" / "ARCHIVE_NOTE.md", """
        # Source snapshot
        This is a source snapshot, not a Git checkout. Historical public-data model files,
        Git metadata and personal presentation materials are excluded. Run the final camera
        with the supplied runtime/models, not the historical demo commands.
        The full repository history remains on GitHub.
        """)
    model_manifest = json.loads((repo / "docs/final_release/model_manifest.json").read_text())
    for item in model_manifest["files"]:
        relative = Path(item["path"])
        src = args.runtime_models / relative.relative_to("runtime/models")
        if digest(src) != item["sha256"]:
            raise ValueError(f"Frozen model hash mismatch: {src.name}")
        copy_file(src, out / relative)
    print("Copied frozen source and model files.", flush=True)

    evaluation = out / "evaluation"
    for name in ["frame_predictions.csv", "videomae_predictions.csv", "result.json", "REPORT.zh-CN.md"]:
        copy_file(args.saved_evaluation / name, evaluation / name)
    results = json.loads((evaluation / "result.json").read_text())
    for key in ["frame_predictions", "videomae_predictions"]:
        if digest(evaluation / (key + ".csv")) != results["source_sha256"][key]:
            raise ValueError(f"Saved evaluation fingerprint mismatch: {key}")

    training = json.loads((args.historical_training_dir / "dataset_manifest.json").read_text())
    base = args.project_data_root
    supplements = json.loads((base / "补充/标注结果/manifest.json").read_text())
    supplemental_lookup = {x["clip_id"]: x for x in supplements["clips"]}
    tests = json.loads((base / "补充/测试集/标注结果/manifest.json").read_text())
    jobs = []
    for item in training["clips"]:
        number, name = item["id"], item["name"]
        if number <= 8:
            folder = base / "SelfDataset" / f"{number}_frames"
            video = base / "SelfDataset" / f"{number}.mov"
        else:
            origin = supplemental_lookup[name]
            folder = base / "补充/标注结果" / origin["frames_folder"]
            video = base / "补充" / origin["source_relative_path"]
        labels = args.video1_labels if number == 1 else folder / "frame_labels.csv"
        jobs.append((str(number), name, item["kind"], item["split"], folder, video,
                     labels, Path(f"dataset/base/{number}_frames")))
    for item in tests["clips"]:
        folder = base / "补充/测试集/标注结果" / item["frames_folder"]
        jobs.append((item["clip_id"], item["clip_id"], "continuous", "test", folder,
                     base / "补充/测试集" / item["source_relative_path"], folder / "frame_labels.csv",
                     Path("dataset/test8") / item["frames_folder"]))
    if len(jobs) != 32:
        raise ValueError("Expected 32 source clips")
    inventory = []
    for clip_id, name, kind, split, folder, video, labels, destination in jobs:
        copy_file(video, out / "dataset/raw" / (clip_id + video.suffix.lower()))
        copy_file(labels, out / destination / "frame_labels.csv")
        with labels.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            image = Path(row["image"])
            if image.is_absolute() or ".." in image.parts:
                raise ValueError(f"Invalid frame path for {clip_id}")
            copy_file(folder / image, out / destination / image)
        for extra in ["index.csv", "annotation_intervals.csv", "intervals.csv"]:
            if (folder / extra).is_file():
                copy_file(folder / extra, out / destination / extra)
        if clip_id == "1":
            copy_file(args.video1_labels.parent / "video1_annotation_intervals.csv",
                      out / destination / "annotation_intervals.csv")
        inventory.append({
            "clip_id": clip_id, "original_name": name, "kind": kind,
            "development_split": {"fit": "train", "val": "validation"}.get(split, split),
            "temporal_head_final_fit": split in {"fit", "val"},
            "raw_video": f"dataset/raw/{clip_id}{video.suffix.lower()}",
            "frames_folder": str(destination), "frame_count": len(rows),
            "video_sha256": digest(video), "frame_labels_sha256": digest(labels),
        })
        print(f"Copied dataset clip {clip_id}: {len(rows)} frames.", flush=True)
    write_json(out / "dataset/manifest.json", {
        "release": RELEASE, "source": "self-recorded only", "sample_fps": 5,
        "training_clips": 14, "validation_clips": 8, "test_clips": 10,
        "temporal_head_final_fit_clips": 22,
        "annotation_note": "Current video 8 labels: 124536, step 5 starts at 29 s. Earlier metadata counts are historical.",
        "clips": inventory,
    })
    copy_file(args.historical_training_dir / "dataset_manifest.json",
              out / "dataset/development_manifest_20261019.json")
    copy_file(base / "SelfDataset/annotations/review_20261021_video8_45_correction/correction.json",
              out / "dataset/video8_correction_20261021.json")
    with (out / "dataset/splits.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["clip_id", "original_name", "kind", "development_split", "temporal_head_final_fit", "frame_count"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)
    copy_file(repo / "docs/final_release/label_map.json", out / "dataset/label_map.json")
    write_text(out / "dataset/README.md", """
        # Frozen self-recorded dataset
        manifest.json and splits.csv describe this package. base/ has original IDs 1–8,
        supplementary IDs 9–16, isolated step-4 IDs 17–20 and step-5 IDs 21–24.
        test8/ contains the eight additional test clips. raw/ has all 32 original videos.
        frame_labels.csv files are copied unchanged; use their actual timestamp columns.
        Transition rows are ignored according to their validity flags; do not convert them to other.
        Original video 1's label file comes from the reviewed 5 fps annotation export.
        Video 8 uses the 2026-10-21 correction (124536, step 5 at 29 seconds).
        development_manifest_20261019.json is the original training manifest, kept for provenance
        and for the archived VideoMAE loader. Its old video-8 test counts are superseded by
        manifest.json and the current base/8_frames/frame_labels.csv. It is not a new split.
        The GRU training source refits on all 22 development clips; test clips are excluded.
        No PSKUS source data or trained weights are included in this package.
        """)

    for name in ["train_yolo.py", "train_heads_scratch.py", "prepare_cloud_data.py"]:
        copy_file(args.historical_training_dir / name, out / "provenance/training_source" / name)
    write_text(out / "provenance/training_source/README.md", """
        # Retained training source
        These are historical cloud scripts, copied unchanged as evidence. They contain
        original cloud paths and some stale comments; executable fit lists determine the
        documented 14/8 selection and 22-clip temporal refit. They are not camera launchers
        and are not automatically executed. No retraining is performed by this release.
        The development manifest records the effective final split; the earlier package
        builder with an obsolete split is intentionally not used to build this archive.
        """)
    for name in ["architecture_timing.json", "incremental_equivalence.json", "duration_regression.json"]:
        copy_file(args.deployment_evidence / name, out / "provenance/deployment" / name)
    # Capture package versions only, excluding local file URLs and private index credentials.
    write_text(out / "provenance/README.md", """
        # Provenance scope
        Existing checks and timing are historical evidence, not new experiments.
        Synthetic architecture timing uses different inference units; it is not a fair
        sustained throughput or peak-memory benchmark. Read source/docs/FINAL_RELEASE.md.
        """)
    write_json(out / "RELEASE.json", {
        "release": RELEASE, "source_commit": revision,
        "repository": "https://github.com/Frostash030612/handwash-who-six-step",
        "data_clips": len(inventory), "sampled_frames": sum(x["frame_count"] for x in inventory),
        "weights_unchanged": True, "new_training_or_inference": False,
        "scope": "technical delivery; coursework report, final presentation and individual reports are separate",
    })
    write_text(out / "install.sh", """
        #!/bin/sh
        set -eu
        cd "$(dirname "$0")"
        "${PYTHON_BIN:-python3.12}" -m venv .venv
        .venv/bin/python -m pip install -r source/docs/final_release/requirements-camera.txt -e './source[compare]'
        printf '\\nInstalled. Start with: .venv/bin/python start_camera.py\\n'
        """)
    write_text(out / "install.ps1", r"""
        $ErrorActionPreference = 'Stop'
        Set-Location $PSScriptRoot
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
        & .\.venv\Scripts\python.exe -m pip install -r source/docs/final_release/requirements-camera.txt -e './source[compare]'
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
        Write-Host 'Start with: .\.venv\Scripts\python.exe start_camera.py'
        """)
    write_text(out / "start_camera.py", """
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent
        command = [sys.executable, str(root / 'source/scripts/run_self_scene_camera.py'),
                   '--runtime-dir', str(root / 'runtime'),
                   'config=' + str(root / 'source/configs/experiments/self_scene_camera.yaml'),
                   *sys.argv[1:]]
        raise SystemExit(subprocess.call(command, cwd=root / 'source'))
        """)
    write_text(out / "verify_files.py", """
        import hashlib
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent
        entries = json.loads((root / 'SHA256SUMS.json').read_text())
        failed = []
        for entry in entries:
            path = root / entry['path']
            if not path.is_file():
                failed.append(entry['path'])
                continue
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            if value.hexdigest() != entry['sha256']:
                failed.append(entry['path'])
        print(f'Checked {len(entries)} files; mismatches: {len(failed)}')
        for name in failed:
            print(name)
        raise SystemExit(bool(failed))
        """)
    write_text(out / "README.md", """
        # Handwashing final technical package
        Read source/docs/FINAL_RELEASE.md or source/docs/FINAL_RELEASE.zh-CN.md first.
        This package includes the frozen four models, source code, 32 self-recorded videos,
        annotated sample frames and previously saved evaluation results. No retraining occurred.

        ## Quick start
        Install Python 3.12, then from this directory:
        ```sh
        PYTHON_BIN=python3.12 bash install.sh
        .venv/bin/python start_camera.py
        ```
        Open http://127.0.0.1:8876/ and grant camera permission.
        If 8876 is occupied, append --port 8877. Installation needs internet; inference
        uses included weights. For Windows use install.ps1 and .venv/Scripts/python.exe.
        The archive was assembled on macOS; Windows has not been newly validated.

        ## Reproduce the saved comparison
        ```sh
        .venv/bin/python source/scripts/compare_self_scene_common_centers.py \\
          --frame-predictions evaluation/frame_predictions.csv \\
          --videomae-predictions evaluation/videomae_predictions.csv \\
          --output reproduced_metrics
        ```
        python verify_files.py checks file hashes only and does not evaluate the models.
        Dataset split and the 22-clip temporal-head refit are documented in dataset/README.md.
        The final metric table comes from saved 10-clip predictions, not the old checkpoint metrics.
        No camera-session history, caches or credentials are included.
        Final coursework report, slides, presentation video and individual reports are separate.
        """)
    entries = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(out.rglob("*")) if p.is_file()]
    write_json(out / "SHA256SUMS.json", entries)
    print(f"Prepared {len(entries)} files; creating ZIP (stored media, no recompression).", flush=True)
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zipped.write(p, Path(out.name) / p.relative_to(out))
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(checksum + "  " + archive.name + "\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": checksum, "files": len(entries), "source_commit": revision}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["repo", "project-data-root", "runtime-models", "historical-training-dir",
                 "video1-labels", "saved-evaluation", "deployment-evidence", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
