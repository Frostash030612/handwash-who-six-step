#!/usr/bin/env python
"""Combine filtered PSKUS + Jurmala frames into one 7-class training set.

Run only after both complete raw archives have been MD5 verified and both
``prepare_data.py`` manifests have been generated with the dual-annotator
quality filter. Images are hard-linked, so this does not duplicate their bytes.
The eight public demo videos are moved to test if present locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import PROJECT_ROOT
from handwash.core.labels import Step
from handwash.data_sources import iter_registered_files
from handwash.io.manifest import read_manifest, write_manifest

SOURCES = ("pskuss", "jurmala")
CLASSES = {
    Step.OTHER: "0_other",
    Step.STEP_1: "1_palm_to_palm",
    Step.STEP_2: "2_palm_over_dorsum",
    Step.STEP_3: "3_palm_to_palm_interlaced",
    Step.STEP_4: "4_backs_of_fingers",
    Step.STEP_5: "5_rotational_thumbs",
    Step.STEP_6: "6_rotational_fingertips",
}


def _require_complete_source(data_root: Path, source: str) -> None:
    raw = data_root / "raw" / source
    inventory = raw / "SOURCES.json"
    if not inventory.is_file():
        raise SystemExit(f"{source} 缺少已校验的来源清单：{inventory}")
    files = json.loads(inventory.read_text(encoding="utf-8")).get("files", {})
    missing = [
        name for name, _, _ in iter_registered_files(source)
        if name.endswith(".zip") and (
            not (raw / name).is_file() or not files.get(name, {}).get("verified")
        )
    ]
    if missing:
        raise SystemExit(f"{source} 原始分片未下全/未校验：{', '.join(missing)}")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.samefile(source):
            raise SystemExit(f"目标文件已存在且并非同一帧：{target}")
        return
    try:
        os.link(source, target)
    except OSError as exc:
        raise SystemExit(
            f"无法硬链接 {source} -> {target}：{exc}。"
            "源帧与输出需要在同一磁盘；不要改用复制直到确认剩余空间。"
        ) from exc


def _project_label(label: Step) -> Step:
    if label is Step.UNKNOWN:
        raise SystemExit("源 manifest 包含 unknown；双标注过滤可能未生效")
    return label if label in CLASSES and label is not Step.OTHER else Step.OTHER


def build(data_root: Path, output: Path) -> dict:
    for source in SOURCES:
        _require_complete_source(data_root, source)

    marker = output / ".building"
    summary_path = output / "summary.json"
    if output.exists() and summary_path.exists() and not marker.exists():
        raise SystemExit(f"数据集已经构建：{output}；如需重建，请指定一个新的 --output 路径")
    if output.exists() and not marker.exists():
        raise SystemExit(f"输出目录已存在但不是可续跑状态：{output}")
    output.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)

    demo_dir = data_root / "external" / "pskus_demo_validation"
    demo_stems = {p.stem for p in demo_dir.rglob("*.mp4")} if demo_dir.is_dir() else set()
    held_out_clips: set[str] = set()
    source_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    clip_split: dict[str, str] = {}
    combined = []

    for source in SOURCES:
        source_root = data_root / "processed" / source
        manifest = source_root / "manifest.csv"
        if not manifest.is_file():
            raise SystemExit(f"{source} 缺少预处理帧清单：{manifest}")
        records = read_manifest(manifest)
        for rec in records:
            if rec.dataset != source or rec.split not in ("train", "val", "test"):
                raise SystemExit(f"来源/划分不符：{source} {rec.clip_id} {rec.split}")
            image = source_root / "frames" / rec.image_path
            if not image.is_file():
                raise SystemExit(f"帧图像不存在：{image}")

            clip_id = f"{source}/{rec.clip_id}"
            split = "test" if source == "pskuss" and Path(rec.clip_id).name in demo_stems else rec.split
            if split == "test" and rec.split != "test" and source == "pskuss":
                held_out_clips.add(clip_id)
            if clip_id in clip_split and clip_split[clip_id] != split:
                raise SystemExit(f"同一视频跨划分：{clip_id}")
            clip_split[clip_id] = split

            label = _project_label(rec.label)
            relative = Path(source) / rec.image_path
            _link(image, output / "frames" / relative)
            digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:16]
            name = f"{source}_{digest}_{rec.frame_index:07d}{image.suffix.lower()}"
            _link(image, output / "ultralytics" / split / CLASSES[label] / name)
            combined.append(replace(
                rec,
                clip_id=clip_id,
                image_path=relative.as_posix(),
                label=label,
                split=split,
            ))
            source_counts[source] += 1
            class_counts[split][CLASSES[label]] += 1

    write_manifest(output / "manifest.csv", combined)
    summary = {
        "sources": list(SOURCES),
        "quality_rule": "two annotators agree; stable_radius_frames=7 in source annotations",
        "label_rule": "WHO steps 1-6 retained; all other source actions mapped to 0_other",
        "demo_holdout": sorted(held_out_clips),
        "source_frames": dict(source_counts),
        "clips_by_split": dict(Counter(clip_split.values())),
        "classes_by_split": {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
        "total_frames": len(combined),
        "hardlinks": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    marker.unlink()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/hospitals_full_7class")
    args = parser.parse_args()
    result = build(args.data_root.resolve(), args.output.resolve())
    print(f"七类完整数据集：{args.output.resolve()}")
    print(f"训练帧总数：{result['total_frames']:,}")
    print(f"原始视频划分：{result['clips_by_split']}")
    print(f"单独保留的演示视频：{len(result['demo_holdout'])} 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
