"""The full seven-class export preserves source-video splits and provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from handwash.core.labels import Step
from handwash.core.schema import FrameRecord
from handwash.io.manifest import read_manifest, write_manifest


def _load_builder():
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_full_hospital_dataset.py"
    spec = importlib.util.spec_from_file_location("build_full_hospital_dataset", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combined_export_is_seven_class_and_holds_out_demo(tmp_path, monkeypatch) -> None:
    builder = _load_builder()
    monkeypatch.setattr(builder, "_require_complete_source", lambda *_: None)
    for source, clip, label in (
        ("pskuss", "DataSet1/demo", Step.STEP_1),
        ("jurmala", "Location_2/other", Step.FAUCET_OFF),
    ):
        root = tmp_path / "processed" / source
        image = root / "frames" / clip / "00000.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"frame")
        write_manifest(root / "manifest.csv", [FrameRecord(
            clip_id=clip,
            frame_index=0,
            image_path=f"{clip}/00000.jpg",
            label=label,
            dataset=source,
            split="train",
            timestamp_s=0.0,
        )])
    demo = tmp_path / "external" / "pskus_demo_validation" / "demo.mp4"
    demo.parent.mkdir(parents=True)
    demo.touch()

    output = tmp_path / "combined"
    summary = builder.build(tmp_path, output)
    records = read_manifest(output / "manifest.csv")

    assert summary["total_frames"] == 2
    assert summary["demo_holdout"] == ["pskuss/DataSet1/demo"]
    assert {(rec.dataset, rec.split, rec.label) for rec in records} == {
        ("pskuss", "test", Step.STEP_1),
        ("jurmala", "train", Step.OTHER),
    }
    assert (output / "ultralytics/test/1_palm_to_palm").is_dir()
    assert (output / "ultralytics/train/0_other").is_dir()
    for rec in records:
        source = tmp_path / "processed" / rec.dataset / "frames" / Path(rec.image_path).relative_to(rec.dataset)
        assert (output / "frames" / rec.image_path).samefile(source)
