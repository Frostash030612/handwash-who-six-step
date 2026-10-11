"""PSKUS 双标注一致与边界帧清洗的纯函数测试。"""

from __future__ import annotations

import pytest

from handwash.core.labels import Step
from handwash.errors import DataError
from handwash.pipelines.prepare import (
    _label_for_sample,
    _pskuss_consensus_stable_mask,
    _records_from_pskuss,
)


def test_consensus_filter_removes_transition_boundary_frames() -> None:
    first = Step.STEP_1
    second = Step.STEP_2
    labels = [first, first, first, first, second, second, second, second]

    assert _pskuss_consensus_stable_mask(
        labels, labels, stable_radius_frames=1
    ) == [True, True, True, False, False, True, True, True]


def test_consensus_filter_removes_disagreement_and_its_neighbours() -> None:
    first = Step.STEP_1
    primary = [first, first, first, first, first]
    secondary = [first, first, Step.STEP_2, first, first]

    assert _pskuss_consensus_stable_mask(
        primary, secondary, stable_radius_frames=1
    ) == [True, False, False, False, True]


def test_consensus_filter_rejects_unaligned_annotation_sequences() -> None:
    with pytest.raises(DataError, match="帧数不一致"):
        _pskuss_consensus_stable_mask(
            [Step.STEP_1], [Step.STEP_1, Step.STEP_1], stable_radius_frames=0
        )


def test_rejected_frames_stay_rejected_during_extraction(tmp_path) -> None:
    from handwash.core.config import load_config

    clip_dir = tmp_path / "DataSet1"
    video_dir = clip_dir / "Videos"
    primary_dir = clip_dir / "Annotations" / "Annotator1"
    secondary_dir = clip_dir / "Annotations" / "Annotator2"
    for directory in (video_dir, primary_dir, secondary_dir):
        directory.mkdir(parents=True)
    (video_dir / "sample.mp4").touch()

    def write_labels(directory, codes):
        rows = ["frame_time,is_washing,movement_code"]
        rows.extend(f"{index * 1000},1,{code}" for index, code in enumerate(codes))
        (directory / "sample.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    write_labels(primary_dir, [1, 1, 1, 1, 1])
    write_labels(secondary_dir, [1, 1, 2, 1, 1])
    rc = load_config(
        ["configs/config.yaml", "configs/data/pskuss.yaml"],
        overrides={"datasets": {"pskuss": {"quality_filter": {"stable_radius_frames": 1}}}},
    )
    frames, clips = _records_from_pskuss(rc, tmp_path)

    assert len(clips) == 1
    assert [frame.label for frame in frames] == [
        Step.STEP_1, Step.UNKNOWN, Step.UNKNOWN, Step.UNKNOWN, Step.STEP_1
    ]
    assert clips[0].label_sequence == tuple(frame.label for frame in frames)
    assert _label_for_sample(clips[0], source_index=2, timestamp_s=2.0) is None
