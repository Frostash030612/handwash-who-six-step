"""PSKUS 双标注一致与边界帧清洗的纯函数测试。"""

from __future__ import annotations

import pytest

from handwash.core.labels import Step
from handwash.errors import DataError
from handwash.pipelines.prepare import _pskuss_consensus_stable_mask


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
