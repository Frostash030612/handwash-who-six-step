"""Temporal segment audit must not reward extra or missing WHO steps."""

from __future__ import annotations

import numpy as np
import pytest

from handwash.core.segment_metrics import action_segments, segment_counts


def test_segments_break_at_other_and_unannotated_centres() -> None:
    actual = action_segments(
        np.asarray([1, 1, 0, 2, 2, 2, 3, 3]),
        np.asarray([True, True, True, True, False, True, True, True]),
    )

    assert [(part.label, part.start, part.end) for part in actual] == [
        (1, 0, 2),
        (2, 3, 4),
        (2, 5, 6),
        (3, 6, 8),
    ]


def test_single_spurious_step_costs_one_false_positive_not_whole_video() -> None:
    truth = action_segments(np.asarray([1, 1, 1, 2, 2, 2]), np.ones(6, dtype=bool))
    predicted = action_segments(np.asarray([1, 1, 3, 2, 2, 2]), np.ones(6, dtype=bool))

    assert segment_counts(truth, predicted) == (2, 1, 0)


def test_missing_short_true_step_is_false_negative() -> None:
    truth = action_segments(np.asarray([1, 1, 3, 2, 2]), np.ones(5, dtype=bool))
    predicted = action_segments(np.asarray([1, 1, 1, 2, 2]), np.ones(5, dtype=bool))

    assert segment_counts(truth, predicted) == (2, 0, 1)


def test_one_to_one_matching_penalizes_repeated_prediction() -> None:
    truth = action_segments(np.asarray([1, 1, 1, 1, 1]), np.ones(5, dtype=bool))
    predicted = action_segments(np.asarray([1, 1, 0, 1, 1]), np.ones(5, dtype=bool))

    assert segment_counts(truth, predicted) == (0, 2, 1)


def test_overlap_threshold_is_enforced() -> None:
    truth = action_segments(np.asarray([1, 1, 1, 1]), np.ones(4, dtype=bool))
    predicted = action_segments(np.asarray([0, 0, 1, 1]), np.ones(4, dtype=bool))

    assert segment_counts(truth, predicted, overlap=0.5) == (1, 0, 0)
    assert segment_counts(truth, predicted, overlap=0.75) == (0, 1, 1)
    with pytest.raises(ValueError):
        segment_counts(truth, predicted, overlap=0)
