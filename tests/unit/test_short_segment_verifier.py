"""A verifier may review short proposals without silently deleting true steps."""

import numpy as np
from scripts.train_short_segment_verifier import _apply_rejections, _proposal_truth

from handwash.core.segment_metrics import ActionSegment


def test_true_one_centre_step_is_positive_training_example() -> None:
    actual = np.asarray([1, 1, 3, 2, 2])
    assert _proposal_truth(ActionSegment(3, 2, 3), actual) == 1


def test_unknown_centre_is_not_a_hard_negative() -> None:
    actual = np.asarray([1, -1, 2])
    assert _proposal_truth(ActionSegment(3, 1, 2), actual) is None


def test_rejected_island_merges_neighbouring_same_step() -> None:
    item = {
        "labels": np.asarray([1, 1, 3, 1, 1]),
        "proposals": [ActionSegment(3, 2, 3)],
        "template": {"targets": np.asarray([1, 1, 1, 1, 1])},
    }
    result = _apply_rejections(item, np.asarray([0.1]), threshold=0.5)
    assert result["probs"].argmax(axis=1).tolist() == [1, 1, 1, 1, 1]


def test_rejected_true_step_remains_countable_error() -> None:
    item = {
        "labels": np.asarray([1, 3, 2]),
        "proposals": [ActionSegment(3, 1, 2)],
        "template": {"targets": np.asarray([1, 3, 2])},
    }
    result = _apply_rejections(item, np.asarray([0.1]), threshold=0.5)
    assert result["probs"].argmax(axis=1).tolist() == [1, 0, 2]
