"""Boundary-near class labels must use only two-annotator agreement."""

import random

import numpy as np
from scripts.train_boundary_yolo_gru import _crop_batch


def _item() -> dict:
    stable = np.ones(64, dtype=np.int64)
    consensus = stable.copy()
    stable[8] = -1  # Agreed, but excluded by the original stable-label rule.
    stable[9] = -1
    consensus[9] = -1  # Annotators do not agree: never supervise this frame.
    return {
        "features": np.zeros((64, 2), dtype=np.float32),
        "stable": stable, "consensus": consensus,
        "boundary_targets": np.zeros(64, dtype=np.float32),
        "boundary_mask": np.ones(64, dtype=bool),
    }


def test_default_reproduces_stable_only_supervision() -> None:
    _, labels, weights, *_ = _crop_batch([_item()], random.Random(42))
    assert labels[0, 8] == -1 and weights[0, 8] == 0
    assert labels[0, 9] == -1 and weights[0, 9] == 0


def test_soft_transition_supervision_never_uses_disagreement() -> None:
    _, labels, weights, *_ = _crop_batch(
        [_item()], random.Random(42), transition_supervision_weight=0.25
    )
    assert labels[0, 8] == 1 and weights[0, 8] == 0.25
    assert labels[0, 9] == -1 and weights[0, 9] == 0
    assert labels[0, 10] == 1 and weights[0, 10] == 1
