"""Transition-aware smoothing for full-video action predictions."""

from __future__ import annotations

import numpy as np

from scripts.evaluate_yolo_full_videos import smooth_hysteresis


def _probs(labels: list[int], confidence: float = 0.95) -> np.ndarray:
    result = np.full((len(labels), 3), (1.0 - confidence) / 2, dtype=np.float32)
    for index, label in enumerate(labels):
        result[index, label] = confidence
    return result


def test_hysteresis_suppresses_one_window_action_spike() -> None:
    actual = smooth_hysteresis(_probs([0, 0, 1, 0, 0]), width=1,
                               confirm_frames=2, margin=0.05)

    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 0])


def test_hysteresis_backdates_a_confirmed_action_transition() -> None:
    actual = smooth_hysteresis(_probs([0, 0, 1, 1, 1]), width=1,
                               confirm_frames=2, margin=0.05)

    np.testing.assert_array_equal(actual, [0, 0, 1, 1, 1])


def test_hysteresis_rejects_persistent_but_weak_transition() -> None:
    probs = np.array([[0.52, 0.46, 0.02]] * 2 + [[0.48, 0.50, 0.02]] * 3,
                     dtype=np.float32)
    actual = smooth_hysteresis(probs, width=1, confirm_frames=2, margin=0.10)

    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 0])


def test_hysteresis_does_not_force_canonical_who_order() -> None:
    actual = smooth_hysteresis(_probs([2, 2, 1, 1, 1]), width=1,
                               confirm_frames=2, margin=0.05)

    np.testing.assert_array_equal(actual, [2, 2, 1, 1, 1])
