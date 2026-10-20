"""Boundary decoders: legacy behaviour is frozen, new evidence protection is explicit."""

from __future__ import annotations

import numpy as np
import pytest

from handwash.core.boundary_decoding import (
    apply_decoder,
    confident_runs,
    decode_boundaries,
    decode_boundary_blend,
    seconds_to_frames,
)


def _legacy_reference(probs: np.ndarray, edges: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    """Verbatim 2026-10-12 algorithm, kept to prove the legacy decoder did not drift."""
    left = np.r_[-np.inf, edges[:-1]]
    right = np.r_[edges[1:], -np.inf]
    peaks = np.flatnonzero((edges >= threshold) & (edges >= left) & (edges > right))
    chosen: list[int] = []
    for position in sorted(peaks.tolist(), key=lambda index: float(edges[index]), reverse=True):
        if position == 0 or position >= len(probs):
            continue
        if all(abs(position - other) >= min_gap for other in chosen):
            chosen.append(position)
    boundaries = sorted(chosen)
    result = probs.copy()
    for begin, end in zip([0, *boundaries], [*boundaries, len(probs)], strict=True):
        if end > begin:
            result[begin:end] = probs[begin:end].mean(axis=0, keepdims=True)
    return result


def _onehot(labels: list[int], confidence: float, classes: int = 7) -> np.ndarray:
    probs = np.full((len(labels), classes), (1 - confidence) / (classes - 1), dtype=np.float32)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def test_legacy_decoder_is_unchanged_on_random_sequences() -> None:
    rng = np.random.default_rng(20261020)
    for _ in range(50):
        length = int(rng.integers(2, 80))
        probs = rng.dirichlet(np.ones(7), size=length).astype(np.float32)
        edges = rng.random(length).astype(np.float32)
        threshold, gap = float(rng.choice([0.25, 0.3, 0.5])), int(rng.integers(1, 6))
        np.testing.assert_array_equal(
            decode_boundaries(probs, edges, threshold=threshold, min_gap=gap),
            _legacy_reference(probs, edges, threshold, gap),
        )


def test_confident_opening_other_is_not_swallowed_by_segment_mean() -> None:
    # Video 8 failure: 18 clear "other" samples, no boundary peak before step 1.
    probs = np.vstack([_onehot([0] * 18, 0.996), _onehot([1] * 42, 0.9)])
    edges = np.full(60, 0.02, dtype=np.float32)

    legacy = decode_boundaries(probs, edges, threshold=0.3, min_gap=4).argmax(axis=1)
    blended = decode_boundary_blend(probs, edges, threshold=0.3, min_gap=4, raw_weight=0.5,
                                    protect_min_frames=3, protect_confidence=0.9).argmax(axis=1)

    assert legacy[:18].tolist() == [1] * 18
    assert blended.tolist() == [0] * 18 + [1] * 42


def test_short_confident_outlier_is_averaged_with_its_segment_not_protected() -> None:
    labels = [2] * 10 + [5, 5] + [2] * 10
    probs = _onehot(labels, 0.95)
    edges = np.zeros(len(labels), dtype=np.float32)

    result = decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=0.5,
                                   protect_min_frames=3, protect_confidence=0.9)

    # The two-sample run is not cut out: it is blended with the mean of the whole segment.
    outlier_mean = probs[:10].mean(axis=0)
    np.testing.assert_allclose(result[:10], 0.5 * probs[:10] + 0.5 * outlier_mean, rtol=1e-6)
    np.testing.assert_allclose(result[10:12], 0.5 * probs[10:12] + 0.5 * probs[10:12].mean(axis=0), rtol=1e-6)
    assert confident_runs(probs, min_frames=3, min_confidence=0.9) == [(0, 10), (12, 22)]


def test_three_sample_confident_run_is_protected_at_point_six_seconds() -> None:
    assert seconds_to_frames(0.6, 5.0) == 3
    assert seconds_to_frames(3.0, 5.0) == 15
    labels = [2] * 10 + [5, 5, 5] + [2] * 10
    assert (2, 3) not in confident_runs(_onehot(labels, 0.95), min_frames=3, min_confidence=0.9)
    assert (10, 13) in confident_runs(_onehot(labels, 0.95), min_frames=3, min_confidence=0.9)
    assert confident_runs(_onehot(labels, 0.85), min_frames=3, min_confidence=0.9) == []


def test_raw_weight_one_is_the_no_averaging_control() -> None:
    probs = np.random.default_rng(1).dirichlet(np.ones(7), size=30).astype(np.float32)
    edges = np.random.default_rng(2).random(30).astype(np.float32)

    result = decode_boundary_blend(probs, edges, threshold=0.3, min_gap=3, raw_weight=1.0,
                                   protect_min_frames=3, protect_confidence=0.9)

    np.testing.assert_array_equal(result, probs)


def test_boundary_peaks_and_protected_runs_both_cut_segments() -> None:
    probs = np.vstack([_onehot([3] * 6, 0.6), _onehot([4] * 6, 0.6), _onehot([0] * 4, 0.97)])
    edges = np.zeros(16, dtype=np.float32)
    edges[6] = 0.9  # boundary head cut between the two unconfident steps

    result = decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=0.0,
                                   protect_min_frames=3, protect_confidence=0.9)

    np.testing.assert_allclose(result[:6], np.repeat(probs[:6].mean(axis=0, keepdims=True), 6, axis=0))
    np.testing.assert_allclose(result[6:12], np.repeat(probs[6:12].mean(axis=0, keepdims=True), 6, axis=0))
    assert result.argmax(axis=1).tolist() == [3] * 6 + [4] * 6 + [0] * 4


def test_decoders_keep_out_of_order_and_repeated_steps() -> None:
    labels = [0] * 5 + [6] * 8 + [4] * 8 + [3] * 8 + [4] * 8 + [0] * 5
    probs = _onehot(labels, 0.8)
    edges = np.zeros(len(labels), dtype=np.float32)
    edges[[5, 13, 21, 29, 37]] = 0.95

    for raw_weight in (0.0, 0.5, 0.75):
        decoded = decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=raw_weight,
                                        protect_min_frames=3, protect_confidence=0.9)
        assert decoded.argmax(axis=1).tolist() == labels


def test_apply_decoder_reads_new_and_legacy_settings() -> None:
    probs = np.vstack([_onehot([1] * 5, 0.7), _onehot([2] * 5, 0.7)])
    edges = np.zeros(10, dtype=np.float32)
    edges[5] = 0.9
    legacy = {"threshold": 0.5, "min_gap": 3}  # 2026-10-19 checkpoints stored no method
    blend = {"method": "boundary_blend", "threshold": 0.5, "min_gap": 3, "raw_weight": 0.75,
             "protect_min_seconds": 0.6, "protect_confidence": 0.9, "sample_fps": 5.0}

    np.testing.assert_array_equal(apply_decoder(probs, None, {"method": "raw"}), probs)
    np.testing.assert_array_equal(apply_decoder(probs, edges, legacy),
                                  decode_boundaries(probs, edges, threshold=0.5, min_gap=3))
    np.testing.assert_array_equal(
        apply_decoder(probs, edges, blend),
        decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=0.75,
                              protect_min_frames=3, protect_confidence=0.9),
    )
    with pytest.raises(ValueError, match="needs boundary probabilities"):
        apply_decoder(probs, None, blend)
    with pytest.raises(ValueError, match="Unknown decoder"):
        apply_decoder(probs, edges, {"method": "viterbi"})


def test_invalid_blend_settings_are_rejected() -> None:
    probs = _onehot([1, 1, 2], 0.9)
    edges = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=1.5,
                              protect_min_frames=3, protect_confidence=0.9)
    with pytest.raises(ValueError):
        decode_boundary_blend(probs, edges, threshold=0.5, min_gap=3, raw_weight=0.5,
                              protect_min_frames=0, protect_confidence=0.9)
    with pytest.raises(ValueError):
        seconds_to_frames(0.0, 5.0)
