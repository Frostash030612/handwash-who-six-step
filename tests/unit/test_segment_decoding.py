"""Behavioral and exhaustive optimality checks for segment decisions."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from handwash.core.segment_decoding import SegmentDecoderConfig, decode_action_segments


def objective(labels, probs, edges, cfg):
    total = sum(np.log(max(probs[i, c], 1e-6)) for i, c in enumerate(labels))
    cuts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
    for start, end in zip(cuts[:-1], cuts[1:], strict=True):
        if start:
            total -= cfg.change_penalty + cfg.boundary_weight * min(
                -np.log(max(edges[start], 1e-6)), cfg.boundary_cost_cap
            )
        if start and end < len(labels) and labels[start] != cfg.background_class:
            total -= cfg.short_action_penalty_per_second * max(
                0, cfg.preferred_action_seconds - (end - start) / cfg.sample_fps
            )
    return total


def test_exact_solution_matches_exhaustive_sequences():
    rng = np.random.default_rng(11)
    cfg = SegmentDecoderConfig(sample_fps=2, preferred_action_seconds=2, change_penalty=0.4)
    for _ in range(4):
        probs = rng.dirichlet(np.ones(3), size=6)
        edges = rng.uniform(0.1, 0.9, size=6)
        result = decode_action_segments(probs, edges, cfg)
        expected = max(objective(y, probs, edges, cfg) for y in product(range(3), repeat=6))
        assert result.objective == pytest.approx(expected)
        assert objective(result.labels, probs, edges, cfg) == pytest.approx(expected)


def test_order_repetition_and_short_background_survive():
    truth = np.repeat([0, 3, 1, 4, 1, 0], [2, 20, 20, 20, 20, 2])
    probs = np.full((len(truth), 7), 0.001)
    probs[np.arange(len(truth)), truth] = 0.994
    edges = np.full(len(truth), 0.01)
    edges[np.r_[False, truth[1:] != truth[:-1]]] = 0.9
    result = decode_action_segments(probs, edges, SegmentDecoderConfig())
    np.testing.assert_array_equal(result.labels, truth)


def test_short_high_confidence_island_can_be_suppressed_but_true_short_action_can_survive():
    probs = np.tile([0.005, 0.99, 0.005], (65, 1))
    probs[30:35] = [0.005, 0.015, 0.98]
    cfg = SegmentDecoderConfig(change_penalty=6, short_action_penalty_per_second=4)
    edges = np.full(65, 0.01)
    result = decode_action_segments(probs, edges, cfg)
    assert np.all(result.labels == 1)
    # Strong class evidence plus two real edge peaks can retain a one-second action.
    probs[30:35] = [0.000001, 0.000001, 0.999998]
    edges[[30, 35]] = 0.99
    retained = decode_action_segments(probs, edges, cfg)
    assert np.all(retained.labels[30:35] == 2)
    assert retained.segments[1]["short_observed_action"]


def test_false_edge_alone_never_rewards_a_cut():
    probs = np.tile([0.01, 0.98, 0.01], (30, 1))
    result = decode_action_segments(probs, np.ones(30), SegmentDecoderConfig())
    assert len(result.segments) == 1
    assert result.segments[0]["mean_raw_class_probability"] == pytest.approx(0.98)


def test_zero_boundary_weight_is_independent_of_edges():
    rng = np.random.default_rng(4)
    probs = rng.dirichlet(np.ones(3), size=20)
    cfg = replace(SegmentDecoderConfig(), boundary_weight=0)
    a = decode_action_segments(probs, None, cfg)
    b = decode_action_segments(probs, np.ones(20), cfg)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.objective == b.objective


def test_empty_and_invalid_input():
    result = decode_action_segments(np.empty((0, 7)), np.empty(0), SegmentDecoderConfig())
    assert result.labels.size == 0
    with pytest.raises(ValueError):
        decode_action_segments([[float("nan"), 1]], [0.5], SegmentDecoderConfig())
    with pytest.raises(ValueError):
        decode_action_segments([[0.1, 0.9]], None, SegmentDecoderConfig())
