"""Boundary F1: annotated transition intervals, tolerance, and one-to-one matching."""

from __future__ import annotations

import pytest

from handwash.core.boundary_metrics import (
    BoundaryScore,
    PredictedBoundary,
    ReferenceBoundary,
    match_boundaries,
    predicted_boundaries,
    reference_boundaries,
    score_boundaries,
)

TIMES = [0.2 * index for index in range(12)]


def test_reference_interval_spans_unlabelled_transition_samples() -> None:
    labels = [0, 0, 0, -100, -100, 1, 1, 2, 2, -100, 2, 2]

    boundaries = reference_boundaries(labels, TIMES)

    assert [(b.before, b.after) for b in boundaries] == [(0, 1), (1, 2)]
    assert boundaries[0].start_s == pytest.approx(0.4) and boundaries[0].end_s == pytest.approx(1.0)
    assert boundaries[1].start_s == pytest.approx(1.2) and boundaries[1].end_s == pytest.approx(1.4)


def test_predicted_change_sits_between_the_two_samples() -> None:
    boundaries = predicted_boundaries([0, 0, 1, 1, 1, 4], TIMES[:6])

    assert [(b.time_s, b.before, b.after, b.index) for b in boundaries] == [
        (pytest.approx(0.3), 0, 1, 2), (pytest.approx(0.9), 1, 4, 5)]


def test_inside_interval_has_zero_error_and_tolerance_bounds_matches() -> None:
    reference = [ReferenceBoundary(1.0, 1.4, 1, 2)]

    def single(time_s: float) -> BoundaryScore:
        return score_boundaries(reference, [PredictedBoundary(time_s, 1, 2, 0)], tolerance_s=0.4)[0]

    assert single(1.2).signed_errors_s == [0.0]
    assert single(1.7).signed_errors_s == [pytest.approx(0.3)]   # late, inside tolerance
    assert single(0.7).signed_errors_s == [pytest.approx(-0.3)]  # early, inside tolerance
    late = single(1.9)
    assert (late.true_positive, late.false_positive, late.false_negative) == (0, 1, 1)


def test_each_reference_matches_at_most_one_prediction() -> None:
    reference = [ReferenceBoundary(1.0, 1.2, 3, 4)]
    predicted = [PredictedBoundary(1.05, 3, 4, 0), PredictedBoundary(1.15, 4, 3, 1)]

    score, matches = score_boundaries(reference, predicted, tolerance_s=0.4)

    assert len(matches) == 1
    assert (score.true_positive, score.false_positive, score.false_negative) == (1, 1, 0)


def test_matching_maximises_matches_where_greedy_nearest_would_not() -> None:
    reference = [ReferenceBoundary(1.0, 1.2, 1, 2), ReferenceBoundary(1.8, 2.0, 2, 3)]
    # 1.5 is 0.3 s from both intervals; giving it to the second would orphan 2.3.
    predicted = [PredictedBoundary(1.5, 1, 2, 0), PredictedBoundary(2.3, 2, 3, 1)]

    matches = match_boundaries(reference, predicted, tolerance_s=0.4)

    assert [(m.reference, m.predicted) for m in matches] == [(0, 0), (1, 1)]


def test_pooled_scores_and_empty_videos() -> None:
    pooled = BoundaryScore()
    pooled.add(BoundaryScore(true_positive=3, false_positive=1, false_negative=0, signed_errors_s=[0.0, 0.2, -0.1]))
    pooled.add(BoundaryScore(true_positive=1, false_positive=0, false_negative=2, signed_errors_s=[0.3]))

    assert pooled.f1 == pytest.approx(8 / 11)
    assert pooled.mean_abs_error_s == pytest.approx(0.15)
    assert pooled.mean_signed_error_s == pytest.approx(0.1)
    assert BoundaryScore().f1 == 1.0 and BoundaryScore().mean_abs_error_s is None
    assert BoundaryScore(false_negative=2).f1 == 0.0


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        reference_boundaries([0, 1], [0.0])
    with pytest.raises(ValueError):
        predicted_boundaries([0, 1, 1], [0.0, 0.2, 0.2])
    with pytest.raises(ValueError):
        match_boundaries([], [], tolerance_s=-0.1)
    with pytest.raises(ValueError):
        match_boundaries([], [PredictedBoundary(1.0, 0, 1, 1), PredictedBoundary(0.5, 1, 0, 0)], tolerance_s=0.4)
