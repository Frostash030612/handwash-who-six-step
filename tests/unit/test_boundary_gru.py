from __future__ import annotations

import numpy as np
import torch

from handwash.pipelines.boundary_gru import (
    BoundaryAwareGRU,
    boundary_targets_from_consensus,
    decode_boundaries,
    score_boundaries,
    weak_boundary_intervals_from_consensus,
    weak_interval_loss,
)


def test_boundary_labels_only_use_two_adjacent_consensus_centres() -> None:
    labels = np.asarray([0, 1, 1, -1, 2, 3], dtype=np.int64)

    target, mask = boundary_targets_from_consensus(labels)

    assert target.tolist() == [0, 1, 0, 0, 0, 1]
    assert mask.tolist() == [False, True, True, False, False, True]


def test_weak_intervals_include_short_unknown_spans_but_not_long_ones() -> None:
    labels = np.asarray([1, -1, 2, 2, -1, 2, 3, -1, -1, -1, -1, 4])

    assert weak_boundary_intervals_from_consensus(labels, max_gap=3) == [(1, 3)]


def test_weak_interval_loss_requires_one_peak_not_every_frame() -> None:
    no_peak = torch.full((1, 4), -5.0)
    one_peak = no_peak.clone()
    one_peak[0, 2] = 5.0

    assert weak_interval_loss(one_peak, [(0, 1, 3)]) < weak_interval_loss(
        no_peak, [(0, 1, 3)]
    )


def test_boundary_gru_has_frame_and_boundary_outputs() -> None:
    model = BoundaryAwareGRU(feature_dim=4, hidden=8, dropout=0.0)
    class_logits, boundary_logits, state = model(torch.randn(2, 9, 4))

    assert class_logits.shape == (2, 9, 7)
    assert boundary_logits.shape == (2, 9)
    assert state.shape == (1, 2, 8)


def test_bidirectional_boundary_gru_uses_two_recurrent_directions() -> None:
    model = BoundaryAwareGRU(feature_dim=4, hidden=8, dropout=0.0, bidirectional=True)
    class_logits, boundary_logits, state = model(torch.randn(2, 9, 4))

    assert class_logits.shape == (2, 9, 7)
    assert boundary_logits.shape == (2, 9)
    assert state.shape == (2, 2, 8)


def test_detached_boundary_loss_cannot_change_the_classification_branch() -> None:
    torch.manual_seed(0)
    features = torch.randn(1, 12, 4)
    for detach, expect_gradient in ((False, True), (True, False)):
        model = BoundaryAwareGRU(feature_dim=4, hidden=8, dropout=0.0, detach_boundary_input=detach)
        _class_logits, boundary_logits, _state = model(features)
        boundary_logits.sum().backward()

        branch = [model.rnn.weight_ih_l0.grad, model.encoder[1].weight.grad]
        assert all((grad is not None and grad.abs().sum() > 0) == expect_gradient for grad in branch)
        assert model.boundary_classifier[0].weight.grad.abs().sum() > 0


def test_old_boundary_checkpoints_without_detach_flag_still_build() -> None:
    old_hparams = {"feature_dim": 4, "hidden": 8, "num_classes": 7, "dropout": 0.3, "bidirectional": False}

    model = BoundaryAwareGRU(**old_hparams)

    assert model.hparams["detach_boundary_input"] is False


def test_boundary_decoder_averages_only_inside_detected_segments() -> None:
    probabilities = np.asarray([
        [0.9, 0.1], [0.6, 0.4], [0.1, 0.9], [0.2, 0.8],
    ], dtype=np.float32)
    boundary = np.asarray([0.0, 0.1, 0.95, 0.1], dtype=np.float32)

    result = decode_boundaries(probabilities, boundary, threshold=0.8, min_gap=2)

    np.testing.assert_allclose(result[0], result[1])
    np.testing.assert_allclose(result[2], result[3])
    assert result[:2, 0].mean() > result[:2, 1].mean()
    assert result[2:, 1].mean() > result[2:, 0].mean()


def test_boundary_metrics_ignore_unannotated_pairs() -> None:
    result = score_boundaries(
        np.asarray([0.1, 0.9, 0.9, 0.2]),
        np.asarray([False, True, False, False]),
        np.asarray([True, True, False, True]),
        threshold=0.5,
    )

    assert result["valid_centres"] == 3
    assert result["true_boundaries"] == 1
    assert result["predicted_boundaries"] == 1
    assert result["precision"] == result["recall"] == result["f1"] == 1.0
