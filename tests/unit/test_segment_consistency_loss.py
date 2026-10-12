"""Safety checks for the optional within-action training regularizer."""

import torch
from scripts.train_boundary_yolo_gru import _segment_consistency_loss


def test_constant_predictions_have_zero_loss_even_across_true_boundary() -> None:
    logits = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]])
    targets = torch.tensor([[0, 0, 1, 1]])
    assert torch.isclose(_segment_consistency_loss(logits, targets), torch.tensor(0.0))


def test_isolated_flicker_is_penalized_only_inside_known_run() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]])
    targets = torch.tensor([[0, 0, 0, -1]])
    assert _segment_consistency_loss(logits, targets) > 0


def test_short_true_action_is_not_removed() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]])
    targets = torch.tensor([[0, 1, 0]])
    assert torch.isclose(_segment_consistency_loss(logits, targets), torch.tensor(0.0))
