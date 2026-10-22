"""Boundary-aware temporal head for the frozen YOLO hand-washing backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Decoders live in the torch-free core layer; re-exported so existing imports keep working.
from handwash.core.boundary_decoding import (  # noqa: F401
    DECODER_METHODS,
    apply_decoder,
    decode_boundaries,
    decode_boundary_blend,
)
from handwash.errors import DataError, ModelError
from handwash.pipelines.exp_demo import EXP_CLASS_NAMES

BOUNDARY_HEAD_FORMAT = "yolo-gru-boundary/1"


def boundary_targets_from_consensus(consensus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create boundary targets only for adjacent, annotated consensus labels."""
    labels = np.asarray(consensus, dtype=np.int64)
    targets = np.zeros(len(labels), dtype=np.float32)
    mask = np.zeros(len(labels), dtype=bool)
    if len(labels) > 1:
        valid = (labels[:-1] >= 0) & (labels[1:] >= 0)
        mask[1:] = valid
        targets[1:] = (valid & (labels[1:] != labels[:-1])).astype(np.float32)
    return targets, mask


def weak_boundary_intervals_from_consensus(
    consensus: np.ndarray, *, max_gap: int = 3
) -> list[tuple[int, int]]:
    """Return half-open intervals containing a change across a short unknown span.

    Two agreed labels that differ imply at least one change between them. When
    one to ``max_gap`` intermediate samples lack consensus, its precise position
    is unknown, so the whole interval is supervised only as a bag of candidates.
    Longer gaps are ignored because several changes could have occurred.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be positive")
    labels = np.asarray(consensus, dtype=np.int64)
    valid = np.flatnonzero(labels >= 0)
    intervals: list[tuple[int, int]] = []
    for previous, current in zip(valid[:-1], valid[1:], strict=False):
        gap = int(current - previous - 1)
        if 1 <= gap <= max_gap and labels[current] != labels[previous]:
            intervals.append((int(previous + 1), int(current + 1)))
    return intervals


def weak_interval_loss(edge_logits: torch.Tensor, intervals: list[tuple[int, int, int]]) -> torch.Tensor:
    """Noisy-OR MIL loss: at least one boundary inside each uncertain interval."""
    if not intervals:
        return edge_logits.sum() * 0.0
    losses = []
    for row, left, right in intervals:
        log_none = nn.functional.logsigmoid(-edge_logits[row, left:right]).sum()
        probability = (-torch.expm1(log_none)).clamp_min(1e-8)
        losses.append(-torch.log(probability))
    return torch.stack(losses).mean()


class BoundaryAwareGRU(nn.Module):
    """GRU action classifier with an auxiliary action-boundary head.

    The feature normalization, encoder, GRU, and action classifier intentionally
    keep the same parameter names as :class:`ExpTemporalHead`, so a trained
    YOLO+GRU checkpoint can initialize this model without replacing it.

    With ``detach_boundary_input`` the boundary head still reads the GRU
    features, but its loss cannot change the encoder, GRU, or classifier.
    """

    arch_name = "boundary_gru"

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden: int = 128,
        num_classes: int = len(EXP_CLASS_NAMES),
        dropout: float = 0.3,
        bidirectional: bool = False,
        detach_boundary_input: bool = False,
    ) -> None:
        super().__init__()
        self.hparams = {
            "feature_dim": int(feature_dim),
            "hidden": int(hidden),
            "num_classes": int(num_classes),
            "dropout": float(dropout),
            "bidirectional": bool(bidirectional),
            "detach_boundary_input": bool(detach_boundary_input),
        }
        self.detach_boundary_input = bool(detach_boundary_input)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=bidirectional)
        temporal_dim = hidden * (2 if bidirectional else 1)
        self.classifier = nn.Linear(temporal_dim, num_classes)
        self.boundary_classifier = nn.Sequential(
            nn.Linear(temporal_dim * 2, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action logits, boundary logits, and the recurrent state."""
        normalized = (features - self.feature_mean) / self.feature_std
        encoded, state = self.rnn(self.encoder(normalized), state)
        source = encoded.detach() if self.detach_boundary_input else encoded
        previous = torch.cat((source[:, :1], source[:, :-1]), dim=1)
        change = torch.abs(source - previous)
        boundary = self.boundary_classifier(torch.cat((source, change), dim=-1)).squeeze(-1)
        return self.classifier(encoded), boundary, state


def score_boundaries(
    scores: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Score a fixed boundary threshold on dual-annotator consensus pairs."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(targets, dtype=bool).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if len(values) != len(truth) or len(values) != len(valid):
        raise ValueError("Boundary scores, targets, and mask must have equal lengths")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    truth = truth[valid]
    predicted = values[valid] >= threshold
    true_positive = int(np.count_nonzero(truth & predicted))
    false_positive = int(np.count_nonzero(~truth & predicted))
    false_negative = int(np.count_nonzero(truth & ~predicted))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "valid_centres": int(valid.sum()),
        "true_boundaries": int(truth.sum()),
        "predicted_boundaries": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def load_boundary_gru(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[BoundaryAwareGRU, dict[str, Any]]:
    """Load and validate a boundary-GRU checkpoint."""
    source = Path(path)
    if not source.is_file():
        raise DataError(f"Boundary-aware GRU weights not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:  # pragma: no cover - backend-specific deserialization errors
        raise ModelError(f"Cannot read boundary-aware GRU weights: {source}", hint=str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("format") != BOUNDARY_HEAD_FORMAT:
        raise ModelError(f"Not a boundary-aware GRU checkpoint: {source}")
    if tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES:
        raise ModelError("The boundary-aware GRU class order differs from the seven-class protocol")
    try:
        head = BoundaryAwareGRU(**payload["hparams"])
        head.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise ModelError(
            f"Incompatible boundary-aware GRU checkpoint structure: {source}", hint=str(exc)
        ) from exc
    head.eval()
    return head, payload
