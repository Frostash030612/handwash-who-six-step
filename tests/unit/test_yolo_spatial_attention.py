"""Small invariants for the experimental YOLO spatial-attention adapter."""

import numpy as np
import torch
from scripts.compare_boundary_viterbi_validation import decode_viterbi

from handwash.pipelines.yolo_spatial_attention import SpatialAttentionPool


def test_zero_initialized_attention_reproduces_global_average() -> None:
    features = torch.randn(3, 5, 4, 6)
    pool = SpatialAttentionPool(channels=5)
    assert torch.allclose(pool(features), features.mean(dim=(-2, -1), keepdim=True), atol=1e-6)


def test_viterbi_does_not_force_step_order() -> None:
    probabilities = np.full((6, 7), 0.005, dtype=np.float32)
    probabilities[:3, 2] = 0.97
    probabilities[3:, 1] = 0.97
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    edges = np.full(6, 0.01, dtype=np.float32)
    edges[3] = 0.99
    decoded = decode_viterbi(probabilities, edges, edge_weight=1.0, change_penalty=1.0)
    assert decoded.tolist() == [2, 2, 2, 1, 1, 1]


def test_viterbi_suppresses_unsupported_one_frame_change() -> None:
    probabilities = np.full((5, 7), 0.001, dtype=np.float32)
    probabilities[:, 1] = 0.9
    probabilities[2, 1] = 0.2
    probabilities[2, 2] = 0.75
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    edges = np.full(5, 0.01, dtype=np.float32)
    decoded = decode_viterbi(probabilities, edges, edge_weight=0.5, change_penalty=2.0)
    assert decoded.tolist() == [1, 1, 1, 1, 1]
