"""Staged boundary and residual-refinement experiments on a frozen GRU.

All temporal convolutions are causal. Whole-sequence ``forward`` is supported;
streaming convolution-state caching is deliberately not part of this research API.
Existing GRU and boundary-GRU checkpoint interfaces are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from handwash.errors import ModelError
from handwash.pipelines.exp_demo import ExpTemporalHead

STRUCTURE_FORMAT = "self-scene-structural-gru/1"


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.activation(self.conv(functional.pad(x, (self.left_padding, 0)))))


class TemporalStack(nn.Module):
    def __init__(self, input_dim: int, channels: int, dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(CausalResidualBlock(channels, d, dropout) for d in dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.projection(x.transpose(1, 2))).transpose(1, 2)


class ActionBoundaryBranch(nn.Module):
    """MLP control or a 15-sample causal temporal receptive field (dilations 1/2/4)."""

    def __init__(self, hidden: int, kind: str, channels: int, dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.network = nn.Sequential(nn.Linear(hidden * 2, channels), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(channels, 1))
        elif kind == "multiscale":
            self.network = nn.Sequential(TemporalStack(hidden * 2, channels, dilations, dropout),
                                         nn.Linear(channels, 1))
        else:
            raise ModelError(f"boundary.kind expected mlp/multiscale, received {kind!r}")

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((encoded[:, :1], encoded[:, :-1]), dim=1)
        inputs = torch.cat((encoded, (encoded - previous).abs()), dim=-1)
        return self.network(inputs).squeeze(-1)


class ActionRefiner(nn.Module):
    """Equal-parameter boundary/no-boundary controls with zero-initialized residual logits."""

    def __init__(self, hidden: int, classes: int, channels: int, dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.network = TemporalStack(hidden + classes + 1, channels, dilations, dropout)
        self.output = nn.Linear(channels, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, encoded: torch.Tensor, logits: torch.Tensor, edges: torch.Tensor,
                *, use_boundary: bool) -> torch.Tensor:
        signal = edges.sigmoid() if use_boundary else torch.zeros_like(edges)
        x = torch.cat((encoded, logits.log_softmax(-1), signal.unsqueeze(-1)), dim=-1)
        return logits + self.output(self.network(x))


def within_action_consistency(logits: torch.Tensor, labels: torch.Tensor, *, truncate: float) -> torch.Tensor:
    """Truncated log-probability MSE only across adjacent known, same-class labels.

    No supervision is imposed across class changes or ignored transition samples.
    The previous probability is detached, following the one-sided MS-TCN loss.
    """
    mask = (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0) & (labels[:, 1:] == labels[:, :-1])
    if not bool(mask.any()):
        return logits.sum() * 0.0
    logp = logits.log_softmax(-1)
    delta = (logp[:, 1:] - logp[:, :-1].detach()).abs().clamp(max=truncate)
    return delta.square().mean(-1)[mask].mean()


class StructuralBoundaryGRU(nn.Module):
    def __init__(self, backbone_hparams: dict[str, Any], boundary_kind: str,
                 channels: int, dilations: list[int], dropout: float,
                 refinement: bool = False, use_boundary: bool = False) -> None:
        super().__init__()
        self.hparams = {
            "backbone_hparams": backbone_hparams, "boundary_kind": boundary_kind,
            "channels": channels, "dilations": list(dilations), "dropout": dropout,
            "refinement": refinement, "use_boundary": use_boundary,
        }
        self.backbone = ExpTemporalHead(**backbone_hparams)
        hidden = int(backbone_hparams["hidden"])
        classes = int(backbone_hparams["num_classes"])
        self.boundary = ActionBoundaryBranch(hidden, boundary_kind, channels, dilations, dropout)
        self.refiner = ActionRefiner(hidden, classes, channels, dilations, dropout) if refinement else None
        self.use_boundary = use_boundary
        self.set_training_stage("refiner" if refinement else "boundary")

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"boundary", "refiner"}:
            raise ModelError(f"training stage expected boundary/refiner, received {stage!r}")
        self.stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        module = self.boundary if stage == "boundary" else self.refiner
        if module is None:
            raise ModelError("training stage refiner requires a configured refinement module")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True) -> StructuralBoundaryGRU:
        super().train(mode)
        self.backbone.eval()
        if self.stage != "boundary":
            self.boundary.eval()
        if self.refiner is not None and self.stage != "refiner":
            self.refiner.eval()
        return self

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        x = (features - self.backbone.feature_mean) / self.backbone.feature_std
        encoded, _ = self.backbone.rnn(self.backbone.encoder(x))
        return encoded, self.backbone.classifier(encoded)

    def forward_encoded(self, encoded: torch.Tensor, base_logits: torch.Tensor,
                        *, ablate_boundary: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        edges = self.boundary(encoded)
        if self.refiner is None:
            return base_logits, edges
        return self.refiner(encoded, base_logits, edges,
                            use_boundary=self.use_boundary and not ablate_boundary), edges

    def forward(self, features: torch.Tensor, *, ablate_boundary: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, logits = self.encode(features)
        return self.forward_encoded(encoded, logits, ablate_boundary=ablate_boundary)
