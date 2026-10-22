from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from handwash.core.config import ResolvedConfig
from handwash.core.labels import STEP_ORDER, LabelSpace, Step
from handwash.errors import BackendUnavailableError, DataError, ModelError
from handwash.paths import ensure_dir
from handwash.pipelines.common import resolve_device

__all__ = [
    "DEFAULT_TEMPORAL_HEAD",
    "EXP_CLASS_NAMES",
    "ExpDemoClassifier",
    "ExpFeatureExtractor",
    "ExpTemporalHead",
    "file_sha256",
    "load_temporal_head",
    "save_temporal_head",
]


EXP_CLASS_NAMES = (
    "0_other_or_nonstandard",
    "1_palm_to_palm",
    "2_palm_over_dorsum",
    "3_palm_to_palm_interlaced",
    "4_backs_of_fingers",
    "5_thumb_rubbing",
    "6_fingertips_to_palm",
)
_HEAD_FORMAT = "exp-temporal-head/1"

DEFAULT_TEMPORAL_HEAD = "exp_temporal_head.pt"


def file_sha256(path: str | Path) -> str:

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise DataError(f"Camera frames must be HWC RGB uint8; received {image.shape} / {image.dtype}")

    return np.ascontiguousarray(image[:, :, ::-1])


class ExpFeatureExtractor:
    def __init__(self, checkpoint: str | Path, *, device: str, cache_dir: str | Path) -> None:

        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise BackendUnavailableError("ultralytics", "This application requires ultralytics.") from exc

        try:
            model = YOLO(str(checkpoint))
        except Exception as exc:
            raise ModelError(f"Unable to load Ultralytics weights: {checkpoint}", hint=str(exc)) from exc
        if model.task != "classify":
            raise ModelError(f"exp.pt requires a classification model; received task {model.task!r}")
        names = tuple(str(model.names.get(index, "")) for index in range(len(model.names)))
        if names != EXP_CLASS_NAMES:
            raise ModelError(
                f"The exp.pt class order differs from the verified model: {names}",
                hint="Use the verified exp.pt weights for this entry point.",
            )
        try:
            self.image_size = int(model.overrides.get("imgsz", 320))
        except (TypeError, ValueError) as exc:
            raise ModelError("exp.pt does not record a valid input size") from exc
        if self.image_size <= 0:
            raise ModelError("exp.pt does not record a valid input size")
        self.model = model
        self.device = device
        self.checkpoint = Path(checkpoint)

        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.model.predict(source=blank, imgsz=self.image_size, device=self.device, verbose=False)
        try:
            linear = self.model.predictor.model.model.model[-1].linear
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError(
                "Cannot locate the exp.pt classification layer",
                hint="The Ultralytics version is incompatible with this adapter.",
            ) from exc
        if not isinstance(linear, nn.Linear):
            raise ModelError(
                "The exp.pt classification head has no Linear layer",
                hint="The Ultralytics version is incompatible with this adapter.",
            )
        self.feature_dim = int(linear.in_features)
        self._captured: list[torch.Tensor] = []
        linear.register_forward_hook(
            lambda _module, inputs, _output: self._captured.append(inputs[0].detach())
        )

    def forward_bgr(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:

        if not frames:
            raise DataError("At least one frame is required")
        self._captured.clear()
        results = self.model.predict(
            source=list(frames),
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if any(result.probs is None for result in results):
            raise ModelError("exp.pt did not return classification probabilities")
        probabilities = np.stack([result.probs.data.detach().cpu().numpy() for result in results]).astype(
            np.float32
        )
        features = torch.cat(self._captured).float().cpu().numpy() if self._captured else np.empty((0,))
        if probabilities.shape != (len(frames), len(EXP_CLASS_NAMES)) or not np.isfinite(probabilities).all():
            raise ModelError("exp.pt returned invalid classification probabilities")
        if features.shape != (len(frames), self.feature_dim) or not np.isfinite(features).all():
            raise ModelError(
                f"Unexpected exp.pt feature shape: {features.shape}; expected ({len(frames)}, {self.feature_dim})"
            )
        return features, probabilities


class ExpTemporalHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1280,
        hidden: int = 128,
        num_classes: int = len(EXP_CLASS_NAMES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hparams = {
            "feature_dim": int(feature_dim),
            "hidden": int(hidden),
            "num_classes": int(num_classes),
            "dropout": float(dropout),
        }

        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``features[B, T, D]`` -> ``(logits[B, T, C], state)``."""
        normalized = (features - self.feature_mean) / self.feature_std
        output, state = self.rnn(self.encoder(normalized), state)
        return self.classifier(output), state


def save_temporal_head(
    path: str | Path,
    head: ExpTemporalHead,
    *,
    exp_sha256: str,
    fps: float,
    metrics: dict[str, Any],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "format": _HEAD_FORMAT,
            "class_names": list(EXP_CLASS_NAMES),
            "exp_sha256": exp_sha256,
            "fps": float(fps),
            "hparams": dict(head.hparams),
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "metrics": metrics,
        },
        target,
    )
    return target


def load_temporal_head(path: str | Path, *, exp_checkpoint: str | Path, fps: float) -> ExpTemporalHead:

    source = Path(path)
    if not source.is_file():
        raise DataError(
            f"Temporal-head weights not found: {source}", hint="Train a compatible temporal head first."
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelError(f"Cannot read temporal-head weights: {source}", hint=str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("format") != _HEAD_FORMAT:
        raise ModelError(f"Not an exp.pt temporal-head checkpoint: {source}")
    if tuple(payload.get("class_names", ())) != EXP_CLASS_NAMES:
        raise ModelError("The temporal-head class order differs from exp.pt")
    if payload.get("exp_sha256") != file_sha256(exp_checkpoint):
        raise ModelError(
            "The temporal head was not trained with the current exp.pt",
            hint="The temporal head must be retrained if the YOLO feature extractor is replaced.",
        )
    if abs(float(payload.get("fps", 0.0)) - float(fps)) > 1e-6:
        raise ModelError(
            f"Temporal-head training sample rate {payload.get('fps')} fps versus current configuration {fps} fps mismatch",
            hint="The GRU learns temporal context at fixed frame intervals; keep dataset.prep.fps consistent with training.",
        )
    head = ExpTemporalHead(**payload["hparams"])
    head.load_state_dict(payload["state_dict"])
    return head.eval()


class _TemporalStream:
    def __init__(self, classifier: ExpDemoClassifier) -> None:
        self._classifier = classifier
        self._state: torch.Tensor | None = None

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        features, _ = self._classifier.extractor.forward_bgr([_rgb_to_bgr(frame)])
        head = self._classifier.temporal_head
        assert head is not None
        logits, state = head(torch.from_numpy(features)[None], self._state)
        probabilities = torch.softmax(logits[0, -1], dim=-1).numpy().astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise ModelError("The temporal head returned invalid classification probabilities")
        self._state = state
        return probabilities


class ExpDemoClassifier:
    def __init__(
        self,
        rc: ResolvedConfig,
        checkpoint: str | Path,
        *,
        temporal_head: str | Path | None = None,
    ) -> None:
        self.rc = rc
        self.space = LabelSpace("exp_demo", (Step.OTHER, *STEP_ORDER))
        self.device = str(resolve_device(rc.runtime.device))
        cache_dir = ensure_dir(rc.resolve_out_dir() / "demo_runtime")
        self.extractor = ExpFeatureExtractor(checkpoint, device=self.device, cache_dir=cache_dir)
        self.image_size = self.extractor.image_size
        self.temporal_head = (
            None
            if temporal_head is None
            else load_temporal_head(temporal_head, exp_checkpoint=checkpoint, fps=rc.dataset.prep.fps)
        )
        if (
            self.temporal_head is not None
            and self.temporal_head.hparams["feature_dim"] != self.extractor.feature_dim
        ):
            raise ModelError("The temporal-head input dimension differs from the exp.pt feature dimension")

        self.model_name = (
            "exp.pt + GRU temporal head" if self.temporal_head is not None else "exp.pt (frame-only)"
        )

    def predict(self, frame: np.ndarray) -> np.ndarray:

        _, probabilities = self.extractor.forward_bgr([_rgb_to_bgr(frame)])
        return probabilities[0]

    def new_stream(self) -> _TemporalStream | None:

        return None if self.temporal_head is None else _TemporalStream(self)
