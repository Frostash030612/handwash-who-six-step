"""Project exceptions with diagnostic and remediation messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "BackendUnavailableError",
    "CheckpointError",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigKeyError",
    "DataError",
    "DataLeakageError",
    "DatasetNotFoundError",
    "EvaluationError",
    "HandwashError",
    "ManifestError",
    "ModelError",
    "ModelNotFoundError",
    "ProtocolError",
    "TrainingError",
    "VideoDecodeError",
]


class HandwashError(Exception):
    exit_code: int = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  → Suggestion: {self.hint}"
        return self.message


class ConfigError(HandwashError):
    exit_code = 2


class ConfigFileNotFoundError(ConfigError):
    def __init__(self, path: str | Path) -> None:
        super().__init__(
            f"Configuration file not found: {path}",
            hint="Check whether the path is relative to the project root and review the supplied configuration.",
        )
        self.path = Path(path)


class ConfigKeyError(ConfigError):
    def __init__(self, key: str, *, expected: str, got: Any, where: str = "config") -> None:
        super().__init__(
            f"{where} key in `{key}` invalid; expected {expected}; received {type(got).__name__}={got!r}",
            hint="Key spelling and nesting must match the dataclasses in core/config.py.",
        )
        self.key = key


class DataError(HandwashError):
    exit_code = 3


class DatasetNotFoundError(DataError):
    def __init__(self, name: str, expected_at: str | Path | None = None) -> None:
        where = f"; expected location: {expected_at}" if expected_at else ""
        super().__init__(
            f"Dataset `{name}` does not exist{where}",
            hint="Set HANDWASH_DATA_ROOT to the prepared dataset location.",
        )
        self.name = name


class VideoDecodeError(DataError):
    def __init__(self, path: str | Path, reason: str) -> None:
        super().__init__(
            f"Video decoding failed: {path} ({reason})",
            hint="Check file integrity and codec compatibility; install opencv-python or imageio-ffmpeg if needed.",
        )
        self.path = Path(path)


class ManifestError(DataError):
    pass


class DataLeakageError(DataError):
    def __init__(self, clip_ids: list[str]) -> None:
        preview = ", ".join(sorted(clip_ids)[:5])
        more = f" including {len(clip_ids)} clips" if len(clip_ids) > 5 else ""
        super().__init__(
            f"Data leakage: the following original videos occur in multiple splits: {preview}{more}",
            hint="Split train/val/test by original video (split.group_key=original_video).",
        )
        self.clip_ids = clip_ids


class ModelError(HandwashError):
    exit_code = 4


class ModelNotFoundError(ModelError):
    def __init__(self, name: str, available: list[str] | None = None) -> None:
        avail = f"; registered: {', '.join(sorted(available))}" if available else ""
        super().__init__(
            f"Model `{name}` is not registered{avail}",
            hint="New model implementations must use the register_model decorator.",
        )
        self.name = name


class CheckpointError(ModelError):
    pass


class BackendUnavailableError(ModelError):
    def __init__(self, backend: str, install_hint: str) -> None:
        super().__init__(f"Backend `{backend}` is unavailable", hint=install_hint)
        self.backend = backend


class TrainingError(HandwashError):
    exit_code = 5


class EvaluationError(HandwashError):
    exit_code = 5


class ProtocolError(HandwashError):
    exit_code = 6
