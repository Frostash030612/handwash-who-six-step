"""Configuration dataclasses, YAML parsing, overrides and validation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from handwash.core.labels import CANONICAL_STEPS, Step, get_label_space
from handwash.errors import ConfigError, ConfigFileNotFoundError
from handwash.paths import (
    CONFIGS_DIR,
    DATA_PROCESSED_DIRNAME,
    DATA_RAW_DIRNAME,
    PROJECT_ROOT,
    SYNTHETIC_DIRNAME,
    ensure_dir,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "AppConfig",
    "ResolvedConfig",
    "apply_overrides",
    "dump_config",
    "hash_config",
    "load_config",
    "merge_mappings",
    "parse_overrides",
]


CONFIG_SCHEMA_VERSION: Final[int] = 2

T = TypeVar("T")


_TRACKING_BACKENDS: Final[tuple[str, ...]] = ("none", "csv")
_ULTRALYTICS_ARCHES: Final[frozenset[str]] = frozenset(
    {"yolo26n-cls", "yolo26m-cls", "yolon-cls", "yolov8n-cls"}
)
_TORCHVISION_ARCHES: Final[frozenset[str]] = frozenset({"mobilenet_v2", "resnet18", "efficientnet_b0"})
_PRECISIONS: Final[tuple[str, ...]] = ("fp32", "fp16", "bf16")
_IMAGE_SIZES: Final[tuple[int, ...]] = (64, 96, 128, 160, 192, 224, 256, 288, 320)
_NORMALIZE_MODES: Final[tuple[str, ...]] = ("imagenet", "zero_one", "minus_one_one")
_TEMPORAL_KINDS: Final[tuple[str, ...]] = ("gru", "tcn", "mean_pool", "none")
_RUN_MODES: Final[tuple[str, ...]] = ("frame", "clip", "hybrid")
_DURATION_UNITS: Final[tuple[str, ...]] = ("seconds", "ratio", "none")
_HORIZONS: Final[tuple[str, ...]] = ("frames", "seconds")
_GROUP_KEYS: Final[tuple[str, ...]] = ("original_video", "clip_id", "participant")


# ============================================================================


# ============================================================================
def merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:

    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_mappings(result[key], value)
        else:
            result[key] = value
    return result


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:

    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"Invalid override format: {item!r}",
                hint="Use key.sub=value, for example train.epochs=5",
            )
        dotted, _, raw = item.partition("=")
        keys = [k.strip() for k in dotted.strip().split(".") if k.strip()]
        if not keys:
            raise ConfigError(f"Empty override key: {item!r}")
        if raw.strip() == "":
            raise ConfigError(
                f"Missing override value: {item!r}",
                hint=(
                    "A value is required after the equals sign. Empty YAML values become null. "
                    "Invalid overrides are rejected before type validation. "
                    "Use key=null for an explicit null value."
                ),
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Cannot parse override value: {item!r} ({exc})") from exc

        cursor = result
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = value
    return result


def apply_overrides(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:

    return merge_mappings(data, overrides)


def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        return " | ".join(_type_name(a) for a in get_args(tp) if a is not type(None)) + " | null"
    if origin in (list, tuple, Sequence):
        args = get_args(tp)
        return f"list[{_type_name(args[0])}]" if args else "list"
    if origin in (dict, Mapping):
        return "dict"
    return getattr(tp, "__name__", str(tp))


def _coerce(value: Any, tp: Any, *, key: str, where: str) -> Any:

    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[X]
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if len(non_none) != len(args):
                return None
            raise ConfigError(f"{where} key in `{key}` must not be null")
        last_error: Exception | None = None
        for candidate in non_none:
            try:
                return _coerce(value, candidate, key=key, where=where)
            except (ConfigError, TypeError, ValueError) as exc:
                last_error = exc
        raise ConfigError(
            f"{where} key in `{key}` type mismatch; expected {_type_name(tp)}; received {value!r}",
            hint=str(last_error) if last_error else None,
        )

    # Literal[...]
    if origin is not None and str(origin).endswith("Literal"):
        allowed = tuple(args)
        if value not in allowed:
            raise ConfigError(
                f"{where} key in `{key}` has an invalid value: {value!r}",
                hint=f"Allowed values: {list(allowed)}",
            )
        return value

    # list / tuple
    if origin in (list, tuple, Sequence):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
            raise ConfigError(
                f"{where} key in `{key}` must be a list; received {type(value).__name__}={value!r}",
                hint="Use a YAML list with separate `- a` lines or `[a, b]`.",
            )
        item_tp = args[0] if args else Any
        seq = [_coerce(v, item_tp, key=f"{key}[]", where=where) for v in value]
        return tuple(seq) if origin is tuple else seq

    # Mapping
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} key in `{key}` must be a mapping; received {type(value).__name__}")
        return dict(value)

    # Any / object
    if tp is Any or tp is object:
        return value

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no"):
            return value.strip().lower() in ("true", "yes")
        raise ConfigError(f"{where} key in `{key}` must be a boolean; received {value!r}")

    if tp in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{where} key in `{key}` must be {tp.__name__}; received {value!r}")
        if tp is int and isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
            raise ConfigError(f"{where} key in `{key}` must be an integer; received {value!r}")
        try:
            return int(value) if tp is int else float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigError(f"{where} key in `{key}` must be {tp.__name__}; received {value!r}") from exc

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(
                f"{where} key in `{key}` must be a string (quote string values); received {value!r}"
            )
        return value

    return value


class _ConfigBase:
    @classmethod
    def from_mapping(cls: type[T], data: Mapping[str, Any] | None, *, where: str | None = None) -> T:
        where = where or cls.__name__
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{where} must be a mapping; received {type(data).__name__}")

        hints = get_type_hints(cls)
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"{where} contains unknown configuration keys: {unknown}",
                hint=(
                    "Keys must match dataclass fields in core/config.py exactly; unknown keys are rejected."
                ),
            )

        kwargs: dict[str, Any] = {}
        for f in fields(cls):  # type: ignore[arg-type]
            if f.name not in data:
                continue
            tp = hints[f.name]
            if is_dataclass(tp) and isinstance(tp, type) and issubclass(tp, _ConfigBase):
                kwargs[f.name] = tp.from_mapping(data[f.name], where=f"{where}.{f.name}")
            else:
                kwargs[f.name] = _coerce(data[f.name], tp, key=f.name, where=where)
        try:
            return cls(**kwargs)  # type: ignore[return-value]
        except TypeError as exc:
            raise ConfigError(f"{where} construction failed: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))  # type: ignore[call-overload]

    def validate(self) -> None:

        return None


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_plain(dataclasses.asdict(value))
    return value


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class ProjectConfig(_ConfigBase):
    name: str = "handwash"
    task: str = "who_six_step_recognition"
    language: str = "zh"


@dataclass(frozen=True, slots=True)
class RuntimeConfig(_ConfigBase):
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"  # auto | cpu | cuda | cuda:0 ...
    num_workers: int = 4
    pin_memory: bool = True
    log_level: str = "INFO"
    tracking: str = "csv"  # none | csv
    tracking_project: str = "handwash"
    run_name: str | None = None

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("runtime.seed must be a non-negative integer")
        if self.tracking not in _TRACKING_BACKENDS:
            raise ConfigError(
                f"runtime.tracking has an invalid value: {self.tracking!r}",
                hint=f"Allowed: {list(_TRACKING_BACKENDS)}",
            )
        if self.num_workers < 0:
            raise ConfigError("runtime.num_workers must not be negative")
        if not self.tracking_project.strip():
            raise ConfigError("runtime.tracking_project must not be empty")
        if self.run_name is not None and (
            not self.run_name.strip()
            or "/" in self.run_name
            or "\\" in self.run_name
            or Path(self.run_name).name != self.run_name
            or self.run_name in (".", "..")
        ):
            raise ConfigError("runtime.run_name must be a single directory name without path separators")


@dataclass(frozen=True, slots=True)
class PathsConfig(_ConfigBase):
    out_dir: str = "outputs"
    models_dir: str = "models"
    cache_dir: str = ".cache/handwash"

    def validate(self) -> None:
        for name in ("out_dir", "models_dir", "cache_dir"):
            value = getattr(self, name)
            if not value.strip():
                raise ConfigError(f"paths.{name} must not be empty")

    def resolve(self, path: str, *, root: Path | None = None) -> Path:
        base = PROJECT_ROOT if root is None else root
        p = Path(path).expanduser()
        return p if p.is_absolute() else base / p


@dataclass(frozen=True, slots=True)
class SplitConfig(_ConfigBase):
    group_key: str = "original_video"
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    stratify_by: str | None = "label_sequence"
    seed: int = 42
    max_frames_per_clip_train: int | None = 200
    max_frames_per_clip_eval: int | None = 100
    guard_leakage: bool = True

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.train, self.val, self.test)):
            raise ConfigError("Split ratios must be finite")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"Split ratios must sum to 1.0; received {total:.6f} (train+val+test)",
                hint="For example, train=0.7, val=0.15, test=0.15.",
            )
        if min(self.train, self.val, self.test) < 0:
            raise ConfigError("Split ratios must not be negative")
        if self.seed < 0:
            raise ConfigError("split.seed must be a non-negative integer")
        if self.group_key not in _GROUP_KEYS:
            raise ConfigError(
                f"split.group_key has an invalid value: {self.group_key!r}",
                hint=f"Allowed: {list(_GROUP_KEYS)}",
            )
        for name, cap in (
            ("max_frames_per_clip_train", self.max_frames_per_clip_train),
            ("max_frames_per_clip_eval", self.max_frames_per_clip_eval),
        ):
            if cap is not None and cap <= 0:
                raise ConfigError(f"split.{name} must be a positive integer or null")


@dataclass(frozen=True, slots=True)
class DataPrepConfig(_ConfigBase):
    fps: float = 5.0
    frame_step: int = 1
    resize_hw: tuple[int, int] = (224, 224)  # (H, W)
    image_ext: str = "jpg"
    jpeg_quality: int = 85
    min_frames_per_clip: int = 10

    def validate(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ConfigError("data.prep.fps must be positive")
        if self.frame_step < 1:
            raise ConfigError("data.prep.frame_step must be >= 1")
        if self.min_frames_per_clip < 1:
            raise ConfigError("data.prep.min_frames_per_clip must be >= 1")
        if len(self.resize_hw) != 2:
            raise ConfigError("data.prep.resize_hw must contain exactly height and width")
        h, w = self.resize_hw
        if h <= 0 or w <= 0:
            raise ConfigError(f"data.prep.resize_hw invalid: {self.resize_hw}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("data.prep.jpeg_quality must be in 1..100")
        if self.image_ext not in ("jpg", "jpeg", "png"):
            raise ConfigError("data.prep.image_ext supports only jpg / jpeg / png")


@dataclass(frozen=True, slots=True)
class DataConfig(_ConfigBase):
    name: str = "pskuss"
    root: str = "data/raw/pskuss/extracted"
    variants: tuple[str, ...] = ()
    label_space: str | None = None
    include_non_wash: bool = True
    prep: DataPrepConfig = field(default_factory=DataPrepConfig)

    def __post_init__(self) -> None:
        if self.label_space is None:
            object.__setattr__(self, "label_space", self.name)

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigError("dataset.name must not be empty")
        if not self.root.strip():
            raise ConfigError("dataset.root must not be empty")


@dataclass(frozen=True, slots=True)
class ModelConfig(_ConfigBase):
    name: str = "yolo26n-cls"
    arch: str = "yolo26n-cls"
    pretrained: str | bool = "auto"
    num_classes: int | None = None
    image_size: int = 224
    normalize: str = "zero_one"
    dropout: float = 0.2
    temporal: TemporalConfig = field(default_factory=lambda: TemporalConfig())

    def validate(self) -> None:
        if isinstance(self.pretrained, str) and not self.pretrained.strip():
            raise ConfigError("model.pretrained must not be an empty string")
        if not self.arch.strip():
            raise ConfigError("model.arch must not be empty")
        if self.image_size not in _IMAGE_SIZES:
            raise ConfigError(
                f"model.image_size has an invalid value: {self.image_size}",
                hint=f"Allowed: {list(_IMAGE_SIZES)}(YOLO input sizes should be multiples of 32).",
            )
        if self.normalize not in _NORMALIZE_MODES:
            raise ConfigError(
                f"model.normalize has an invalid value: {self.normalize!r}",
                hint=f"Allowed: {list(_NORMALIZE_MODES)}",
            )
        arch = self.arch.strip().lower()
        if "yolo" in arch and self.normalize != "zero_one":
            raise ConfigError(
                "Ultralytics classification models require model.normalize=zero_one",
                hint="Inputs must be pixels in [0, 1]; do not apply ImageNet mean/std normalization again.",
            )
        pretrained_enabled = self.pretrained is True or (
            isinstance(self.pretrained, str)
            and self.pretrained.strip().lower() in {"auto", "true", "imagenet"}
        )
        if arch in _TORCHVISION_ARCHES and pretrained_enabled and self.normalize != "imagenet":
            raise ConfigError(
                f"Using ImageNet-pretrained weights for {arch} requires model.normalize=imagenet",
                hint="For other normalization schemes, explicitly set model.pretrained=false and document the experiment.",
            )
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.num_classes is not None and self.num_classes < 1:
            raise ConfigError("model.num_classes must be a positive integer or null")


@dataclass(frozen=True, slots=True)
class TemporalConfig(_ConfigBase):
    kind: str = "none"  # gru | tcn | mean_pool | none
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    window: int = 16
    stride: int = 8
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1

    def validate(self) -> None:
        if self.kind not in _TEMPORAL_KINDS:
            raise ConfigError(
                f"model.temporal.kind has an invalid value: {self.kind!r}",
                hint=f"Allowed: {list(_TEMPORAL_KINDS)}",
            )
        if self.window < 1 or not 1 <= self.stride <= self.window:
            raise ConfigError("model.temporal.stride must be in 1..window")
        if self.kind != "none" and self.window < 2:
            raise ConfigError("Temporal models require model.temporal.window >= 2")
        if self.num_layers < 1:
            raise ConfigError("model.temporal.num_layers must be >= 1")
        if self.hidden_size < 1 or self.kernel_size < 1:
            raise ConfigError("model.temporal.hidden_size / kernel_size must be a positive integer")
        if any(dilation < 1 for dilation in self.dilations):
            raise ConfigError("model.temporal.dilations values must be positive integers")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.temporal.dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainConfig(_ConfigBase):
    mode: str = "frame"  # frame | clip; hybrid is inference-only
    epochs: int = 20
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 5e-4
    optimizer: str = "adamw"  # adamw | sgd
    momentum: float = 0.9
    scheduler: str = "cosine"  # cosine | step | none
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.05
    class_weights: str = "none"  # none | balanced
    early_stopping_patience: int = 5
    grad_clip_norm: float = 1.0
    precision: str = "fp32"  # fp32 | fp16 | bf16
    accumulate_grad_batches: int = 1
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig())
    focal_gamma: float = 0.0

    def validate(self) -> None:
        if self.mode not in ("frame", "clip"):
            raise ConfigError(
                f"train.mode has an invalid or unsupported value: {self.mode!r}",
                hint="Training supports frame or clip mode; hybrid combines inference outputs and is not joint training.",
            )
        if self.precision not in _PRECISIONS:
            raise ConfigError(
                f"train.precision has an invalid value: {self.precision!r}",
                hint=f"Allowed: {list(_PRECISIONS)}",
            )
        if self.epochs < 1:
            raise ConfigError("train.epochs must be >= 1")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ConfigError("train.batch_size / eval_batch_size must be >= 1")
        if self.lr <= 0:
            raise ConfigError("train.lr must be positive")
        if not math.isfinite(self.lr) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError("train.lr / weight_decay must be finite, with non-negative weight_decay")
        if self.early_stopping_patience < 1:
            raise ConfigError("train.early_stopping_patience must be >= 1")
        if self.accumulate_grad_batches < 1:
            raise ConfigError("train.accumulate_grad_batches must be >= 1")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm < 0:
            raise ConfigError("train.grad_clip_norm must be finite and non-negative")
        if not math.isfinite(self.warmup_epochs) or self.warmup_epochs < 0:
            raise ConfigError("train.warmup_epochs must be finite and non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError("train.label_smoothing must be in [0, 1)")
        if not 0.0 <= self.focal_gamma <= 5.0:
            raise ConfigError("train.focal_gamma must be in [0, 5]")
        if self.optimizer not in ("adamw", "sgd"):
            raise ConfigError(
                f"train.optimizer has an invalid value: {self.optimizer!r}", hint="Allowed: adamw / sgd"
            )
        if not math.isfinite(self.momentum) or not 0.0 < self.momentum <= 1.0:
            raise ConfigError("train.momentum must be in (0, 1]")
        if self.scheduler not in ("cosine", "step", "none"):
            raise ConfigError(
                f"train.scheduler has an invalid value: {self.scheduler!r}",
                hint="Allowed: cosine / step / none",
            )
        if self.class_weights not in ("none", "balanced"):
            raise ConfigError("train.class_weights supports only none / balanced")


@dataclass(frozen=True, slots=True)
class AugmentConfig(_ConfigBase):
    random_resized_crop: bool = True
    crop_scale: tuple[float, float] = (0.7, 1.0)
    horizontal_flip: bool = True
    color_jitter: float = 0.2
    rotation_deg: float = 8.0
    gaussian_blur: float = 0.1
    randaugment: bool = False

    def validate(self) -> None:
        lo, hi = self.crop_scale
        if not 0.0 < lo <= hi <= 1.0:
            raise ConfigError(
                f"train.augment.crop_scale invalid: {self.crop_scale}", hint="must satisfy 0<a<=b<=1"
            )
        for key, value in (
            ("color_jitter", self.color_jitter),
            ("rotation_deg", self.rotation_deg),
            ("gaussian_blur", self.gaussian_blur),
        ):
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"train.augment.{key} must be finite and non-negative")
        if self.randaugment:
            raise ConfigError(
                "train.augment.randaugment is not implemented",
                hint="Disable randaugment; supported augmentations are random crop, horizontal flip, rotation, color jitter and blur.",
            )


@dataclass(frozen=True, slots=True)
class EvalConfig(_ConfigBase):
    splits: tuple[str, ...] = ("val", "test")
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix")
    save_confusion_matrix: bool = True
    save_predictions: bool = True
    bootstrap_ci: bool = False
    bootstrap_samples: int = 1000
    extra_datasets: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.splits:
            raise ConfigError("eval.splits must not be empty")
        allowed_splits = {"train", "val", "test", "external"}
        if any(split not in allowed_splits for split in self.splits):
            raise ConfigError(f"eval.splits may use only {sorted(allowed_splits)}")
        if len(set(self.splits)) != len(self.splits):
            raise ConfigError("eval.splits must not contain duplicate values")
        if len(set(self.extra_datasets)) != len(self.extra_datasets):
            raise ConfigError("eval.extra_datasets must not contain duplicate values")
        if any(not name.strip() for name in self.extra_datasets):
            raise ConfigError("eval.extra_datasets must not contain empty names")
        if self.bootstrap_samples < 1:
            raise ConfigError("eval.bootstrap_samples must be >= 1")
        allowed_metrics = {"accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"}
        if any(metric not in allowed_metrics for metric in self.metrics):
            raise ConfigError(f"eval.metrics may contain only {sorted(allowed_metrics)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ConfigError("eval.metrics must not contain duplicate values")


@dataclass(frozen=True, slots=True)
class AssessConfig(_ConfigBase):
    smooth_window: int = 9
    min_confidence: float = 0.4

    min_segment_frames: int = 5
    min_segment_s: float = 1.0
    min_total_duration_s: float = 40.0
    reference_total_duration_s: float = 50.0
    min_step_duration_s: float = 3.0
    step_duration_ratio: float = 0.4
    allow_repeats: bool = False
    missing_tolerance: int = 0
    duration_check: str = "ratio"  # seconds | ratio | none
    order_check: bool = True
    action_chain_method: str = "adaptive_boundary_gru"
    action_chain_smoothing_seconds: float = 1.0
    action_chain_outlier_z: float = -2.0
    action_chain_min_comparable_segments: int = 4
    require_faucet_events: bool = False
    report_language: str = "zh"

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("assess.min_confidence must be in [0, 1]")
        if self.smooth_window < 1:
            raise ConfigError("assess.smooth_window must be >= 1")
        if self.min_segment_frames < 1:
            raise ConfigError("assess.min_segment_frames must be >= 1")
        if not math.isfinite(self.min_segment_s) or self.min_segment_s < 0:
            raise ConfigError("assess.min_segment_s must be finite and non-negative")
        if (
            not math.isfinite(self.min_total_duration_s)
            or not math.isfinite(self.reference_total_duration_s)
            or not math.isfinite(self.min_step_duration_s)
            or self.min_total_duration_s < 0
            or self.reference_total_duration_s <= 0
            or self.min_step_duration_s < 0
        ):
            raise ConfigError(
                "Assessment duration thresholds must be finite; the reference duration must be positive"
            )
        if not 0.0 <= self.step_duration_ratio <= 1.0:
            raise ConfigError("assess.step_duration_ratio must be in [0, 1]")
        if self.duration_check not in _DURATION_UNITS:
            raise ConfigError(
                f"assess.duration_check has an invalid value: {self.duration_check!r}",
                hint=f"Allowed: {list(_DURATION_UNITS)}",
            )
        if self.action_chain_method not in ("legacy", "adaptive_boundary_gru"):
            raise ConfigError("assess.action_chain_method must be legacy or adaptive_boundary_gru")
        if not math.isfinite(self.action_chain_smoothing_seconds) or self.action_chain_smoothing_seconds < 0:
            raise ConfigError("assess.action_chain_smoothing_seconds must be finite and non-negative")
        if not math.isfinite(self.action_chain_outlier_z) or self.action_chain_outlier_z >= 0:
            raise ConfigError("assess.action_chain_outlier_z must be finite and negative")
        if self.action_chain_min_comparable_segments < 3:
            raise ConfigError("assess.action_chain_min_comparable_segments must be at least 3")
        if self.missing_tolerance < 0:
            raise ConfigError("assess.missing_tolerance must not be negative")
        if self.report_language != "zh":
            raise ConfigError("Legacy assessment reports support only the zh locale")


@dataclass(frozen=True, slots=True)
class InferConfig(_ConfigBase):
    mode: str = "frame"  # frame | clip | hybrid
    temporal_apply: bool = False
    smooth_window: int = 9
    save_frame_predictions: bool = True
    save_overlay_video: bool = False
    batch_size: int = 64
    tta: bool = False

    def validate(self) -> None:
        if self.mode not in _RUN_MODES:
            raise ConfigError(
                f"infer.mode has an invalid value: {self.mode!r}", hint=f"Allowed: {list(_RUN_MODES)}"
            )
        if self.smooth_window < 1:
            raise ConfigError("infer.smooth_window must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("infer.batch_size must be >= 1")


def _default_dataset_profiles() -> dict[str, dict[str, Any]]:

    names = ("kaggle", "pskuss", "metc", "jurmala", "hospitals_full", "selfrecorded", "synthetic")
    profiles: dict[str, dict[str, Any]] = {}
    for name in names:
        root = f"{DATA_RAW_DIRNAME}/{name}" if name != "synthetic" else SYNTHETIC_DIRNAME
        processed = f"{DATA_PROCESSED_DIRNAME}/{name}"
        profiles[name] = {
            "root": root,
            "processed_dir": processed,
            "frames_dir": f"{processed}/frames",
            "manifest": f"{processed}/manifest.csv",
        }
    profiles["pskuss"]["root"] = f"{DATA_RAW_DIRNAME}/pskuss/extracted"
    profiles["jurmala"]["root"] = f"{DATA_RAW_DIRNAME}/jurmala/extracted"
    profiles["hospitals_full"]["root"] = f"{DATA_PROCESSED_DIRNAME}/hospitals_full_7class"
    profiles["hospitals_full"]["processed_dir"] = f"{DATA_PROCESSED_DIRNAME}/hospitals_full_7class"
    profiles["hospitals_full"]["frames_dir"] = f"{DATA_PROCESSED_DIRNAME}/hospitals_full_7class/frames"
    profiles["hospitals_full"]["manifest"] = f"{DATA_PROCESSED_DIRNAME}/hospitals_full_7class/manifest.csv"
    profiles["metc"]["external_only"] = True
    return profiles


@dataclass(frozen=True, slots=True)
class AppConfig(_ConfigBase):
    schema_version: int = CONFIG_SCHEMA_VERSION
    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    datasets: Mapping[str, Mapping[str, Any]] = field(default_factory=_default_dataset_profiles)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    assess: AssessConfig = field(default_factory=AssessConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"Configuration schema_version={self.schema_version} versus the expected {CONFIG_SCHEMA_VERSION} mismatch",
                hint="Migrate the configuration fields for the expected schema version.",
            )
        for _name, section in (
            ("runtime", self.runtime),
            ("paths", self.paths),
            ("dataset", self.dataset),
            ("dataset.prep", self.dataset.prep),
            ("split", self.split),
            ("model", self.model),
            ("model.temporal", self.model.temporal),
            ("train", self.train),
            ("train.augment", self.train.augment),
            ("eval", self.eval),
            ("assess", self.assess),
            ("infer", self.infer),
        ):
            section.validate()
        if not self.datasets:
            raise ConfigError(
                "datasets section must not be empty",
                hint="At least one dataset profile is required.",
            )
        if self.dataset.name not in self.datasets:
            raise ConfigError(
                f"dataset.name='{self.dataset.name}' is not defined in datasets",
                hint=f"Defined: {sorted(self.datasets)}",
            )
        source_space = get_label_space(self.dataset.label_space or self.dataset.name)
        if self.model.temporal.kind != "none" and self.train.mode != "clip":
            raise ConfigError(
                f"model.temporal.kind={self.model.temporal.kind!r} requires train.mode=clip",
                hint="Frame-mode training supplies isolated frames, so GRU/TCN/mean_pool cannot see temporal context.",
            )
        if self.model.temporal.kind != "none" and self.infer.mode == "frame":
            raise ConfigError(
                f"model.temporal.kind={self.model.temporal.kind!r} cannot use infer.mode=frame",
                hint="Temporal inference requires clip mode; use hybrid to combine frame and temporal outputs.",
            )
        label_space_has_auxiliary = any(label not in CANONICAL_STEPS for label in source_space.labels)
        if self.dataset.include_non_wash != label_space_has_auxiliary:
            raise ConfigError(
                "dataset.include_non_wash is inconsistent with dataset.label_space",
                hint=(
                    f"Label space {source_space.name!r} "
                    f"{'includes' if label_space_has_auxiliary else 'excludes'}non-six-step classes; "
                    f"include_non_wash must be {str(label_space_has_auxiliary).lower()}."
                ),
            )
        if self.assess.require_faucet_events and not {Step.FAUCET_ON, Step.FAUCET_OFF}.issubset(
            set(source_space.labels)
        ):
            raise ConfigError(
                "assess.require_faucet_events=true requires both faucet_on and faucet_off in the model label space",
                hint="This dataset lacks both faucet-event labels; disable the option or choose a compatible label space.",
            )
        if self.model.num_classes is not None and self.model.num_classes != len(source_space):
            raise ConfigError(
                f"model.num_classes={self.model.num_classes} versus label space "
                f"{source_space.name!r} class count {len(source_space)} mismatch",
                hint="Set model.num_classes to null or choose a matching label space.",
            )
        for target_name in self.eval.extra_datasets:
            if target_name not in self.datasets:
                raise ConfigError(
                    f"eval.extra_datasets in {target_name!r} is not defined in datasets",
                    hint=f"Defined: {sorted(self.datasets)}",
                )
            target_spec = self.datasets[target_name]
            get_label_space(str(target_spec.get("label_space") or target_name))

    def dataset_spec(self, name: str | None = None) -> Mapping[str, Any]:

        key = name or self.dataset.name
        if key not in self.datasets:
            raise ConfigError(f"Undefined dataset profile: {key!r}", hint=f"Defined: {sorted(self.datasets)}")
        return dict(self.datasets[key])

    def label_space_name(self) -> str:
        return self.dataset.label_space or self.dataset.name

    def resolve_out_dir(self, extra: str | None = None) -> Path:
        base = self.paths.resolve(self.paths.out_dir)
        if self.runtime.run_name:
            base = base / self.runtime.run_name
        if extra:
            base = base / extra
        return base

    def resolve_models_dir(self) -> Path:
        return self.paths.resolve(self.paths.models_dir)


# ============================================================================


# ============================================================================
@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    config: AppConfig
    sources: tuple[str, ...]
    overrides: Mapping[str, Any]
    config_hash: str
    out_dir: Path
    label_space: str
    weight_path: Path | None = None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.config, item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "config_hash": self.config_hash,
            "sources": list(self.sources),
            "overrides": dict(self.overrides),
            "label_space": self.label_space,
            "out_dir": str(self.out_dir),
            "weight_path": str(self.weight_path) if self.weight_path else None,
            "config": self.config.to_dict(),
        }


def hash_config(payload: Any) -> str:

    canonical = json.dumps(_to_plain(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dump_config(resolved: ResolvedConfig, path: str | Path) -> Path:

    target = Path(path)
    ensure_dir(target.parent)
    payload = resolved.to_dict()
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return target


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigFileNotFoundError(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration YAML parsing failed: {path} ({exc})") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"The top level of a configuration must be a mapping: {path}; received {type(loaded).__name__}"
        )
    return dict(loaded)


def load_config(
    paths: Sequence[str | Path] = (),
    *,
    overrides: Mapping[str, Any] | None = None,
    extra_paths: Sequence[str | Path] = (),
) -> ResolvedConfig:

    base_paths: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        base_paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    if not base_paths:
        base_paths = [CONFIGS_DIR / "config.yaml"]

    merged: dict[str, Any] = {}
    sources: list[str] = []
    for path in base_paths:
        merged = merge_mappings(merged, _read_yaml(path))
        sources.append(str(path))
    for item in extra_paths:
        path = Path(item).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
        merged = merge_mappings(merged, _read_yaml(path))
        sources.append(str(path))

    if overrides:
        merged = apply_overrides(merged, overrides)

    app = AppConfig.from_mapping(merged)
    app.validate()

    out_dir = app.resolve_out_dir()
    label_space = app.label_space_name()
    weight_path = _resolve_pretrained_path(app)

    return ResolvedConfig(
        config=app,
        sources=tuple(sources),
        overrides=dict(overrides or {}),
        config_hash=hash_config(merged),
        out_dir=out_dir,
        label_space=label_space,
        weight_path=weight_path,
    )


def _resolve_pretrained_path(app: AppConfig) -> Path | None:
    """Resolve explicit local weights independently of the caller's working directory."""
    raw = app.model.pretrained
    if not isinstance(raw, str):
        return None

    arch = app.model.arch.strip().lower()
    alias_arch = {"yolon-cls": "yolo11n-cls"}.get(arch, arch)
    official_name = f"{alias_arch}.pt"
    token = raw.strip().lower()
    if token in {"auto", "true", "imagenet"}:
        if arch in {"yolo26n-cls", "yolo26m-cls", "yolon-cls", "yolov8n-cls"}:
            local_official = app.resolve_models_dir() / official_name
            if local_official.is_file():
                return local_official.resolve()
        return None
    if token == "false":
        return None

    configured = Path(raw).expanduser()
    candidates = (
        (configured,)
        if configured.is_absolute()
        else (app.resolve_models_dir() / configured, app.paths.resolve(str(configured)))
    )
    for candidate in candidates:
        if candidate.is_file():
            if arch not in _ULTRALYTICS_ARCHES:
                raise ConfigError(
                    f"model.arch={arch!r} does not support initialization from local Ultralytics .pt weights",
                    hint="Local .pt initialization supports yolo26n-cls, yolo26m-cls, yolon-cls and yolov8n-cls only.",
                )
            return candidate.resolve()

    known_model_files = {f"{arch}.pt", f"{alias_arch}.pt"}
    if configured.name.lower() in known_model_files:
        # Ultralytics may download its official architecture weight by name.
        return None
    if configured.suffix.lower() in {".pt", ".pth", ".ckpt"} or configured.parent != Path():
        searched = ", ".join(str(path) for path in candidates)
        raise ConfigError(
            f"Cannot find the model.pretrained weight file: {raw!r}",
            hint=f"Checked: {searched}. Place the file in paths.models_dir or specify an absolute path.",
        )
    raise ConfigError(
        f"Unrecognized model.pretrained weight name: {raw!r}",
        hint=f"Use auto, true, false, imagenet or an existing local weight file; current arch={arch!r}.",
    )
