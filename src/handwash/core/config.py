"""配置系统：YAML -> 冻结 dataclass，严格校验，未知键直接报错。

设计目标（CONTRIBUTING.md R3 / R14）
------------------------------------
1. **单一来源**：一切可调参数都必须在这里声明，代码里不允许出现魔数。
2. **拼错就炸**：未知键、类型不符、取值越界一律抛 ``ConfigError``，
   而不是静默用默认值（这是小组项目最常见的"结果对不上"根因）。
3. **分层叠加**：``configs/config.yaml`` + ``configs/data/*.yaml`` +
   ``configs/models/*.yaml`` + CLI 覆盖（``key.sub=value``）逐层深合并。
4. **可追溯**：加载结果带 ``config_hash``，并原样落盘到输出目录。

本模块是 L1 契约层唯一允许读文件的例外（只读 YAML）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import json
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from handwash.errors import ConfigError, ConfigFileNotFoundError
from handwash.core.labels import CANONICAL_STEPS, Step, get_label_space
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
    "load_config",
    "dump_config",
    "hash_config",
    "merge_mappings",
    "apply_overrides",
    "parse_overrides",
]

#: 配置结构版本。破坏性改动（删字段/改语义）必须 +1，并同步 docs/CONFIG.md。
CONFIG_SCHEMA_VERSION: Final[int] = 2

T = TypeVar("T")

# --- 字段合法取值 -----------------------------------------------------------
_TRACKING_BACKENDS: Final[tuple[str, ...]] = ("none", "csv")
_ULTRALYTICS_ARCHES: Final[frozenset[str]] = frozenset(
    {"yolo26n-cls", "yolo26m-cls", "yolon-cls", "yolov8n-cls"}
)
_TORCHVISION_ARCHES: Final[frozenset[str]] = frozenset(
    {"mobilenet_v2", "resnet18", "efficientnet_b0"}
)
_PRECISIONS: Final[tuple[str, ...]] = ("fp32", "fp16", "bf16")
_IMAGE_SIZES: Final[tuple[int, ...]] = (64, 96, 128, 160, 192, 224, 256, 288, 320)
_NORMALIZE_MODES: Final[tuple[str, ...]] = ("imagenet", "zero_one", "minus_one_one")
_TEMPORAL_KINDS: Final[tuple[str, ...]] = ("gru", "tcn", "mean_pool", "none")
_RUN_MODES: Final[tuple[str, ...]] = ("frame", "clip", "hybrid")
_DURATION_UNITS: Final[tuple[str, ...]] = ("seconds", "ratio", "none")
_HORIZONS: Final[tuple[str, ...]] = ("frames", "seconds")
_GROUP_KEYS: Final[tuple[str, ...]] = ("original_video", "clip_id", "participant")


# ============================================================================
# 通用校验工具
# ============================================================================
def merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """递归深合并：``overlay`` 覆盖 ``base``，同键且双方都是 mapping 时继续下钻。"""
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_mappings(result[key], value)
        else:
            result[key] = value
    return result


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """把 CLI 的 ``["model.arch=gru", "train.epochs=5"]`` 解析成嵌套 dict。

    值用 YAML 标量规则解析，因此 ``null / true / 3 / 0.5 / [a,b]`` 都能正确还原类型。
    """
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"覆盖参数格式错误：{item!r}",
                hint="正确写法是 key.sub=value，例如 train.epochs=5",
            )
        dotted, _, raw = item.partition("=")
        keys = [k.strip() for k in dotted.strip().split(".") if k.strip()]
        if not keys:
            raise ConfigError(f"覆盖参数的键为空：{item!r}")
        if raw.strip() == "":
            raise ConfigError(
                f"覆盖参数缺少取值：{item!r}",
                hint=(
                    '等号右边不能为空。YAML 会把空值解析成 null，从而静默地把参数变成 null 或'
                    '在类型校验时才报错；这里直接拦截，报错位置离问题更近。'
                    '需要显式置空请写 key=null。'
                ),
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"覆盖参数的值无法解析：{item!r}（{exc}）") from exc

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
    """在已合并的配置 dict 上应用 CLI 覆盖（同样是深合并）。"""
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
    """把 YAML 标量强制成声明类型；不合法直接抛 ConfigError。"""
    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[X]
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if len(non_none) != len(args):
                return None
            raise ConfigError(f"{where} 中的键 `{key}` 不允许为 null")
        last_error: Exception | None = None
        for candidate in non_none:
            try:
                return _coerce(value, candidate, key=key, where=where)
            except (ConfigError, TypeError, ValueError) as exc:  # 试下一个候选类型
                last_error = exc
        raise ConfigError(
            f"{where} 中的键 `{key}` 类型不符：期望 {_type_name(tp)}，实际 {value!r}",
            hint=str(last_error) if last_error else None,
        )

    # Literal[...]
    if origin is not None and str(origin).endswith("Literal"):
        allowed = tuple(args)
        if value not in allowed:
            raise ConfigError(
                f"{where} 中的键 `{key}` 取值非法：{value!r}",
                hint=f"允许的取值：{list(allowed)}",
            )
        return value

    # list / tuple
    if origin in (list, tuple, Sequence):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
            raise ConfigError(
                f"{where} 中的键 `{key}` 应为列表，实际 {type(value).__name__}={value!r}",
                hint="YAML 里请写成 `- a` 的多行列表或 `[a, b]`。",
            )
        item_tp = args[0] if args else Any
        seq = [_coerce(v, item_tp, key=f"{key}[]", where=where) for v in value]
        return tuple(seq) if origin is tuple else seq

    # Mapping
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} 中的键 `{key}` 应为 mapping，实际 {type(value).__name__}")
        return dict(value)

    # Any / object
    if tp is Any or tp is object:
        return value

    # 标量
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no"):
            return value.strip().lower() in ("true", "yes")
        raise ConfigError(f"{where} 中的键 `{key}` 应为 bool，实际 {value!r}")

    if tp in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{where} 中的键 `{key}` 应为 {tp.__name__}，实际 {value!r}")
        if tp is int and isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise ConfigError(f"{where} 中的键 `{key}` 应为整数，实际 {value!r}")
        try:
            return int(value) if tp is int else float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigError(f"{where} 中的键 `{key}` 应为 {tp.__name__}，实际 {value!r}") from exc

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(
                f"{where} 中的键 `{key}` 应为字符串（需要字符串时请加引号），实际 {value!r}"
            )
        return value

    return value


class _ConfigBase:
    """所有配置 dataclass 的基类：严格构造 + 序列化 + 取值校验。"""

    @classmethod
    def from_mapping(cls: type[T], data: Mapping[str, Any] | None, *, where: str | None = None) -> T:
        where = where or cls.__name__
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{where} 应为 mapping，实际 {type(data).__name__}")

        hints = get_type_hints(cls)
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"{where} 中出现未知配置键：{unknown}",
                hint=(
                    "键名必须与 src/handwash/core/config.py 中的 dataclass 字段完全一致；"
                    "拼错的键不会被忽略。若确实需要新参数，请按 CONTRIBUTING.md R14 走 RFC。"
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
            raise ConfigError(f"{where} 构造失败：{exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))  # type: ignore[call-overload]

    def validate(self) -> None:
        """子类覆写：做跨字段的业务校验。默认什么都不做。"""
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
# 各配置段
# ============================================================================
@dataclass(frozen=True, slots=True)
class ProjectConfig(_ConfigBase):
    """项目标识：出现在输出目录名与日志里。"""

    name: str = "handwash"
    task: str = "who_six_step_recognition"
    language: str = "zh"


@dataclass(frozen=True, slots=True)
class RuntimeConfig(_ConfigBase):
    """运行时开关：随机性、设备、日志、追踪。"""

    seed: int = 42
    deterministic: bool = True
    device: str = "auto"  # auto | cpu | cuda | cuda:0 ...
    num_workers: int = 4
    pin_memory: bool = True
    log_level: str = "INFO"
    tracking: str = "csv"  # none | csv
    tracking_project: str = "handwash"
    run_name: str | None = None  # None -> 由时间戳自动生成

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigError("runtime.seed 必须为非负整数")
        if self.tracking not in _TRACKING_BACKENDS:
            raise ConfigError(
                f"runtime.tracking 取值非法：{self.tracking!r}", hint=f"允许：{list(_TRACKING_BACKENDS)}"
            )
        if self.num_workers < 0:
            raise ConfigError("runtime.num_workers 不能为负")
        if not self.tracking_project.strip():
            raise ConfigError("runtime.tracking_project 不能为空")
        if self.run_name is not None and (
            not self.run_name.strip()
            or "/" in self.run_name
            or "\\" in self.run_name
            or Path(self.run_name).name != self.run_name
            or self.run_name in (".", "..")
        ):
            raise ConfigError("runtime.run_name 必须是单个目录名，不能包含路径分隔符")


@dataclass(frozen=True, slots=True)
class PathsConfig(_ConfigBase):
    """路径。相对路径一律相对**仓库根目录**解析（避免"在我电脑上能跑"）。"""

    out_dir: str = "outputs"
    models_dir: str = "models"
    cache_dir: str = ".cache/handwash"

    def validate(self) -> None:
        for name in ("out_dir", "models_dir", "cache_dir"):
            value = getattr(self, name)
            if not value.strip():
                raise ConfigError(f"paths.{name} 不能为空")

    def resolve(self, path: str, *, root: Path | None = None) -> Path:
        base = PROJECT_ROOT if root is None else root
        p = Path(path).expanduser()
        return p if p.is_absolute() else base / p


@dataclass(frozen=True, slots=True)
class SplitConfig(_ConfigBase):
    """数据划分。

    ``group_key`` 是防数据泄漏的关键：必须以**原始视频**为单位划分，
    否则同一段视频的相邻帧会同时落进 train 和 test（见选题文档第五节）。
    """

    group_key: str = "original_video"
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    stratify_by: str | None = "label_sequence"  # None 表示不分层
    seed: int = 42
    max_frames_per_clip_train: int | None = 200
    max_frames_per_clip_eval: int | None = 100
    guard_leakage: bool = True  # 发现跨 split 的原始视频直接报错

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.train, self.val, self.test)):
            raise ConfigError("split 比例必须是有限数值")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"split 比例之和必须为 1.0，实际 {total:.6f}（train+val+test）",
                hint="例如 train=0.7, val=0.15, test=0.15。",
            )
        if min(self.train, self.val, self.test) < 0:
            raise ConfigError("split 比例不能为负")
        if self.seed < 0:
            raise ConfigError("split.seed 必须为非负整数")
        if self.group_key not in _GROUP_KEYS:
            raise ConfigError(
                f"split.group_key 取值非法：{self.group_key!r}", hint=f"允许：{list(_GROUP_KEYS)}"
            )
        for name, cap in (
            ("max_frames_per_clip_train", self.max_frames_per_clip_train),
            ("max_frames_per_clip_eval", self.max_frames_per_clip_eval),
        ):
            if cap is not None and cap <= 0:
                raise ConfigError(f"split.{name} 必须为正整数或 null")


@dataclass(frozen=True, slots=True)
class DataPrepConfig(_ConfigBase):
    """抽帧超参。**先在 CPU 上抽好的帧，再训练**，避免训练时解码成为瓶颈。"""

    fps: float = 5.0
    frame_step: int = 1
    resize_hw: tuple[int, int] = (224, 224)  # (H, W)
    image_ext: str = "jpg"
    jpeg_quality: int = 85
    min_frames_per_clip: int = 10

    def validate(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ConfigError("data.prep.fps 必须为正")
        if self.frame_step < 1:
            raise ConfigError("data.prep.frame_step 必须 >= 1")
        if self.min_frames_per_clip < 1:
            raise ConfigError("data.prep.min_frames_per_clip 必须 >= 1")
        if len(self.resize_hw) != 2:
            raise ConfigError("data.prep.resize_hw 必须恰好包含高度和宽度")
        h, w = self.resize_hw
        if h <= 0 or w <= 0:
            raise ConfigError(f"data.prep.resize_hw 非法：{self.resize_hw}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("data.prep.jpeg_quality 必须在 1..100")
        if self.image_ext not in ("jpg", "jpeg", "png"):
            raise ConfigError("data.prep.image_ext 只支持 jpg / jpeg / png")


@dataclass(frozen=True, slots=True)
class DataConfig(_ConfigBase):
    """数据集配置。``name`` 必须对应 core/labels.py 里的标签空间或 dataset 注册名。"""

    name: str = "pskuss"
    root: str = "data/raw/pskuss/extracted"
    variants: tuple[str, ...] = ()
    label_space: str | None = None  # None -> 与 name 相同
    include_non_wash: bool = True
    prep: DataPrepConfig = field(default_factory=DataPrepConfig)

    def __post_init__(self) -> None:
        if self.label_space is None:
            object.__setattr__(self, "label_space", self.name)

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigError("dataset.name 不能为空")
        if not self.root.strip():
            raise ConfigError("dataset.root 不能为空")


@dataclass(frozen=True, slots=True)
class ModelConfig(_ConfigBase):
    """模型与预处理配置。

    ``arch`` 取值必须已在 models 注册表中注册（见 src/handwash/models/__init__.py）。
    """

    name: str = "yolo26n-cls"
    arch: str = "yolo26n-cls"
    pretrained: str | bool = "auto"  # auto | True | False | 本地权重路径
    num_classes: int | None = None  # None -> 由标签空间推导
    image_size: int = 224
    normalize: str = "zero_one"
    dropout: float = 0.2
    temporal: "TemporalConfig" = field(default_factory=lambda: TemporalConfig())

    def validate(self) -> None:
        if isinstance(self.pretrained, str) and not self.pretrained.strip():
            raise ConfigError("model.pretrained 不能是空字符串")
        if not self.arch.strip():
            raise ConfigError("model.arch 不能为空")
        if self.image_size not in _IMAGE_SIZES:
            raise ConfigError(
                f"model.image_size 取值非法：{self.image_size}",
                hint=f"允许：{list(_IMAGE_SIZES)}（YOLO 系列对非 32 倍数尺寸支持不佳）。",
            )
        if self.normalize not in _NORMALIZE_MODES:
            raise ConfigError(
                f"model.normalize 取值非法：{self.normalize!r}", hint=f"允许：{list(_NORMALIZE_MODES)}"
            )
        arch = self.arch.strip().lower()
        if "yolo" in arch and self.normalize != "zero_one":
            raise ConfigError(
                "Ultralytics 分类模型要求 model.normalize=zero_one",
                hint="输入应为 [0, 1] 像素；不要再做 ImageNet 均值/方差标准化。",
            )
        pretrained_enabled = self.pretrained is True or (
            isinstance(self.pretrained, str)
            and self.pretrained.strip().lower() in {"auto", "true", "imagenet"}
        )
        if arch in _TORCHVISION_ARCHES and pretrained_enabled and self.normalize != "imagenet":
            raise ConfigError(
                f"使用 ImageNet 预训练权重的 {arch} 要求 model.normalize=imagenet",
                hint="如果要使用其他归一化方式，请显式设置 model.pretrained=false 并按实验方案记录。",
            )
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout 必须在 [0, 1) 区间")
        if self.num_classes is not None and self.num_classes < 1:
            raise ConfigError("model.num_classes 必须为正整数或 null")


@dataclass(frozen=True, slots=True)
class TemporalConfig(_ConfigBase):
    """时序模块配置：解决"单帧误判导致的步骤跳变"（研究问题 3）。"""

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
                f"model.temporal.kind 取值非法：{self.kind!r}", hint=f"允许：{list(_TEMPORAL_KINDS)}"
            )
        if self.window < 1 or not 1 <= self.stride <= self.window:
            raise ConfigError("model.temporal.stride 必须在 1..window 范围内")
        if self.kind != "none" and self.window < 2:
            raise ConfigError("启用时序模型时 model.temporal.window 至少为 2")
        if self.num_layers < 1:
            raise ConfigError("model.temporal.num_layers 必须 >= 1")
        if self.hidden_size < 1 or self.kernel_size < 1:
            raise ConfigError("model.temporal.hidden_size / kernel_size 必须为正整数")
        if any(dilation < 1 for dilation in self.dilations):
            raise ConfigError("model.temporal.dilations 中的值必须为正整数")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.temporal.dropout 必须在 [0, 1) 区间")


@dataclass(frozen=True, slots=True)
class TrainConfig(_ConfigBase):
    """训练超参。默认值刻意偏"小数据能跑通"，做正式实验时在 configs/experiments/ 覆盖。"""

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
    augment: "AugmentConfig" = field(default_factory=lambda: AugmentConfig())
    focal_gamma: float = 0.0  # 0 表示普通交叉熵

    def validate(self) -> None:
        if self.mode not in ("frame", "clip"):
            raise ConfigError(
                f"train.mode 取值非法或尚未支持：{self.mode!r}",
                hint="训练目前只支持 frame / clip；hybrid 是推理融合模式，不代表联合训练。",
            )
        if self.precision not in _PRECISIONS:
            raise ConfigError(
                f"train.precision 取值非法：{self.precision!r}", hint=f"允许：{list(_PRECISIONS)}"
            )
        if self.epochs < 1:
            raise ConfigError("train.epochs 必须 >= 1")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ConfigError("train.batch_size / eval_batch_size 必须 >= 1")
        if self.lr <= 0:
            raise ConfigError("train.lr 必须为正")
        if not math.isfinite(self.lr) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError("train.lr / weight_decay 必须为有限数值，且 weight_decay 不得为负")
        if self.early_stopping_patience < 1:
            raise ConfigError("train.early_stopping_patience 必须 >= 1")
        if self.accumulate_grad_batches < 1:
            raise ConfigError("train.accumulate_grad_batches 必须 >= 1")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm < 0:
            raise ConfigError("train.grad_clip_norm 必须是非负有限数值")
        if not math.isfinite(self.warmup_epochs) or self.warmup_epochs < 0:
            raise ConfigError("train.warmup_epochs 必须是非负有限数值")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError("train.label_smoothing 必须在 [0, 1)")
        if not 0.0 <= self.focal_gamma <= 5.0:
            raise ConfigError("train.focal_gamma 应在 [0, 5] 区间")
        if self.optimizer not in ("adamw", "sgd"):
            raise ConfigError(f"train.optimizer 取值非法：{self.optimizer!r}", hint="允许：adamw / sgd")
        if not math.isfinite(self.momentum) or not 0.0 < self.momentum <= 1.0:
            raise ConfigError("train.momentum 必须在 (0, 1] 区间")
        if self.scheduler not in ("cosine", "step", "none"):
            raise ConfigError(
                f"train.scheduler 取值非法：{self.scheduler!r}", hint="允许：cosine / step / none"
            )
        if self.class_weights not in ("none", "balanced"):
            raise ConfigError("train.class_weights 只支持 none / balanced")


@dataclass(frozen=True, slots=True)
class AugmentConfig(_ConfigBase):
    """训练期数据增强开关。"""

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
            raise ConfigError(f"train.augment.crop_scale 非法：{self.crop_scale}", hint="应为 0<a<=b<=1")
        for key, value in (
            ("color_jitter", self.color_jitter),
            ("rotation_deg", self.rotation_deg),
            ("gaussian_blur", self.gaussian_blur),
        ):
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"train.augment.{key} 必须为非负有限数值")
        if self.randaugment:
            raise ConfigError(
                "train.augment.randaugment 尚未实现",
                hint="关闭 randaugment；当前支持随机裁剪、水平翻转、旋转、色彩扰动和模糊。",
            )


@dataclass(frozen=True, slots=True)
class EvalConfig(_ConfigBase):
    """评估配置：内部测试 + 跨场景测试（研究问题 2）。"""

    splits: tuple[str, ...] = ("val", "test")
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix")
    save_confusion_matrix: bool = True
    save_predictions: bool = True
    bootstrap_ci: bool = False
    bootstrap_samples: int = 1000
    extra_datasets: tuple[str, ...] = ()  # 例如 ("metc",) 做跨场景对比

    def validate(self) -> None:
        if not self.splits:
            raise ConfigError("eval.splits 不能为空")
        allowed_splits = {"train", "val", "test", "external"}
        if any(split not in allowed_splits for split in self.splits):
            raise ConfigError(f"eval.splits 只能使用 {sorted(allowed_splits)}")
        if len(set(self.splits)) != len(self.splits):
            raise ConfigError("eval.splits 不能有重复值")
        if len(set(self.extra_datasets)) != len(self.extra_datasets):
            raise ConfigError("eval.extra_datasets 不能有重复值")
        if any(not name.strip() for name in self.extra_datasets):
            raise ConfigError("eval.extra_datasets 不能包含空名称")
        if self.bootstrap_samples < 1:
            raise ConfigError("eval.bootstrap_samples 必须 >= 1")
        allowed_metrics = {"accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"}
        if any(metric not in allowed_metrics for metric in self.metrics):
            raise ConfigError(f"eval.metrics 只能包含 {sorted(allowed_metrics)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ConfigError("eval.metrics 不能有重复值")


@dataclass(frozen=True, slots=True)
class AssessConfig(_ConfigBase):
    """WHO 完整性判定阈值。**所有阈值只在这里改**（CONTRIBUTING.md R14）。"""

    smooth_window: int = 9
    min_confidence: float = 0.4
    #: 连续多少帧（换算成秒不得低于 min_segment_s）才算一段有效动作
    min_segment_frames: int = 5
    min_segment_s: float = 1.0
    min_total_duration_s: float = 40.0  # WHO 建议完整流程 40—60 秒
    reference_total_duration_s: float = 50.0
    min_step_duration_s: float = 3.0
    step_duration_ratio: float = 0.4  # 单步相对"平均应得时长"的最低比例
    allow_repeats: bool = False
    missing_tolerance: int = 0  # 允许漏几步仍判"基本完整"
    duration_check: str = "ratio"  # seconds | ratio | none
    order_check: bool = True
    require_faucet_events: bool = False
    report_language: str = "zh"

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigError("assess.min_confidence 必须在 [0, 1]")
        if self.smooth_window < 1:
            raise ConfigError("assess.smooth_window 必须 >= 1")
        if self.min_segment_frames < 1:
            raise ConfigError("assess.min_segment_frames 必须 >= 1")
        if not math.isfinite(self.min_segment_s) or self.min_segment_s < 0:
            raise ConfigError("assess.min_segment_s 必须为非负有限数值")
        if (
            not math.isfinite(self.min_total_duration_s)
            or not math.isfinite(self.reference_total_duration_s)
            or not math.isfinite(self.min_step_duration_s)
            or self.min_total_duration_s < 0
            or self.reference_total_duration_s <= 0
            or self.min_step_duration_s < 0
        ):
            raise ConfigError("assess 时长阈值必须是有限数值；参考总时长必须大于 0")
        if not 0.0 <= self.step_duration_ratio <= 1.0:
            raise ConfigError("assess.step_duration_ratio 必须在 [0, 1]")
        if self.duration_check not in _DURATION_UNITS:
            raise ConfigError(
                f"assess.duration_check 取值非法：{self.duration_check!r}",
                hint=f"允许：{list(_DURATION_UNITS)}",
            )
        if self.missing_tolerance < 0:
            raise ConfigError("assess.missing_tolerance 不能为负")
        if self.report_language != "zh":
            raise ConfigError("assess.report_language 目前只支持 zh")


@dataclass(frozen=True, slots=True)
class InferConfig(_ConfigBase):
    """推理配置：单段视频 -> 逐帧预测 -> 时序平滑。"""

    mode: str = "frame"  # frame | clip | hybrid
    temporal_apply: bool = False
    smooth_window: int = 9
    save_frame_predictions: bool = True
    save_overlay_video: bool = False
    batch_size: int = 64
    tta: bool = False

    def validate(self) -> None:
        if self.mode not in _RUN_MODES:
            raise ConfigError(f"infer.mode 取值非法：{self.mode!r}", hint=f"允许：{list(_RUN_MODES)}")
        if self.smooth_window < 1:
            raise ConfigError("infer.smooth_window 必须 >= 1")
        if self.batch_size < 1:
            raise ConfigError("infer.batch_size 必须 >= 1")


def _default_dataset_profiles() -> dict[str, dict[str, Any]]:
    """默认数据集档案。

    存在的意义：保证"默认配置是自洽的"——``AppConfig().validate()`` 必须能通过。
    否则任何直接构造 ``AppConfig`` 的代码（测试、notebook、二次开发）都要先
    手工塞一个 datasets 段，很容易漏掉并得到难懂的报错。

    真实路径仍以 ``configs/config.yaml`` 的 ``datasets`` 段为准（它会整体覆盖本默认值）。
    """
    # 注意：本函数是"配置默认值"的唯一定义处，这里的相对路径字面量
    # 已在 scripts/check_structure.py 的 LITERAL_EXCEPTIONS 中登记豁免
    # （配置默认值天然就是路径字符串，走 paths 常量反而会绕圈）。
    names = ("kaggle", "pskuss", "metc", "jurmala", "selfrecorded", "synthetic")
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
    profiles["metc"]["external_only"] = True
    return profiles


@dataclass(frozen=True, slots=True)
class AppConfig(_ConfigBase):
    """顶层配置：与 configs/config.yaml 的键结构一一对应。"""

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
                f"配置 schema_version={self.schema_version} 与代码期望的 {CONFIG_SCHEMA_VERSION} 不一致",
                hint="请按 CHANGELOG.md 的迁移说明更新配置文件，不要手工改这个数字了事。",
            )
        for name, section in (
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
                "datasets 段不能为空",
                hint="至少要有一个数据集档案，例如 datasets.kaggle.root / datasets.pskuss.root。",
            )
        if self.dataset.name not in self.datasets:
            raise ConfigError(
                f"dataset.name='{self.dataset.name}' 未在 datasets 中定义",
                hint=f"已定义：{sorted(self.datasets)}",
            )
        source_space = get_label_space(self.dataset.label_space or self.dataset.name)
        if self.model.temporal.kind != "none" and self.train.mode != "clip":
            raise ConfigError(
                f"model.temporal.kind={self.model.temporal.kind!r} 需要 train.mode=clip",
                hint="逐帧训练会把每帧单独送入时序模块，GRU/TCN/mean_pool 将看不到连续上下文。",
            )
        if self.model.temporal.kind != "none" and self.infer.mode == "frame":
            raise ConfigError(
                f"model.temporal.kind={self.model.temporal.kind!r} 不能使用 infer.mode=frame",
                hint="时序模型推理至少要使用 clip；需要融合单帧和时序输出时使用 hybrid。",
            )
        label_space_has_auxiliary = any(label not in CANONICAL_STEPS for label in source_space.labels)
        if self.dataset.include_non_wash != label_space_has_auxiliary:
            raise ConfigError(
                "dataset.include_non_wash 与 dataset.label_space 不一致",
                hint=(
                    f"标签空间 {source_space.name!r} "
                    f"{'包含' if label_space_has_auxiliary else '不包含'}非 WHO 六步类别；"
                    f"include_non_wash 应设为 {str(label_space_has_auxiliary).lower()}。"
                ),
            )
        if self.assess.require_faucet_events and not {
            Step.FAUCET_ON, Step.FAUCET_OFF
        }.issubset(set(source_space.labels)):
            raise ConfigError(
                "assess.require_faucet_events=true 需要模型标签空间同时包含 faucet_on 和 faucet_off",
                hint="当前数据集不含完整水龙头事件标签；关闭该选项或改用支持这两类的标签空间。",
            )
        if self.model.num_classes is not None and self.model.num_classes != len(source_space):
            raise ConfigError(
                f"model.num_classes={self.model.num_classes} 与标签空间 "
                f"{source_space.name!r} 的类别数 {len(source_space)} 不一致",
                hint="将 model.num_classes 设为 null，或选择匹配的标签空间。",
            )
        for target_name in self.eval.extra_datasets:
            if target_name not in self.datasets:
                raise ConfigError(
                    f"eval.extra_datasets 中的 {target_name!r} 未在 datasets 中定义",
                    hint=f"已定义：{sorted(self.datasets)}",
                )
            target_spec = self.datasets[target_name]
            get_label_space(str(target_spec.get("label_space") or target_name))

    # --- 便捷派生属性 -----------------------------------------------------
    def dataset_spec(self, name: str | None = None) -> Mapping[str, Any]:
        """取某个数据集档案的原始 mapping（不构造 DataConfig，供 IO 层按需读取）。"""
        key = name or self.dataset.name
        if key not in self.datasets:
            raise ConfigError(f"未定义的数据集档案：{key!r}", hint=f"已定义：{sorted(self.datasets)}")
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
# 加载结果
# ============================================================================
@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """加载完成的配置 + 溯源信息。所有 pipeline 的入参都应该是它。"""

    config: AppConfig
    sources: tuple[str, ...]
    overrides: Mapping[str, Any]
    config_hash: str
    out_dir: Path
    label_space: str
    weight_path: Path | None = None

    # --- 透明代理，让 `rc.split` 这类写法依然可用 -------------------------
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
    """配置指纹：用于判断两次实验是否真的用了同一套参数。"""
    canonical = json.dumps(_to_plain(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dump_config(resolved: ResolvedConfig, path: str | Path) -> Path:
    """把最终生效配置写进输出目录（每个 run 都必须有 resolved_config.yaml）。"""
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
        raise ConfigError(f"配置文件 YAML 解析失败：{path}（{exc}）") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"配置文件的顶层必须是 mapping：{path}，实际 {type(loaded).__name__}")
    return dict(loaded)


def load_config(
    paths: Sequence[str | Path] = (),
    *,
    overrides: Mapping[str, Any] | None = None,
    extra_paths: Sequence[str | Path] = (),
) -> ResolvedConfig:
    """加载并校验配置。

    Parameters
    ----------
    paths:
        base 配置文件，按顺序深合并（后者覆盖前者）。空则用 ``configs/config.yaml``。
    overrides:
        CLI 覆盖（已解析成嵌套 dict），最后应用。
    extra_paths:
        实验覆盖文件，插在 ``paths`` 之后、``overrides`` 之前。

    Examples
    --------
    >>> rc = load_config(["configs/config.yaml"], overrides={"train": {"epochs": 3}})
    >>> rc.train.epochs
    3
    """
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
        (configured,) if configured.is_absolute() else
        (app.resolve_models_dir() / configured, app.paths.resolve(str(configured)))
    )
    for candidate in candidates:
        if candidate.is_file():
            if arch not in _ULTRALYTICS_ARCHES:
                raise ConfigError(
                    f"model.arch={arch!r} 不支持从本地 Ultralytics .pt 权重初始化",
                    hint="本地 .pt 初始化目前只支持 yolo26n-cls、yolo26m-cls、yolon-cls 和 yolov8n-cls。",
                )
            return candidate.resolve()

    known_model_files = {f"{arch}.pt", f"{alias_arch}.pt"}
    if configured.name.lower() in known_model_files:
        # Ultralytics may download its official architecture weight by name.
        return None
    if configured.suffix.lower() in {".pt", ".pth", ".ckpt"} or configured.parent != Path("."):
        searched = ", ".join(str(path) for path in candidates)
        raise ConfigError(
            f"找不到 model.pretrained 指定的权重文件：{raw!r}",
            hint=f"已检查：{searched}。把文件放到 paths.models_dir，或填写绝对路径。",
        )
    raise ConfigError(
        f"无法识别 model.pretrained 权重名：{raw!r}",
        hint=f"可用 auto、true、false、imagenet，或现有本地权重文件；当前 arch={arch!r}。",
    )
