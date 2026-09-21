"""全项目异常体系。

规则（CONTRIBUTING.md R9）：
    * 库代码**禁止**裸 ``raise Exception`` / ``sys.exit`` / ``assert`` 做输入校验。
    * 可预期的失败一律抛这里的子类，由最外层 CLI 统一转成退出码与友好提示。
    * 异常消息必须能回答"哪个文件 / 哪个字段 / 期望什么 / 实际什么"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "HandwashError",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigKeyError",
    "DataError",
    "DatasetNotFoundError",
    "VideoDecodeError",
    "ManifestError",
    "DataLeakageError",
    "ModelError",
    "ModelNotFoundError",
    "CheckpointError",
    "BackendUnavailableError",
    "TrainingError",
    "EvaluationError",
    "ProtocolError",
]


class HandwashError(Exception):
    """项目内所有自定义异常的基类。"""

    exit_code: int = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  → 建议：{self.hint}"
        return self.message


# --- 配置 -------------------------------------------------------------------
class ConfigError(HandwashError):
    """配置相关的任何问题（缺字段、类型错、取值非法）。"""

    exit_code = 2


class ConfigFileNotFoundError(ConfigError):
    def __init__(self, path: str | Path) -> None:
        super().__init__(
            f"找不到配置文件：{path}",
            hint="确认路径是否相对仓库根目录；用 `handwash doctor` 可列出全部可用配置。",
        )
        self.path = Path(path)


class ConfigKeyError(ConfigError):
    """未知键或类型不匹配。严格模式下未知键必须报错，不能静默忽略。"""

    def __init__(self, key: str, *, expected: str, got: Any, where: str = "config") -> None:
        super().__init__(
            f"{where} 中的键 `{key}` 非法：期望 {expected}，实际为 {type(got).__name__}={got!r}",
            hint="键名拼写与层级必须与 src/handwash/core/config.py 中的 dataclass 完全一致。",
        )
        self.key = key


# --- 数据 -------------------------------------------------------------------
class DataError(HandwashError):
    """数据层通用错误。"""

    exit_code = 3


class DatasetNotFoundError(DataError):
    def __init__(self, name: str, expected_at: str | Path | None = None) -> None:
        where = f"，期望位置：{expected_at}" if expected_at else ""
        super().__init__(
            f"数据集 `{name}` 不存在{where}",
            hint="见 docs/DATA.md 的下载链接；也可设置环境变量 HANDWASH_DATA_ROOT 指向外部数据盘。",
        )
        self.name = name


class VideoDecodeError(DataError):
    def __init__(self, path: str | Path, reason: str) -> None:
        super().__init__(
            f"视频解码失败：{path}（{reason}）",
            hint="确认文件完整、扩展名与编码匹配；必要时安装 opencv-python 或 imageio-ffmpeg。",
        )
        self.path = Path(path)


class ManifestError(DataError):
    """manifest 缺列、类型错、或与标签空间不一致。"""


class DataLeakageError(DataError):
    """检测到同一原始视频同时出现在多个 split —— 必须直接失败，不能继续训练。"""

    def __init__(self, clip_ids: list[str]) -> None:
        preview = ", ".join(sorted(clip_ids)[:5])
        more = f" 等 {len(clip_ids)} 段" if len(clip_ids) > 5 else ""
        super().__init__(
            f"数据泄漏：以下原始视频跨越了多个 split：{preview}{more}",
            hint="必须按原始视频划分 train/val/test（configs/config.yaml -> split.group_key=original_video）。",
        )
        self.clip_ids = clip_ids


# --- 模型 -------------------------------------------------------------------
class ModelError(HandwashError):
    """模型层通用错误。"""

    exit_code = 4


class ModelNotFoundError(ModelError):
    def __init__(self, name: str, available: list[str] | None = None) -> None:
        avail = f"；已注册：{', '.join(sorted(available))}" if available else ""
        super().__init__(
            f"模型 `{name}` 未注册{avail}",
            hint="新模型必须用 @register_model('名字') 装饰器注册，见 src/handwash/models/registry.py。",
        )
        self.name = name


class CheckpointError(ModelError):
    pass


class BackendUnavailableError(ModelError):
    """可选后端（torch / ultralytics / opencv）没装。"""

    def __init__(self, backend: str, install_hint: str) -> None:
        super().__init__(f"后端 `{backend}` 不可用", hint=install_hint)
        self.backend = backend


# --- 训练 / 评估 ------------------------------------------------------------
class TrainingError(HandwashError):
    exit_code = 5


class EvaluationError(HandwashError):
    exit_code = 5


# --- 业务规则 ---------------------------------------------------------------
class ProtocolError(HandwashError):
    """WHO 完整性判定相关的错误（阈值非法、步骤序列不可判定等）。"""

    exit_code = 6
