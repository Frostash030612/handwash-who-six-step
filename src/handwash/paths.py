"""路径常量：全项目**唯一**允许出现 ``"data"``、``"outputs"`` 等目录名的地方。

规则依据 CONTRIBUTING.md R8：任何其他模块若出现硬编码目录字符串，
``scripts/check_structure.py`` 会报错。

本模块属于 L0 层，只能依赖标准库。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "DATA_RAW",
    "DATA_INTERIM",
    "DATA_PROCESSED",
    "DATA_EXTERNAL",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "DOCS_DIR",
    "TESTS_DIR",
    "SCRIPTS_DIR",
    "DATA_RAW_DIRNAME",
    "DATA_INTERIM_DIRNAME",
    "DATA_PROCESSED_DIRNAME",
    "DATA_EXTERNAL_DIRNAME",
    "OUTPUTS_DIRNAME",
    "MODELS_DIRNAME",
    "SYNTHETIC_DIRNAME",
    "DEFAULT_MANIFEST_RELPATH",
    "CHECKPOINTS_SUBDIR",
    "BEST_CHECKPOINT_NAME",
    "LAST_CHECKPOINT_NAME",
    "data_root",
    "ensure_dir",
    "project_path",
    "resolve_relative",
    "checkpoint_path",
]


def _detect_project_root() -> Path:
    """从当前文件位置推断仓库根目录（src/handwash/paths.py -> 根）。"""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _detect_project_root()

SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_EXTERNAL: Path = DATA_DIR / "external"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

#: 目录名常量（**不要在其它模块里重复写这些字符串**，见 CONTRIBUTING.md R8）。
#: 代码里出现 `"data/processed"` 这类字面量会被 scripts/check_structure.py 拦下。
DATA_RAW_DIRNAME: Final[str] = "data/raw"
DATA_INTERIM_DIRNAME: Final[str] = "data/interim"
DATA_PROCESSED_DIRNAME: Final[str] = "data/processed"
DATA_EXTERNAL_DIRNAME: Final[str] = "data/external"
OUTPUTS_DIRNAME: Final[str] = "outputs"
MODELS_DIRNAME: Final[str] = "models"
SYNTHETIC_DIRNAME: Final[str] = "data/interim/synthetic"

#: 默认 manifest 路径（相对仓库根），供配置与脚本共用一个来源
DEFAULT_MANIFEST_RELPATH: Final[Path] = Path(DATA_PROCESSED_DIRNAME) / "manifest.csv"

#: 每个实验目录下放 checkpoint 的子目录名。
#: 冻结为常量：`handwash train` 与 `handwash evaluate` 必须指向同一个位置，
#: 否则会出现"训练完评估却找不到权重"这种低级但很费时的错。
CHECKPOINTS_SUBDIR: Final[str] = "models"

#: 默认 checkpoint 文件名（best = 按验证集 Macro-F1 选出）
BEST_CHECKPOINT_NAME: Final[str] = "best.pt"
LAST_CHECKPOINT_NAME: Final[str] = "last.pt"


def data_root() -> Path:
    """数据根目录：默认 ``<repo>/data``，可由环境变量 ``HANDWASH_DATA_ROOT`` 覆盖。

    覆盖场景：数据集放在外置硬盘 / 实验室共享盘，不想复制进仓库。
    """
    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    return DATA_DIR


def project_path(*parts: str | os.PathLike[str], root: Path | None = None) -> Path:
    """在仓库根目录下拼接路径（不创建目录）。"""
    base = PROJECT_ROOT if root is None else Path(root)
    return base.joinpath(*[str(p) for p in parts])


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """确保目录存在并返回它。所有写文件前都必须走这里，禁止裸 ``mkdir``。"""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_relative(value: str | os.PathLike[str], *, root: Path | None = None) -> Path:
    """把配置里的路径解析成绝对路径。

    规则：相对路径一律相对**仓库根目录**（不是当前工作目录）——
    这样无论从哪个目录启动脚本，结果都一致，杜绝"在我电脑上能跑"。
    """
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    base = PROJECT_ROOT if root is None else Path(root)
    return base / path


def checkpoint_path(out_dir: str | Path, name: str = BEST_CHECKPOINT_NAME) -> Path:
    """``<实验目录>/models/<name>`` —— 训练保存与评估加载共用这一个函数。"""
    return Path(out_dir) / CHECKPOINTS_SUBDIR / name
