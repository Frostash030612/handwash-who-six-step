"""Package-relative paths and optional external dataset-root resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "BEST_CHECKPOINT_NAME",
    "CHECKPOINTS_SUBDIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "DATA_EXTERNAL",
    "DATA_EXTERNAL_DIRNAME",
    "DATA_INTERIM",
    "DATA_INTERIM_DIRNAME",
    "DATA_PROCESSED",
    "DATA_PROCESSED_DIRNAME",
    "DATA_RAW",
    "DATA_RAW_DIRNAME",
    "DEFAULT_MANIFEST_RELPATH",
    "DOCS_DIR",
    "LAST_CHECKPOINT_NAME",
    "MODELS_DIR",
    "MODELS_DIRNAME",
    "OUTPUTS_DIR",
    "OUTPUTS_DIRNAME",
    "PROJECT_ROOT",
    "SCRIPTS_DIR",
    "SRC_DIR",
    "SYNTHETIC_DIRNAME",
    "TESTS_DIR",
    "checkpoint_path",
    "data_root",
    "ensure_dir",
    "project_path",
    "resolve_relative",
]


def _detect_project_root() -> Path:

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


DATA_RAW_DIRNAME: Final[str] = "data/raw"
DATA_INTERIM_DIRNAME: Final[str] = "data/interim"
DATA_PROCESSED_DIRNAME: Final[str] = "data/processed"
DATA_EXTERNAL_DIRNAME: Final[str] = "data/external"
OUTPUTS_DIRNAME: Final[str] = "outputs"
MODELS_DIRNAME: Final[str] = "models"
SYNTHETIC_DIRNAME: Final[str] = "data/interim/synthetic"


DEFAULT_MANIFEST_RELPATH: Final[Path] = Path(DATA_PROCESSED_DIRNAME) / "manifest.csv"


CHECKPOINTS_SUBDIR: Final[str] = "models"


BEST_CHECKPOINT_NAME: Final[str] = "best.pt"
LAST_CHECKPOINT_NAME: Final[str] = "last.pt"


def data_root() -> Path:

    override = os.environ.get("HANDWASH_DATA_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    return DATA_DIR


def project_path(*parts: str | os.PathLike[str], root: Path | None = None) -> Path:

    base = PROJECT_ROOT if root is None else Path(root)
    return base.joinpath(*[str(p) for p in parts])


def ensure_dir(path: str | os.PathLike[str]) -> Path:

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_relative(value: str | os.PathLike[str], *, root: Path | None = None) -> Path:

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if root is not None:
        return Path(root) / path
    # Paths under data/ follow HANDWASH_DATA_ROOT, as documented. Outputs and
    # configuration files remain relative to the repository root.
    if path.parts and path.parts[0] == "data":
        return data_root().joinpath(*path.parts[1:])
    return PROJECT_ROOT / path


def checkpoint_path(out_dir: str | Path, name: str = BEST_CHECKPOINT_NAME) -> Path:

    return Path(out_dir) / CHECKPOINTS_SUBDIR / name
