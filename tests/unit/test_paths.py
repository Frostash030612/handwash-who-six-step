"""路径常量契约测试（src/handwash/paths.py）。

这个模块是全项目唯一允许出现 "data" / "outputs" 字面量的地方（CONTRIBUTING R8），
它同时也是"能不能在我电脑上跑"的根源：PROJECT_ROOT 推断错一步，所有相对路径都会
指到别的地方。因此这里既锁根目录的推断方式，也锁相对/绝对路径的解析语义。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from handwash import paths
from handwash.paths import (
    DATA_DIR,
    DATA_EXTERNAL,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_RAW,
    PROJECT_ROOT,
    data_root,
    ensure_dir,
    project_path,
)

pytestmark = pytest.mark.unit


# --- 根目录推断 -------------------------------------------------------------
def test_project_root_points_at_repository_root() -> None:
    """根目录由 ``src/handwash/paths.py`` 上溯两级得到；判据是仓库根才有的文件。"""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "handwash" / "paths.py").is_file()
    assert PROJECT_ROOT.name != "handwash"
    assert PROJECT_ROOT.is_absolute()


def test_project_root_equals_package_parents_two() -> None:
    """推断方式必须与源码一致：paths.py -> handwash -> src -> 根。"""
    assert Path(paths.__file__).resolve().parents[2] == PROJECT_ROOT


@pytest.mark.parametrize(
    ("constant", "expected"),
    [
        ("SRC_DIR", "src"),
        ("CONFIGS_DIR", "configs"),
        ("DATA_DIR", "data"),
        ("MODELS_DIR", "models"),
        ("OUTPUTS_DIR", "outputs"),
        ("DOCS_DIR", "docs"),
        ("TESTS_DIR", "tests"),
        ("SCRIPTS_DIR", "scripts"),
    ],
)
def test_top_level_constants_are_single_directory_names(constant: str, expected: str) -> None:
    """顶层目录名一旦改动，所有文档与脚本里的路径都会失效。"""
    value = getattr(paths, constant)
    assert value == PROJECT_ROOT / expected


def test_data_subdirectories_form_the_documented_pipeline_layout() -> None:
    """interim/processed/external 的阶段划分是数据流水线的约定，不能随意挪。"""
    assert DATA_RAW == DATA_DIR / "raw"
    assert DATA_INTERIM == DATA_DIR / "interim"
    assert DATA_PROCESSED == DATA_DIR / "processed"
    assert DATA_EXTERNAL == DATA_DIR / "external"


def test_every_directory_constant_is_an_absolute_path() -> None:
    """**目录**常量必须是绝对路径：相对路径常量会让"从别的目录运行脚本"得到不同结果。

    例外：``DEFAULT_MANIFEST_RELPATH`` 刻意是**相对**仓库根的 Path 对象
    （它是配置默认值，必须可跨机器搬移，见 paths.py 的注释）。它以 ``RELPATH``
    结尾命名，就是为了让"这是相对路径"这件事在调用处一眼可见。
    """
    for name in paths.__all__:
        value = getattr(paths, name)
        if isinstance(value, Path) and not name.endswith("RELPATH"):
            assert value.is_absolute(), name

    assert not paths.DEFAULT_MANIFEST_RELPATH.is_absolute()
    assert str(paths.DEFAULT_MANIFEST_RELPATH).endswith("manifest.csv")


# --- ensure_dir -------------------------------------------------------------
def test_ensure_dir_creates_nested_directories(tmp_path: Path) -> None:
    """写文件前统一走 ensure_dir，避免各自裸 mkdir 时忘记 parents=True。"""
    target = tmp_path / "a" / "b" / "c"
    assert ensure_dir(target) == target
    assert target.is_dir()


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    """同一路径被多次调用（多个模块都会建同一个输出目录）必须无副作用。"""
    target = tmp_path / "nested" / "out"
    first = ensure_dir(target)
    marker = target / "keep.txt"
    marker.write_text("x", encoding="utf-8")
    second = ensure_dir(target)
    assert first == second
    assert marker.read_text(encoding="utf-8") == "x"


def test_ensure_dir_accepts_string_path(tmp_path: Path) -> None:
    """调用方经常直接传入配置里的字符串路径。"""
    target = ensure_dir(str(tmp_path / "from_str"))
    assert isinstance(target, Path)
    assert target.is_dir()


def test_ensure_dir_on_existing_file_raises(tmp_path: Path) -> None:
    """目标是文件时不能静默通过，否则后续写文件会在难以理解的地方失败。"""
    file_path = tmp_path / "afile"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(OSError):
        ensure_dir(file_path)


# --- data_root --------------------------------------------------------------
def test_data_root_defaults_to_repo_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设置环境变量时就是 <repo>/data，保证新人 clone 后行为一致。"""
    monkeypatch.delenv("HANDWASH_DATA_ROOT", raising=False)
    assert data_root() == DATA_DIR


def test_data_root_ignores_blank_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """空字符串（常见于 CI 里写了 ``VAR=``）必须当作"未设置"。"""
    monkeypatch.setenv("HANDWASH_DATA_ROOT", "   ")
    assert data_root() == DATA_DIR


def test_data_root_honours_absolute_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """数据集放在外置硬盘时用绝对路径覆盖，不能在仓库里再复制一份。"""
    monkeypatch.setenv("HANDWASH_DATA_ROOT", str(tmp_path))
    resolved = data_root()
    assert resolved == tmp_path.resolve()
    assert resolved != DATA_DIR


def test_data_root_resolves_relative_env_against_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """相对路径必须相对仓库根解析，而不是相对当前工作目录（否则结果随 cwd 变化）。"""
    monkeypatch.setenv("HANDWASH_DATA_ROOT", "external_data")
    assert data_root() == (PROJECT_ROOT / "external_data").resolve()


def test_data_root_expands_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """写 ``~`` 是常见习惯，必须展开而不是当成字面目录名。"""
    monkeypatch.setenv("HANDWASH_DATA_ROOT", "~/handwash_data")
    resolved = data_root()
    assert "~" not in str(resolved)
    assert resolved.is_absolute()


# --- project_path -----------------------------------------------------------
def test_project_path_joins_under_project_root_by_default() -> None:
    """拼接路径默认以仓库根为基准，避免模块里出现 "outputs/..." 之类的相对字面量。"""
    expected = PROJECT_ROOT / "outputs" / "exp1" / "metrics.json"
    assert project_path("outputs", "exp1", "metrics.json") == expected


def test_project_path_accepts_custom_root(tmp_path: Path) -> None:
    """测试与会话级脚本会用临时目录当根。"""
    assert project_path("a", "b", root=tmp_path) == tmp_path / "a" / "b"


def test_project_path_accepts_pathlike_parts() -> None:
    """允许传入 Path 片段，省掉调用方的一堆 str() 转换。"""
    assert project_path(Path("outputs"), Path("run")) == PROJECT_ROOT / "outputs" / "run"


def test_project_path_does_not_create_directories(tmp_path: Path) -> None:
    """拼接路径是纯函数，创建目录必须由 ensure_dir 显式完成。"""
    target = project_path("never_created", root=tmp_path)
    assert not target.exists()


def test_project_path_with_no_parts_returns_root(tmp_path: Path) -> None:
    """边界情况：不传片段时返回根目录本身。"""
    assert project_path(root=tmp_path) == tmp_path
