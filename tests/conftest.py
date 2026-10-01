"""tests/conftest.py —— 全局 fixture 与公共工具。

约定（全组必须一致，否则"我这儿过了"没有意义）：
    * 所有测试默认标记为 ``unit``：秒级完成、不读数据集、不碰 GPU。
      需要真实数据请标 ``integration``；需要训练/推理请标 ``slow``；
      需要 CUDA 请标 ``gpu``。
    * 需要随机性的测试必须自己固定种子（``np.random.default_rng(seed)``），
      不允许依赖全局随机状态 —— 否则同样的代码会在不同机器上给出不同结论。
    * 任何写文件的操作只能落在 pytest 的 ``tmp_path`` / ``tmp_path_factory``
      里。仓库根目录是共享的，测试往里写东西会污染组员的实验产物。
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from handwash.core.labels import CANONICAL_STEPS, Step
from handwash.paths import SRC_DIR

#: 契约层测试统一使用的抽帧帧率（与 configs 默认 data.prep.fps 一致）。
FPS: float = 5.0

#: 契约层测试统一使用的"一步做多久"（帧数 = 秒数 × FPS）。
STEP_SECONDS: float = 10.0


def _ensure_src_on_path() -> None:
    """让 ``import handwash`` 在未 pip install -e 的机器上也能工作。

    正常路径是 ``pip install -e .``（pyproject 里 ``package-dir = {"" = "src"}``）；
    这里只是兜底，避免新组员刚 clone 就满屏 ImportError。
    """
    entry = str(SRC_DIR)
    if SRC_DIR.is_dir() and entry not in sys.path:
        sys.path.insert(0, entry)


_ensure_src_on_path()


# ============================================================================
# 临时目录：默认落在仓库内的 .tmp/
# ----------------------------------------------------------------------------
# 为什么不用 pytest 自带的 tmp_path？
#   pytest 默认在系统临时目录（%TEMP% / /tmp）下再建子目录，这在受限环境
#   （CI 容器、企业策略锁定的机器、某些沙箱）里会被拒绝，表现为
#   `PermissionError: [WinError 5]`，且**所有用到 tmp_path 的测试一起报错**，
#   看起来像代码坏了，实际是环境不允许。
#   改成在仓库内 .tmp/ 下建独立目录（已在 .gitignore 忽略），行为完全一致：
#   每个测试仍有自己的目录，测试之间互不干扰，也不会污染 outputs/。
#
# 若你希望用系统临时目录（本地开发通常更快），删掉下面三个 fixture 即可。
#
# 也可以用环境变量指定根目录（用于受限环境）：
#     Windows  $env:HANDWASH_TMP_ROOT="$env:TEMP\hw_pytest"
#     macOS    HANDWASH_TMP_ROOT=/tmp/hw_pytest pytest
# 为什么需要这个开关：某些受限环境（CI 容器、带 ACL 限制的沙箱）不允许在仓库
# 目录内新建目录，于是所有用到 tmp_path 的测试都会以 PermissionError 报 setup 错误。
# 这时把临时根换到系统临时目录即可，不必改动测试代码。
# ============================================================================
_WS_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp"


def _temp_root() -> Path:
    """决定测试临时目录的根。

    优先级：``HANDWASH_TMP_ROOT`` 环境变量 > 仓库内 ``.tmp/``。
    仓库内是默认值（便于在同一盘上做 IO，也方便出问题时手工翻看残留文件），
    但它要求仓库目录可写；受限环境请用环境变量覆盖到系统临时目录。
    """
    override = os.environ.get("HANDWASH_TMP_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return _WS_TEMP_ROOT


class _WorkspaceTempFactory:
    """``tmp_path_factory`` 的最小实现：在临时根目录下建目录。

    目录名带进程号与自增计数，保证并发跑测试（pytest-xdist）也不会撞名。
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise RuntimeError(
                f"无法创建测试临时目录：{self._root}\n"
                "当前环境不允许在仓库内写文件。请改用系统临时目录，例如：\n"
                '    Windows  $env:HANDWASH_TMP_ROOT="$env:TEMP\\hw_pytest"\n'
                "    macOS    export HANDWASH_TMP_ROOT=/tmp/hw_pytest\n"
                "（设置后重新运行 pytest）"
            ) from exc
        self._counter = 0

    @property
    def basetemp(self) -> Path:
        return self._root

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        self._counter += 1
        suffix = f"-{os.getpid()}-{self._counter}" if numbered else ""
        target = self._root / f"{basename}{suffix}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def getbasetemp(self) -> Path:
        return self._root


@pytest.fixture(scope="session")
def tmp_path_factory() -> _WorkspaceTempFactory:
    """会话级临时目录工厂（见上方说明）。"""
    return _WorkspaceTempFactory(_temp_root())


@pytest.fixture
def tmp_path(tmp_path_factory: _WorkspaceTempFactory, request: pytest.FixtureRequest) -> Path:
    """每个测试一个独立的临时目录（见上方说明）。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.node.name)[:50]
    return tmp_path_factory.mktemp(safe or "test")


def make_sequence(
    specs: Sequence[tuple[Step, int]] | Mapping[Step, int],
    *,
    default_frames: int | None = None,
) -> list[Step]:
    """把 ``[(Step.STEP_1, 25), ...]`` 展开成逐帧标签序列。

    为什么需要它：``build_report`` 的输入是**逐帧**标签，手写 250 个元素既难读
    又容易写错一步的帧数，测试意图会被淹没。
    """
    items = list(specs.items()) if isinstance(specs, Mapping) else list(specs)
    if default_frames is None:
        default_frames = int(STEP_SECONDS * FPS)
    return [step for step, frames in items for _ in range(int(frames or default_frames))]


@pytest.fixture(scope="session", autouse=True)
def _cleanup_workspace_temp_root() -> object:
    """测试结束后清掉本进程在 ``.tmp/`` 下建的目录，避免仓库里堆垃圾。

    只删本进程自己创建的目录（名字里带本进程号），不碰别人的。
    """
    yield None
    marker = f"-{os.getpid()}-"
    root = _WS_TEMP_ROOT
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.is_dir() and marker in child.name:
            shutil.rmtree(child, ignore_errors=True)


@pytest.fixture(scope="session")
def fps() -> float:
    """测试统一帧率：5 fps 下 1 帧 = 0.2 秒，便于把秒数换算成帧数。"""
    return FPS


@pytest.fixture(scope="session")
def frames_per_step() -> int:
    """``STEP_SECONDS`` 秒对应的帧数（默认 50 帧 = 10 秒）。"""
    return int(STEP_SECONDS * FPS)


@pytest.fixture
def full_protocol_sequence(frames_per_step: int) -> list[Step]:
    """一段"完全规范"的六步序列：每步 10 秒，顺序正确，共 60 秒。

    用途：作为基线，任何判定异常都可以先用它做对照。
    """
    return make_sequence([(step, frames_per_step) for step in CANONICAL_STEPS])


@pytest.fixture
def base_config_payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """一份最小但合法的配置 mapping。

    关键点：``paths.out_dir`` 指向临时目录，这样 ``load_config`` 派生的
    ``out_dir`` 不会落到仓库根的 ``outputs/``，测试不会污染组员产物。
    """
    out_dir = tmp_path_factory.mktemp("handwash-out") / "outputs"
    return {
        "schema_version": 1,
        "project": {"name": "testproj"},
        "runtime": {"seed": 7, "run_name": "pytest"},
        "paths": {"out_dir": str(out_dir)},
        "dataset": {"name": "kaggle", "root": "data/raw/kaggle"},
        "datasets": {"kaggle": {"root": "data/raw/kaggle"}},
        "train": {"epochs": 2},
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """把 mapping 写成 UTF-8 YAML 并返回路径（只写在 tmp_path 内）。"""

    def _write(payload: Mapping[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    return _write


@pytest.fixture
def config_file(write_config: Callable[..., Path], base_config_payload: dict[str, Any]) -> Path:
    """最小合法配置文件。"""
    return write_config(base_config_payload)
