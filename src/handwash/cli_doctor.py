"""``handwash doctor``：环境与配置自检。

新组员克隆仓库后的第一条命令。它回答四个问题：
    1. 我装的依赖够不够？（缺什么、怎么补）
    2. 配置能加载吗？有没有拼错的键、越界的取值？
    3. 数据在哪？manifest 生成了吗？划分是否泄漏？
    4. 能不能用 GPU？主模型权重能不能加载？

设计原则：**只报告，不修改**。doctor 永远不写文件、不下载权重、不改环境。
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any

from handwash import __version__
from handwash.core.config import ResolvedConfig
from handwash.core.labels import LABEL_SPACES, get_label_space
from handwash.logging import get_logger
from handwash.paths import PROJECT_ROOT, data_root

__all__ = ["run_doctor", "collect_checks"]

log = get_logger(__name__)

#: (导入名, 是否必需, 安装提示)
_PACKAGES: tuple[tuple[str, bool, str], ...] = (
    ("numpy", True, "conda env update -f environment.yml"),
    ("yaml", True, "conda install pyyaml"),
    ("torch", True, "conda install -c pytorch pytorch"),
    ("torchvision", True, "conda install -c pytorch torchvision"),
    ("PIL", True, "conda install pillow"),
    ("pandas", False, "conda install pandas（画表/分析用）"),
    ("sklearn", False, "conda install scikit-learn（对比指标时用）"),
    ("matplotlib", False, "conda install matplotlib（混淆矩阵图用）"),
    ("cv2", False, "conda install -c conda-forge opencv（视频解码更快）"),
    ("imageio", False, "conda install imageio imageio-ffmpeg（cv2 缺失时的回退解码器）"),
    ("ultralytics", False, "pip install ultralytics（主模型 YOLO26n-cls 需要）"),
    ("pytest", False, "pip install pytest（跑测试用）"),
    ("ruff", False, "pip install ruff（代码检查用）"),
)

_OK = "OK"
_WARN = "WARN"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _check_packages() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for name, required, hint in _PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            rows.append((name, _OK, f"{version}"))
        except Exception as exc:  # noqa: BLE001 - 导入失败原因很多（含二进制不兼容）
            status = _FAIL if required else _WARN
            rows.append((name, status, f"缺失（{type(exc).__name__}）；{hint}"))
    return rows


def _check_runtime() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    py = f"{platform.python_version()} ({platform.system()} {platform.machine()})"
    rows.append(("python", _OK if sys.version_info >= (3, 10) else _FAIL, py))
    rows.append(("handwash", _OK, __version__))
    rows.append(("project_root", _OK, str(PROJECT_ROOT)))

    try:
        import torch

        rows.append(("torch.cuda.is_available", _OK if torch.cuda.is_available() else _WARN,
                     str(torch.cuda.is_available())))
        if torch.cuda.is_available():
            rows.append(("gpu", _OK, torch.cuda.get_device_name(0)))
            rows.append(("cuda_version", _OK, str(torch.version.cuda)))
        else:
            rows.append(("gpu", _WARN, "不可用：训练会自动使用 CPU（会明显变慢）"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("torch", _FAIL, f"{type(exc).__name__}: {exc}"))
    return rows


def _check_config(rc: ResolvedConfig | None, error: Exception | None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if error is not None or rc is None:
        rows.append(("config", _FAIL, f"加载失败：{error}"))
        return rows

    rows.append(("config_hash", _OK, rc.config_hash))
    rows.append(("config_sources", _OK, " -> ".join(rc.sources)))
    rows.append(("label_space", _OK, f"{rc.label_space}（{len(get_label_space(rc.label_space))} 类）"))
    rows.append(("arch / mode", _OK, f"{rc.model.arch} / {rc.train.mode}"))
    rows.append(("out_dir", _OK, str(rc.out_dir)))

    # 交叉检查：YOLO 适配器必须配 zero_one 归一化，否则等于归一化两次
    if "yolo" in str(rc.model.arch).lower() and rc.model.normalize == "imagenet":
        rows.append(
            (
                "normalize",
                _FAIL,
                "使用 YOLO 分类模型时 model.normalize 必须是 zero_one"
                "（ultralytics 内部已做归一化，否则会归一化两次，准确率异常偏低）",
            )
        )
    else:
        rows.append(("normalize", _OK, rc.model.normalize))

    if rc.dataset.name not in LABEL_SPACES:
        rows.append(
            (
                "dataset.name",
                _WARN,
                f"`{rc.dataset.name}` 不是内置标签空间，将使用 `{rc.label_space}` 作为标签空间",
            )
        )
    return rows


def _check_data(rc: ResolvedConfig | None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    rows.append(("data_root", _OK, str(data_root())))
    if rc is None:
        rows.append(("dataset", _SKIP, "配置未加载，跳过数据检查"))
        return rows

    try:
        spec = rc.dataset_spec()
    except Exception as exc:  # noqa: BLE001
        rows.append(("dataset", _FAIL, str(exc)))
        return rows

    from pathlib import Path

    root = spec.get("root") or rc.dataset.root
    resolved = Path(str(root)) if Path(str(root)).is_absolute() else PROJECT_ROOT / str(root)
    if "synthetic" in str(rc.dataset.name):
        rows.append(("dataset.root", _OK, f"{resolved}（合成数据不需要真实文件）"))
    elif resolved.exists():
        rows.append(("dataset.root", _OK, str(resolved)))
    else:
        rows.append(("dataset.root", _WARN, f"不存在：{resolved}（见 docs/DATA.md 下载数据）"))

    manifest = spec.get("manifest") or "data/processed/manifest.csv"
    manifest_path = Path(str(manifest)) if Path(str(manifest)).is_absolute() else PROJECT_ROOT / str(manifest)
    if manifest_path.exists():
        try:
            from handwash.io.manifest import manifest_summary, read_manifest

            records = read_manifest(manifest_path)
            summary = manifest_summary(records)
            rows.append(
                (
                    "manifest",
                    _OK,
                    f"{manifest_path.name}：{summary['num_frames']} 帧 / {summary['num_clips']} 段 / "
                    f"split={summary['frames_per_split']}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(("manifest", _FAIL, f"存在但校验失败：{exc}"))
    else:
        rows.append(
            ("manifest", _WARN, f"未生成：{manifest_path}（先运行 handwash prepare；训练会自动回退到合成数据）")
        )
    return rows


def _check_model(rc: ResolvedConfig | None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if rc is None:
        rows.append(("model", _SKIP, "配置未加载，跳过模型检查"))
        return rows

    arch = str(rc.model.arch).lower()
    if "yolo" in arch:
        try:
            from handwash.models.yolo26_cls import probe_yolo_availability

            probe = probe_yolo_availability(rc.model.pretrained if isinstance(rc.model.pretrained, str) else "yolo26n-cls.pt")
            if not probe.get("ultralytics"):
                rows.append(("ultralytics", _FAIL, str(probe.get("hint"))))
            elif probe.get("loadable"):
                rows.append(("yolo 权重", _OK, f"可加载（ultralytics {probe.get('version')}）"))
            else:
                rows.append(("yolo 权重", _WARN, f"{probe.get('error')}；{probe.get('hint')}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(("yolo", _FAIL, f"{type(exc).__name__}: {exc}"))
    else:
        rows.append(("model.arch", _OK, arch))
    return rows


def collect_checks(rc: ResolvedConfig | None = None, error: Exception | None = None) -> list[tuple[str, str, str]]:
    """执行全部检查，返回 ``(项目, 状态, 说明)`` 列表。"""
    rows: list[tuple[str, str, str]] = []
    rows.extend(_check_runtime())
    rows.extend(_check_packages())
    rows.extend(_check_config(rc, error))
    rows.extend(_check_data(rc))
    rows.extend(_check_model(rc))
    return rows


def run_doctor(rc: ResolvedConfig | None = None, error: Exception | None = None) -> int:
    """打印体检报告；有 FAIL 时返回非零退出码（CI 可直接用）。"""
    rows = collect_checks(rc, error)
    width = max(len(name) for name, _, _ in rows) + 2
    print("=" * 78)
    print(f"handwash doctor —— 环境与配置自检（v{__version__}）")
    print("=" * 78)
    for name, status, detail in rows:
        print(f"[{status:<4}] {name:<{width}} {detail}")
    print("-" * 78)

    failures = [row for row in rows if row[1] == _FAIL]
    warnings = [row for row in rows if row[1] == _WARN]
    if failures:
        print(f"发现 {len(failures)} 个必须解决的问题，{len(warnings)} 个提醒。")
        for name, _, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    if warnings:
        print(f"环境可用，但有 {len(warnings)} 个提醒（不阻塞开发）。")
    else:
        print("全部检查通过。")
    return 0
