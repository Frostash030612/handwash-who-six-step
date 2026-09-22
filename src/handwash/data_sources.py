"""公开数据集的来源登记：记录号、许可、体积、md5 校验值。

为什么单独一个模块（CONTRIBUTING.md R3）
------------------------------------------
数据集来源、许可、校验值属于**配置数据**：Zenodo 会加文件、上游会重新打包，
但改动都是纯数据替换。放在这里之后：
    * ``scripts/download_data.py`` 只负责"下载 + 校验 + 记账"的机器逻辑；
    * 换数据集、加数据集不需要动任何脚本；
    * 单元测试可以对着这份表断言，完全不需要网络。

三层数据的定位见 ``docs/DATA_COLLABORATION.md``：
    原始数据不进 Git（从公开来源按记录号下载 + md5 校验一致性）；
    抽帧结果不进 Git（组内共享介质流转）；
    来源清单 / 划分 / 指标进 Git（这才是让结果可复现的东西）。

【单位约定 —— 踩过坑，务必遵守】
    登记表里的 ``size_bytes`` **一律是字节**，用下面的 ``KiB`` / ``MiB`` / ``GiB``
    辅助常量书写。

    为什么不用 GB：写这个模块时先用了"GB"当单位，结果在换算上连续出错两次 ——
    一次把 GB 当 TiB（多乘 1024），一次把 MB 当 GiB（DataSet1 变成 597 GB）。
    根因是"GB"这个单位本身有歧义（10^9 还是 2^30？）。
    现在用显式字节 + 明确的二进制单位常量，**读代码时一眼能看出量级**。

    真实体积仍以 Zenodo API 返回的 ``size`` 为准；登记值只用于离线预检与进度提示，
    因此允许是约数。
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "DATASETS",
    "ZENODO_API",
    "ZENODO_FILE_URL",
    "KiB",
    "MiB",
    "GiB",
    "get_dataset",
    "all_dataset_names",
    "requires_manual_download",
    "total_size_bytes",
    "total_size_gb",
]

ZENODO_API: Final[str] = "https://zenodo.org/api/records/{record}"
ZENODO_FILE_URL: Final[str] = "https://zenodo.org/records/{record}/files/{name}?download=1"

#: 二进制单位常量。用它书写体积，避免手写长数字数错位数。
KiB: Final[int] = 1024
MiB: Final[int] = 1024 * KiB
GiB: Final[int] = 1024 * MiB

#: 数据集登记表。键名与 ``configs/config.yaml`` 的 ``datasets`` 段保持一致。
#: 每个文件写成 ``(文件名, 体积字节, 公布的 md5)``；md5 留空表示"运行时向 API 查询"。
DATASETS: Final[dict[str, dict[str, Any]]] = {
    # ---------------------------------------------------------------------
    # PSKUS —— 主训练数据（真实医院环境，逐帧标注）
    # 11 个分片，正好按体积均衡分给 4 个人
    # ---------------------------------------------------------------------
    "pskuss": {
        "record": 4537209,
        "name": "PSKUS Hand Washing Video Dataset",
        "license": "CC-BY-SA-4.0",
        "home": "https://zenodo.org/records/4537209",
        "paper": "https://doi.org/10.3390/data6040038",
        "notes": "主数据集：真实医院环境、逐帧标注。约 17.1 GiB，按分片下载。",
        # 解压到 <dataset_dir>/extracted/，与 zip 分片分开；
        # 不要解压到 <dataset_dir>/pskuss/ —— 那样目录名会和数据集名重复，很绕。
        "extract_to": "extracted",
        #: configs/config.yaml 里 dataset.root 应当指向这个相对路径（相对仓库根）
        "config_root": "data/raw/pskuss/extracted",
        "files": (
            # 小 -> 大排序，便于"先下小的验证整条流水线"
            ("DataSet4.zip", 107 * MiB, ""),
            ("DataSet3.zip", 450 * MiB, ""),
            ("DataSet1.zip", 556 * MiB, ""),
            ("DataSet7.zip", 641 * MiB, ""),
            ("DataSet9.zip", 858 * MiB, ""),
            ("DataSet11.zip", 1076 * MiB, ""),
            ("DataSet10.zip", 1241 * MiB, ""),
            ("DataSet2.zip", 2162 * MiB, ""),
            ("DataSet5.zip", 2455 * MiB, ""),
            ("DataSet6.zip", 3529 * MiB, ""),
            ("DataSet8.zip", 4475 * MiB, ""),
            ("README.md", 8 * KiB, ""),
            ("statistics.csv", 300 * KiB, ""),
            ("summary.csv", 3 * KiB, ""),
        ),
        # 元数据类小文件：每个人都要下（不是视频，合计只有几百 KB）
        "always": ("README.md", "statistics.csv", "summary.csv"),
    },
    # ---------------------------------------------------------------------
    # METC —— 跨场景测试（实验室环境，72 名参与者）
    # 只有 3 个分片、约 2 GiB，建议每人都下全量
    # ---------------------------------------------------------------------
    "metc": {
        "record": 5808789,
        "name": "METC Hand Washing Video Dataset",
        "license": "CC-BY-4.0",
        "home": "https://zenodo.org/records/5808789",
        "paper": "https://doi.org/10.3390/data6040038",
        "notes": "跨场景测试集：实验室环境。约 2 GiB，建议全量下载。",
        "extract_to": "extracted",
        "config_root": "data/raw/metc/extracted",
        "files": (
            ("Interface_number_3.zip", 440 * MiB, ""),
            ("Interface_number_2.zip", 683 * MiB, ""),
            ("Interface_number_1.zip", 902 * MiB, ""),
            ("statistics.csv", 30 * KiB, ""),
            ("summary.csv", 3 * KiB, ""),
        ),
        "always": ("statistics.csv", "summary.csv"),
    },
    # ---------------------------------------------------------------------
    # Jurmala —— 可选扩展（约 15.8 GiB，作业不必要）
    # ---------------------------------------------------------------------
    "jurmala": {
        "record": 5808764,
        "name": "Jurmala Hand Washing Video Dataset",
        "license": "CC-BY-4.0",
        "home": "https://zenodo.org/records/5808764",
        "paper": "",
        "notes": "可选扩展数据（约 15.8 GiB）。时间与算力充足时再加，不是完成作业的必要条件。",
        "extract_to": "extracted",
        "config_root": "data/raw/jurmala/extracted",
        "files": (
            ("Location_2.zip", 13 * MiB, ""),
            ("Location_4.zip", 3508 * MiB, ""),
            ("Location_1.zip", 3641 * MiB, ""),
            ("Location_3.zip", 9005 * MiB, ""),
            ("statistics.csv", 300 * KiB, ""),
            ("summary.csv", 3 * KiB, ""),
        ),
        "always": ("statistics.csv", "summary.csv"),
    },
    # ---------------------------------------------------------------------
    # Kaggle —— 快速原型。Zenodo 没有，需要 Kaggle 账号或走 GitHub 镜像
    # ---------------------------------------------------------------------
    "kaggle": {
        "record": None,
        "name": "Kaggle Hand Wash Dataset (7-class)",
        "license": "见 Kaggle 数据集页条款",
        "home": "https://www.kaggle.com/datasets/realtimear/hand-wash-dataset",
        "mirror": "https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar",
        "paper": "",
        "notes": (
            "快速原型数据（约 300 MiB）。Zenodo 上没有，两条路："
            "(1) 装 kaggle CLI 并配置 API token，然后 "
            "`kaggle datasets download -d realtimear/hand-wash-dataset`；"
            "(2) 用 GitHub 镜像直链下载六分类整理版。下完解压到 data/raw/kaggle/。"
        ),
        "extract_to": "kaggle",
        "files": (),
        "always": (),
    },
    # ---------------------------------------------------------------------
    # 合成数据 —— 不需要下载，由代码生成（冒烟测试用）
    # ---------------------------------------------------------------------
    "synthetic": {
        "record": None,
        "name": "Synthetic hand-washing frames (generated locally)",
        "license": "本项目生成，无外部许可约束",
        "home": "",
        "paper": "https://doi.org/10.3390/jimaging11070208",
        "notes": (
            "不用下载：运行 `python scripts/train_model.py "
            "--config configs/experiments/smoke.yaml` 会自动生成。"
            "仅用于验证代码链路，指标不可写入报告。"
        ),
        "extract_to": "synthetic",
        "files": (),
        "always": (),
    },
}


def get_dataset(name: str) -> dict[str, Any]:
    """按名字取数据集登记项（大小写不敏感）。"""
    key = str(name).strip().lower()
    if key not in DATASETS:
        raise KeyError(
            f"未登记的数据集：{name!r}；已登记：{sorted(DATASETS)}。"
            "新增请改 src/handwash/data_sources.py 的 DATASETS 表。"
        )
    return DATASETS[key]


def all_dataset_names() -> list[str]:
    """返回全部已登记的数据集名（排序，输出稳定）。"""
    return sorted(DATASETS)


def requires_manual_download(name: str) -> bool:
    """是否为"无法脚本化下载"的数据集（没有 Zenodo 记录号）。"""
    return DATASETS[str(name).strip().lower()].get("record") is None


def iter_registered_files(name: str) -> list[tuple[str, int, str]]:
    """返回 ``[(文件名, 体积字节, md5), ...]``（单位已在登记表里统一为字节）。"""
    entry = get_dataset(name)
    out: list[tuple[str, int, str]] = []
    for row in entry.get("files", ()):
        key = str(row[0])
        size = int(row[1])
        md5 = str(row[2]) if len(row) > 2 and row[2] else ""
        out.append((key, size, md5))
    return out


def total_size_bytes(name: str) -> int:
    """登记的文件体积合计（字节）。"""
    return int(sum(size for _, size, _ in iter_registered_files(name)))


def total_size_gb(name: str) -> float:
    """登记的文件体积合计（GiB），仅用于提示；真实大小以 API 为准。"""
    return round(total_size_bytes(name) / GiB, 2)
