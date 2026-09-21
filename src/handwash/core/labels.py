"""标签空间：WHO 六步洗手法的**唯一**权威定义与跨数据集映射。

为什么必须有这个文件（CONTRIBUTING.md R10）：
    四个公开数据集的命名空间完全不同（PSKUS 的 "Step 1"、Kaggle 的
    "Step1_water"、METC 的 "Step_1"…）。一旦各组员各自在代码里写字符串
    比较，训练标签和评估标签就会悄悄错位，指标全废。
    规则：**任何模块都不得再出现洗手步骤的字面量**，一律 import 本模块。

本模块属于 L1 契约层：只用标准库 + PyYAML。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from handwash.errors import ConfigError

__all__ = [
    "Step",
    "StepFamily",
    "CANONICAL_STEPS",
    "WASH_STEPS",
    "NON_WASH_STEPS",
    "STEP_ORDER",
    "STEP_ORDER_INDEX",
    "num_classes",
    "STEP_ZH",
    "STEP_EN",
    "LABEL_SPACES",
    "LabelSpace",
    "get_label_space",
    "canonicalize",
    "canonicalize_sequence",
    "to_space",
    "is_wash_step",
]


class StepFamily(str, Enum):
    """步骤大类：用于区分"洗手动作"与"洗手过程之外的辅助动作"。"""

    WASH = "wash"
    NON_WASH = "non_wash"


class Step(str, Enum):
    """WHO "How to Handwash" 六步 + 数据集中出现的非步骤类别。

    字符串值即**规范名**，全局唯一，落进 manifest / 报告 / 指标。
    """

    STEP_1 = "step_1_palm_to_palm"
    STEP_2 = "step_2_palm_over_dorsum"
    STEP_3 = "step_3_fingers_interlaced"
    STEP_4 = "step_4_backs_of_fingers"
    STEP_5 = "step_5_rotational_thumbs"
    STEP_6 = "step_6_rotational_fingertips"
    FAUCET_ON = "faucet_on"
    FAUCET_OFF = "faucet_off"
    WASHING_HANDS = "washing_hands"
    OTHER = "other"
    UNKNOWN = "unknown"

    @property
    def family(self) -> StepFamily:
        """该类别属于洗手动作还是非洗手动作。"""
        return StepFamily.WASH if self in WASH_STEP_SET else StepFamily.NON_WASH

    @property
    def order_index(self) -> int:
        """在 WHO 六步中的序号（1..6）；非六步类别返回 0。"""
        return STEP_ORDER_INDEX.get(self, 0)

    @property
    def is_who_step(self) -> bool:
        return self in STEP_ORDER_SET


# --- 集合常量 ---------------------------------------------------------------
CANONICAL_STEPS: Final[tuple[Step, ...]] = (
    Step.STEP_1,
    Step.STEP_2,
    Step.STEP_3,
    Step.STEP_4,
    Step.STEP_5,
    Step.STEP_6,
)

STEP_ORDER: Final[tuple[Step, ...]] = CANONICAL_STEPS
STEP_ORDER_SET: Final[frozenset[Step]] = frozenset(CANONICAL_STEPS)
STEP_ORDER_INDEX: Final[Mapping[Step, int]] = {s: i + 1 for i, s in enumerate(CANONICAL_STEPS)}

#: 真正"搓洗"的六步（术语上 non-wash 与 wash 的区分依据）
WASH_STEPS: Final[tuple[Step, ...]] = CANONICAL_STEPS
WASH_STEP_SET: Final[frozenset[Step]] = STEP_ORDER_SET

#: 洗手流程中的辅助动作，不参与"漏步"判定
NON_WASH_STEPS: Final[tuple[Step, ...]] = (
    Step.FAUCET_ON,
    Step.FAUCET_OFF,
    Step.WASHING_HANDS,
    Step.OTHER,
    Step.UNKNOWN,
)

STEP_ZH: Final[Mapping[Step, str]] = {
    Step.STEP_1: "掌心相对搓洗",
    Step.STEP_2: "手背对掌心搓洗",
    Step.STEP_3: "掌心相对十指交叉",
    Step.STEP_4: "指背对掌心互扣",
    Step.STEP_5: "拇指旋转搓洗",
    Step.STEP_6: "指尖在掌心旋转",
    Step.FAUCET_ON: "打开水龙头",
    Step.FAUCET_OFF: "关闭水龙头",
    Step.WASHING_HANDS: "过渡性洗手动作",
    Step.OTHER: "其他动作",
    Step.UNKNOWN: "未知/未识别",
}

STEP_EN: Final[Mapping[Step, str]] = {
    Step.STEP_1: "Palm to palm",
    Step.STEP_2: "Palm over dorsum",
    Step.STEP_3: "Fingers interlaced",
    Step.STEP_4: "Backs of fingers to opposing palms",
    Step.STEP_5: "Rotational rubbing of thumbs",
    Step.STEP_6: "Rotational rubbing of fingertips",
    Step.FAUCET_ON: "Faucet on",
    Step.FAUCET_OFF: "Faucet off",
    Step.WASHING_HANDS: "Washing hands (transitional)",
    Step.OTHER: "Other",
    Step.UNKNOWN: "Unknown",
}


def num_classes(*, include_non_wash: bool = True) -> int:
    """标签空间大小。"""
    return len(Step) if include_non_wash else len(CANONICAL_STEPS)


def is_wash_step(label: "Step | str") -> bool:
    """判断是否为 WHO 六步之一（接受 Step 或任意别名字符串）。"""
    try:
        return canonicalize(label).family is StepFamily.WASH
    except ConfigError:
        return False


# --- 数据集命名空间 ---------------------------------------------------------
#: 各数据集原始标签 -> 规范 Step 的映射。
#: 新增数据集时**只改这里**，不要在任何 pipeline 里写 if/else 判断数据集名。
_DATASET_ALIASES: Final[Mapping[str, Mapping[str, Step]]] = {
    # PSKUS（Zenodo 4537209）：逐帧标注，类名形如 "Step 1" / "Faucet on"
    "pskuss": {
        "step 1": Step.STEP_1,
        "step 2": Step.STEP_2,
        "step 3": Step.STEP_3,
        "step 4": Step.STEP_4,
        "step 5": Step.STEP_5,
        "step 6": Step.STEP_6,
        "faucet on": Step.FAUCET_ON,
        "faucet off": Step.FAUCET_OFF,
        "washing hands": Step.WASHING_HANDS,
        "other": Step.OTHER,
        "unknown": Step.UNKNOWN,
    },
    # METC（Zenodo 5808789）：帧级标签，命名略有差异
    "metc": {
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
        "faucet_on": Step.FAUCET_ON,
        "faucet_off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    # Kaggle realtimear/hand-wash-dataset：目录名形如 "Step1_water"
    # 说明：该数据集的 Step1-6 只覆盖搓洗动作，不含开关水龙头。
    "kaggle": {
        "step1_water": Step.STEP_1,
        "step2_water": Step.STEP_2,
        "step3_water": Step.STEP_3,
        "step4_water": Step.STEP_4,
        "step5_water": Step.STEP_5,
        "step6_water": Step.STEP_6,
        "step7_water": Step.OTHER,
        "step1": Step.STEP_1,
        "step2": Step.STEP_2,
        "step3": Step.STEP_3,
        "step4": Step.STEP_4,
        "step5": Step.STEP_5,
        "step6": Step.STEP_6,
        "step7": Step.OTHER,
        "not_washing": Step.OTHER,
        "nowashing": Step.OTHER,
    },
    # Jurmala（Zenodo 5808764）：与 PSKUS 同源采集规范
    "jurmala": {
        "step 1": Step.STEP_1,
        "step 2": Step.STEP_2,
        "step 3": Step.STEP_3,
        "step 4": Step.STEP_4,
        "step 5": Step.STEP_5,
        "step 6": Step.STEP_6,
        "faucet on": Step.FAUCET_ON,
        "faucet off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    # 组员自采视频：目录名直接用规范名或 stepN
    "selfrecorded": {
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
        "faucet_on": Step.FAUCET_ON,
        "faucet_off": Step.FAUCET_OFF,
        "other": Step.OTHER,
    },
    # 合成数据（synthetic-hand-washing）
    "synthetic": {
        "step_1": Step.STEP_1,
        "step_2": Step.STEP_2,
        "step_3": Step.STEP_3,
        "step_4": Step.STEP_4,
        "step_5": Step.STEP_5,
        "step_6": Step.STEP_6,
    },
}

#: 规范名本身（含中英文别名），任何命名空间都可识别
_CANONICAL_ALIASES: Final[dict[str, Step]] = {
    **{s.value: s for s in Step},
    **{s.value.replace("_", " "): s for s in Step},
    **{f"step{i}": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    **{f"step {i}": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    **{f"第{i}步": s for i, s in enumerate(CANONICAL_STEPS, start=1)},
    "palm to palm": Step.STEP_1,
    "palm over dorsum": Step.STEP_2,
    "fingers interlaced": Step.STEP_3,
    "backs of fingers": Step.STEP_4,
    "rotational thumbs": Step.STEP_5,
    "rotational fingertips": Step.STEP_6,
    "水龙头开": Step.FAUCET_ON,
    "水龙头关": Step.FAUCET_OFF,
    "其他": Step.OTHER,
    "未知": Step.UNKNOWN,
}


def _normalize(text: str) -> str:
    """归一化标签字符串：去空白、统一小写、下划线/连字符/空格等价。"""
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower()).strip()


class LabelSpace:
    """一个数据集（或一个模型输出）的标签命名空间。

    Parameters
    ----------
    name:
        命名空间名称，如 ``"kaggle"``。必须与 configs 中 ``dataset.name`` 一致。
    labels:
        按索引顺序排列的规范 Step。**顺序即模型输出通道顺序**，一旦冻结不得重排。
    aliases:
        命名空间内的额外别名，覆盖默认映射。
    """

    __slots__ = ("name", "_labels", "_index", "_aliases")

    def __init__(self, name: str, labels: Sequence[Step], aliases: Mapping[str, Step] | None = None) -> None:
        if not labels:
            raise ConfigError(f"标签空间 `{name}` 为空")
        if len(set(labels)) != len(labels):
            dup = [s.value for s in labels if list(labels).count(s) > 1]
            raise ConfigError(f"标签空间 `{name}` 存在重复类别：{sorted(set(dup))}")
        self.name = name
        self._labels: tuple[Step, ...] = tuple(labels)
        self._index: Mapping[Step, int] = {s: i for i, s in enumerate(self._labels)}
        self._aliases: Mapping[str, Step] = dict(aliases or {})

    # --- 基本属性 ---------------------------------------------------------
    @property
    def labels(self) -> tuple[Step, ...]:
        """按索引顺序返回全部类别（即 ``index -> label``）。"""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, item: object) -> bool:
        try:
            return self.canonicalize(item) in self._index  # type: ignore[arg-type]
        except ConfigError:
            return False

    def __iter__(self):
        return iter(self._labels)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"LabelSpace(name={self.name!r}, size={len(self._labels)})"

    # --- 转换 -------------------------------------------------------------
    def canonicalize(self, raw: object) -> Step:
        """把数据集的原始标签转成规范 ``Step``。

        Raises
        ------
        ConfigError
            无法识别时抛错（**不要**静默映射到 OTHER —— 那会污染指标）。
        """
        if isinstance(raw, Step):
            return raw
        key = _normalize(str(raw))
        # 三个候选键：原样归一化、去掉全部分隔符、去掉分隔符后补上下划线
        candidates = (key, key.replace(" ", ""), key.replace(" ", "_"))
        for table in (self._aliases, _DATASET_ALIASES.get(self.name, {}), _CANONICAL_ALIASES):
            for candidate in candidates:
                if candidate in table:
                    return table[candidate]
        raise ConfigError(
            f"标签空间 `{self.name}` 无法识别标签：{raw!r}",
            hint="请在 src/handwash/core/labels.py 的 _DATASET_ALIASES 中补充映射，而不是在业务代码里特判。",
        )

    def to_index(self, label: object) -> int:
        """规范标签 -> 模型输出通道下标。"""
        step = self.canonicalize(label)
        try:
            return self._index[step]
        except KeyError as exc:
            raise ConfigError(
                f"标签 `{step.value}` 不在标签空间 `{self.name}` 内",
                hint=f"该命名空间的类别为：{[s.value for s in self._labels]}",
            ) from exc

    def to_label(self, index: int) -> Step:
        """通道下标 -> 规范标签。"""
        if not 0 <= index < len(self._labels):
            raise ConfigError(
                f"类别下标越界：{index}（标签空间 `{self.name}` 共 {len(self._labels)} 类）"
            )
        return self._labels[index]

    def to_space(self, label: object, target: "LabelSpace") -> int:
        """把本命名空间的标签转到另一个命名空间的通道下标。

        用途：PSKUS 训练的模型 + Kaggle 评估脚本，两边类别数不同也能对齐。
        """
        step = self.canonicalize(label)
        if step not in target._index:
            raise ConfigError(
                f"标签 `{step.value}` 无法映射到标签空间 `{target.name}`",
                hint="跨数据集评估时请用 macro 指标并显式声明 target 空间缺失的类别为 N/A。",
            )
        return target._index[step]

    def indices_of(self, steps: Iterable[Step]) -> list[int]:
        """批量取下标，忽略不在本空间内的类别。"""
        return [self._index[s] for s in steps if s in self._index]


#: 预置命名空间。**索引顺序已冻结**：改动等于让所有旧 checkpoint 失效。
LABEL_SPACES: Final[Mapping[str, LabelSpace]] = {
    # 主实验：六步 + 开关龙头等辅助动作
    "pskuss": LabelSpace(
        "pskuss",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.WASHING_HANDS,
            Step.OTHER,
        ),
    ),
    "metc": LabelSpace(
        "metc",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.OTHER,
        ),
    ),
    # 快速原型：只有六步（Kaggle 的 step7/no-washing 归入 other 用途，不入此空间）
    "kaggle": LabelSpace("kaggle", CANONICAL_STEPS),
    "jurmala": LabelSpace("jurmala", CANONICAL_STEPS),
    "synthetic": LabelSpace("synthetic", CANONICAL_STEPS),
    "selfrecorded": LabelSpace(
        "selfrecorded",
        (
            Step.STEP_1,
            Step.STEP_2,
            Step.STEP_3,
            Step.STEP_4,
            Step.STEP_5,
            Step.STEP_6,
            Step.FAUCET_ON,
            Step.FAUCET_OFF,
            Step.OTHER,
        ),
    ),
}


def get_label_space(name: str) -> LabelSpace:
    """按名字取标签空间（大小写不敏感）。"""
    key = _normalize(name).replace(" ", "")
    if key in LABEL_SPACES:
        return LABEL_SPACES[key]
    raise ConfigError(
        f"未定义的标签空间：{name!r}",
        hint=f"可用：{sorted(LABEL_SPACES)}；新增请改 core/labels.py 并升 config 的 schema_version。",
    )


def canonicalize(raw: object, *, dataset: str = "selfrecorded") -> Step:
    """便捷函数：单标签 -> 规范 Step（默认按自采命名空间解析）。"""
    return get_label_space(dataset).canonicalize(raw)


def canonicalize_sequence(raws: Iterable[object], *, dataset: str = "selfrecorded") -> tuple[Step, ...]:
    """便捷函数：标签序列 -> 规范 Step 序列（逐帧预测结果常用）。"""
    space = get_label_space(dataset)
    return tuple(space.canonicalize(r) for r in raws)


def to_space(label: object, space: str, *, dataset: str = "selfrecorded") -> int:
    """便捷函数：标签 -> 指定命名空间的通道下标。"""
    return get_label_space(dataset).to_space(label, get_label_space(space))
