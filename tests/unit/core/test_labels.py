"""标签空间契约测试（src/handwash/core/labels.py）。

这些用例存在的意义：四份公开数据集的目录名/类名各不相同，一旦"哪个字符串等于
哪一步"这件事在不同模块里各写一遍，训练标签与评估标签就会静默错位，所有指标
全部失效且难以察觉。所以这里把映射、通道顺序、跨空间转换全部钉死。
"""

from __future__ import annotations

import pytest

from handwash.core.labels import (
    CANONICAL_STEPS,
    LABEL_SPACES,
    NON_WASH_STEPS,
    STEP_EN,
    STEP_ORDER,
    STEP_ORDER_INDEX,
    STEP_ZH,
    LabelSpace,
    Step,
    StepFamily,
    canonicalize,
    canonicalize_sequence,
    get_label_space,
    is_wash_step,
    num_classes,
    to_space,
)
from handwash.errors import ConfigError

pytestmark = pytest.mark.unit


# --- 类目定义 ---------------------------------------------------------------
def test_step_is_string_enum() -> None:
    """Step 必须是 str-Enum：否则 JSON/YAML 落盘会变成不可读的枚举对象。"""
    assert issubclass(Step, str)
    assert Step.STEP_1 == "step_1_palm_to_palm"
    assert f"{Step.STEP_6.value}" == "step_6_rotational_fingertips"


def test_step_members_are_exactly_expected() -> None:
    """成员集合是全局契约：少一个会让训练通道错位，多一个会让旧权重失效。"""
    assert [s.name for s in Step] == [
        "STEP_1",
        "STEP_2",
        "STEP_3",
        "STEP_4",
        "STEP_5",
        "STEP_6",
        "FAUCET_ON",
        "FAUCET_OFF",
        "WASHING_HANDS",
        "OTHER",
        "UNKNOWN",
    ]


def test_canonical_steps_are_six_who_steps_in_order() -> None:
    """CANONICAL_STEPS 的顺序就是 WHO 建议的执行顺序，不可重排。"""
    assert CANONICAL_STEPS == (
        Step.STEP_1,
        Step.STEP_2,
        Step.STEP_3,
        Step.STEP_4,
        Step.STEP_5,
        Step.STEP_6,
    )
    assert STEP_ORDER == CANONICAL_STEPS
    assert [s.order_index for s in CANONICAL_STEPS] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (Step.STEP_1, StepFamily.WASH),
        (Step.STEP_6, StepFamily.WASH),
        (Step.FAUCET_ON, StepFamily.NON_WASH),
        (Step.FAUCET_OFF, StepFamily.NON_WASH),
        (Step.WASHING_HANDS, StepFamily.NON_WASH),
        (Step.OTHER, StepFamily.NON_WASH),
        (Step.UNKNOWN, StepFamily.NON_WASH),
    ],
)
def test_family_separates_wash_from_auxiliary(label: Step, expected: StepFamily) -> None:
    """辅助动作（开关水龙头等）不得参与漏步判定，所以 family 必须泾渭分明。"""
    assert label.family is expected


def test_non_wash_steps_cover_everything_outside_canonical() -> None:
    """NON_WASH_STEPS 与 CANONICAL_STEPS 必须无重叠、无遗漏地覆盖 Step。"""
    assert set(CANONICAL_STEPS).isdisjoint(NON_WASH_STEPS)
    assert set(CANONICAL_STEPS) | set(NON_WASH_STEPS) == set(Step)


@pytest.mark.parametrize("label", list(Step))
def test_chinese_and_english_names_exist_for_every_label(label: Step) -> None:
    """报告要直接展示中文名；缺一个键就会在演示时 KeyError。"""
    assert STEP_ZH[label].strip()
    assert STEP_EN[label].strip()


@pytest.mark.parametrize(
    ("label", "index"),
    [(Step.STEP_1, 1), (Step.STEP_3, 3), (Step.STEP_6, 6), (Step.FAUCET_ON, 0), (Step.UNKNOWN, 0)],
)
def test_order_index(label: Step, index: int) -> None:
    """非六步类别统一返回 0，避免调用方写成 if label == 'other' 之类的特判。"""
    assert label.order_index == index
    assert label.is_who_step == (index > 0)
    # STEP_ORDER_INDEX 只收录六步，查非六步时应由 order_index 的默认值兜底
    assert STEP_ORDER_INDEX.get(label, 0) == index


def test_num_classes_counts_who_steps_only_when_requested() -> None:
    """模型输出通道数 = 标签空间大小，两种口径都必须稳定。"""
    assert num_classes() == len(Step) == 11
    assert num_classes(include_non_wash=False) == 6


def test_is_wash_step_accepts_aliases_and_rejects_garbage() -> None:
    """is_wash_step 用于数据清洗，遇到无法识别的字符串必须返回 False 而非抛错。"""
    assert is_wash_step(Step.STEP_2) is True
    assert is_wash_step("Step 2") is True
    assert is_wash_step("faucet_on") is False
    assert is_wash_step("完全不是标签") is False


# --- 标签空间 ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("space", "size"),
    [("kaggle", 6), ("jurmala", 6), ("synthetic", 6), ("pskuss", 10), ("metc", 9), ("selfrecorded", 9)],
)
def test_label_space_sizes_are_frozen(space: str, size: int) -> None:
    """类别数与顺序一旦改动，所有已训练的 checkpoint 与已发布的指标都作废。"""
    assert len(get_label_space(space)) == size
    assert get_label_space(space).name == space


def test_kaggle_space_is_canonical_six_in_order() -> None:
    """kaggle 快速原型空间的通道顺序必须等于 WHO 步骤顺序。"""
    space = get_label_space("kaggle")
    assert space.labels == CANONICAL_STEPS
    assert space.indices_of(CANONICAL_STEPS) == [0, 1, 2, 3, 4, 5]


def test_label_spaces_mapping_is_consistent() -> None:
    """字典键必须与 LabelSpace.name 相同，否则 get_label_space 会返回错的对象。"""
    for name, space in LABEL_SPACES.items():
        assert space.name == name
        assert len(set(space.labels)) == len(space.labels)


def test_label_space_rejects_empty_and_duplicate_labels() -> None:
    """空空间/重复类别属于配置错误，必须当场炸掉而不是留到训练时才崩。"""
    with pytest.raises(ConfigError, match="为空"):
        LabelSpace("bad", ())
    with pytest.raises(ConfigError, match="重复类别"):
        LabelSpace("dup", (Step.STEP_1, Step.STEP_1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Step1_water", Step.STEP_1),
        ("Step 5", Step.STEP_5),
        ("step_5", Step.STEP_5),
        ("step5", Step.STEP_5),
        ("第5步", Step.STEP_5),
        ("STEP 5", Step.STEP_5),
        ("  step-5  ", Step.STEP_5),
        ("step6_water", Step.STEP_6),
    ],
)
def test_kaggle_space_canonicalizes_common_aliases(raw: str, expected: Step) -> None:
    """归一化必须容忍大小写、空格、下划线、连字符与中文写法。"""
    assert get_label_space("kaggle").canonicalize(raw) == expected


@pytest.mark.parametrize("step_no", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("template", ["Step {n}", "step_{n}", "step{n}", "第{n}步"])
def test_all_alias_templates_map_to_same_step(step_no: int, template: str) -> None:
    """四种常见命名模板必须指向同一步：PSKUS/Kaggle/METC/自采视频各用一种。"""
    expected = CANONICAL_STEPS[step_no - 1]
    assert get_label_space("selfrecorded").canonicalize(template.format(n=step_no)) == expected


def test_unknown_label_raises_instead_of_silently_becoming_other() -> None:
    """静默映射到 other 会污染指标，因此必须显式报错并提示去哪里补映射。"""
    with pytest.raises(ConfigError, match="无法识别标签"):
        get_label_space("kaggle").canonicalize("this_is_not_a_label")


def test_canonicalize_accepts_step_instance_unchanged() -> None:
    """已经是 Step 的对象必须原样返回，避免二次解析引入歧义。"""
    assert get_label_space("kaggle").canonicalize(Step.STEP_4) is Step.STEP_4


def test_step_in_space_uses_canonicalization() -> None:
    """``in`` 运算符要能直接用数据集原始字符串判断，否则调用方会自己写映射。"""
    space = get_label_space("metc")
    assert "Step_3" in space
    assert Step.STEP_3 in space
    assert Step.UNKNOWN not in space
    assert "not_a_label" not in space


def test_space_iteration_and_length_agree_with_labels() -> None:
    """迭代顺序即通道顺序，任何实现都必须一致。"""
    space = get_label_space("pskuss")
    assert list(space) == list(space.labels)
    assert len(space) == len(space.labels) == 10


@pytest.mark.parametrize("bad_index", [-1, 10, 999])
def test_to_label_rejects_out_of_range_index(bad_index: int) -> None:
    """越界下标如果被静默裁剪，会把预测结果悄悄指向错误的步骤。"""
    with pytest.raises(ConfigError, match="下标越界"):
        get_label_space("pskuss").to_label(bad_index)


def test_to_index_and_to_label_are_inverse() -> None:
    """通道下标与标签必须互为反函数，这是训练/推理对齐的基础。"""
    space = get_label_space("metc")
    for index, step in enumerate(space.labels):
        assert space.to_index(step) == index
        assert space.to_label(index) == step


def test_to_index_rejects_label_missing_from_space() -> None:
    """kaggle 空间没有 washing_hands —— 必须报错并列出可用类别。"""
    with pytest.raises(ConfigError, match="不在标签空间"):
        get_label_space("kaggle").to_index(Step.WASHING_HANDS)


def test_to_space_maps_shared_classes_across_datasets() -> None:
    """跨数据集评估时要把标签翻译成目标空间的通道下标。"""
    pskuss = get_label_space("pskuss")
    metc = get_label_space("metc")
    assert pskuss.to_space(Step.STEP_1, metc) == 0
    assert pskuss.to_space(Step.FAUCET_OFF, metc) == 7


def test_to_space_raises_when_target_space_lacks_class() -> None:
    """目标空间缺失的类别不能被凑成 other，否则跨场景对比会虚高。"""
    with pytest.raises(ConfigError, match="无法映射"):
        get_label_space("pskuss").to_space(Step.WASHING_HANDS, get_label_space("metc"))
    with pytest.raises(ConfigError, match="无法映射"):
        get_label_space("metc").to_space(Step.FAUCET_ON, get_label_space("kaggle"))


def test_indices_of_ignores_classes_outside_space() -> None:
    """批量取下标时跳过不属于本空间的类别，用于"只评估共有类别"。"""
    pskuss = get_label_space("pskuss")
    assert pskuss.indices_of([Step.STEP_1, Step.WASHING_HANDS, Step.STEP_6]) == [0, 8, 5]
    assert get_label_space("kaggle").indices_of([Step.STEP_1, Step.WASHING_HANDS]) == [0]


def test_get_label_space_is_case_insensitive_and_lists_options_on_error() -> None:
    """名字来自配置文件，大小写不该成为翻车原因；拼错时要提示可用值。"""
    assert get_label_space("KaGgLe") is get_label_space("kaggle")
    with pytest.raises(ConfigError, match="未定义的标签空间"):
        get_label_space("不存在的数据集")


def test_module_level_canonicalize_uses_selfrecorded_namespace_by_default() -> None:
    """默认命名空间覆盖组员自采视频的命名习惯。"""
    assert canonicalize("step_2") == Step.STEP_2
    assert canonicalize("faucet_on", dataset="metc") == Step.FAUCET_ON


def test_canonicalize_sequence_returns_tuple_of_steps() -> None:
    """逐帧预测结果常以字符串列表出现，必须能整体转换且保持顺序。"""
    converted = canonicalize_sequence(["step_1", "Step 2", "第3步"])
    assert converted == (Step.STEP_1, Step.STEP_2, Step.STEP_3)
    assert isinstance(converted, tuple)


def test_canonicalize_sequence_propagates_unknown_label_error() -> None:
    """序列里混入脏标签必须整批失败，不能只丢一帧。"""
    with pytest.raises(ConfigError, match="无法识别标签"):
        canonicalize_sequence(["step_1", "???"])


def test_module_level_to_space_helper() -> None:
    """便捷函数与 LabelSpace.to_space 必须给出同样的结果。"""
    assert to_space("Step 1", "metc", dataset="pskuss") == 0
    assert to_space(Step.STEP_6, "kaggle") == 5
