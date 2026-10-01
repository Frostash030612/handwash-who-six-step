"""配置系统契约测试（src/handwash/core/config.py）。

为什么这块最重要：小组项目里"结果对不上"的根因九成是配置被静默忽略 —— 有人把
``epochs`` 拼成 ``epochz``，或者把 ``split`` 比例写成 0.8/0.15/0.15，程序不报错，
跑出来的指标却不是同一套设置。本模块的测试就是钉死"拼错必炸、类型不符必炸、
越界必炸"。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from handwash.core.config import (
    CONFIG_SCHEMA_VERSION,
    AppConfig,
    ResolvedConfig,
    apply_overrides,
    dump_config,
    hash_config,
    load_config,
    merge_mappings,
    parse_overrides,
)
from handwash.errors import ConfigError, ConfigFileNotFoundError

pytestmark = pytest.mark.unit

#: 项目里真实存在的主配置文件。它由其他组员并发维护，因此只能"存在才测"。
REAL_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """按 ``"train.epochs"`` 形式写入深层键（用于构造非法配置）。"""
    cursor: dict[str, Any] = payload
    keys = dotted.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value
    return payload


# --- 深合并 / 覆盖解析 -------------------------------------------------------
def test_merge_mappings_deep_merges_and_overrides() -> None:
    """分层叠加配置时，子段必须逐键覆盖而不是整段替换。"""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overlay = {"a": {"b": 9}, "e": 5}
    assert merge_mappings(base, overlay) == {"a": {"b": 9, "c": 2}, "d": 3, "e": 5}


def test_merge_mappings_does_not_mutate_inputs() -> None:
    """原地修改入参会让"同一份默认配置"被上一个实验污染。"""
    base = {"a": {"b": 1}}
    overlay = {"a": {"b": 2}}
    merge_mappings(base, overlay)
    assert base == {"a": {"b": 1}}
    assert overlay == {"a": {"b": 2}}


def test_merge_mappings_scalar_replaces_mapping() -> None:
    """类型不同时以 overlay 为准，否则删不掉一个子段。"""
    assert merge_mappings({"a": {"b": 1}}, {"a": None}) == {"a": None}


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (["train.epochs=5"], {"train": {"epochs": 5}}),
        (["model.image_size=128", "runtime.run_name=abc"],
         {"model": {"image_size": 128}, "runtime": {"run_name": "abc"}}),
        (["runtime.run_name=null"], {"runtime": {"run_name": None}}),
        (["runtime.deterministic=false"], {"runtime": {"deterministic": False}}),
        (["train.lr=0.0003"], {"train": {"lr": 0.0003}}),
        (["eval.splits=[test]"], {"eval": {"splits": ["test"]}}),
    ],
)
def test_parse_overrides_restores_yaml_scalar_types(items: list[str], expected: dict[str, Any]) -> None:
    """CLI 传进来的永远是字符串，必须按 YAML 标量还原类型，否则 5 会变成 "5"。"""
    assert parse_overrides(items) == expected


def test_parse_overrides_keeps_int_and_null_types_exactly() -> None:
    """int 与 None 的还原必须精确：int("5") 静默转换会掩盖 "many" 这类错误。"""
    parsed = parse_overrides(["train.epochs=5", "runtime.run_name=null"])
    assert parsed["train"]["epochs"] == 5
    assert isinstance(parsed["train"]["epochs"], int)
    assert parsed["runtime"]["run_name"] is None


def test_parse_overrides_merges_repeated_prefixes() -> None:
    """同一前缀出现多次时必须合并成同一层，不能后写的把前写的整段顶掉。"""
    parsed = parse_overrides(["train.epochs=5", "train.batch_size=8"])
    assert parsed == {"train": {"epochs": 5, "batch_size": 8}}


@pytest.mark.parametrize("item", ["train.epochs", "just_a_key", "=5"])
def test_parse_overrides_rejects_malformed_item(item: str) -> None:
    """漏等号（或漏键名）的覆盖参数说明用户以为改了设置其实没改，必须直接报错。"""
    with pytest.raises(ConfigError):
        parse_overrides([item])


def test_apply_overrides_is_deep_merge() -> None:
    """CLI 覆盖作用在已合并的 dict 上，语义必须与 merge_mappings 一致。"""
    data = {"train": {"epochs": 20, "lr": 0.001}}
    assert apply_overrides(data, {"train": {"epochs": 3}}) == {"train": {"epochs": 3, "lr": 0.001}}


# --- 哈希 -------------------------------------------------------------------
def test_hash_config_is_stable_for_equal_payloads() -> None:
    """同一套参数即使键顺序不同，指纹也必须相同（否则"是否同一实验"无法判断）。"""
    first = hash_config({"a": 1, "b": {"c": 2}})
    second = hash_config({"b": {"c": 2}, "a": 1})
    assert first == second
    assert len(first) == 16
    assert all(ch in "0123456789abcdef" for ch in first)


def test_hash_config_differs_for_different_payloads() -> None:
    """改动任何一个参数都必须改变指纹，否则输出目录会互相覆盖。"""
    assert hash_config({"a": 1}) != hash_config({"a": 2})
    assert hash_config({"a": 1}) != hash_config({"a": 1, "b": 1})


def test_hash_config_handles_non_json_native_values() -> None:
    """配置里存在 tuple / Path 等类型，指纹计算必须能消化它们。"""
    assert hash_config({"t": (1, 2), "p": Path("outputs")}) == hash_config({"t": [1, 2], "p": Path("outputs")})


# --- load_config 正常路径 ---------------------------------------------------
def test_load_config_returns_resolved_config_with_expected_fields(config_file: Path) -> None:
    """ResolvedConfig 是 pipeline 的唯一入参，字段缺失会到处 AttributeError。"""
    resolved = load_config([config_file])
    assert isinstance(resolved, ResolvedConfig)
    assert resolved.config_hash and len(resolved.config_hash) == 16
    assert resolved.sources == (str(config_file),)
    assert resolved.label_space == "kaggle"
    assert isinstance(resolved.out_dir, Path)


def test_load_config_proxies_attribute_access(config_file: Path) -> None:
    """``rc.train.epochs`` 这类写法必须能用，否则调用方要写两层 rc.config.。"""
    resolved = load_config([config_file])
    assert resolved.train.epochs == 2
    assert resolved.runtime.seed == 7
    assert resolved.config.train.epochs == 2


def test_load_config_resolves_out_dir_under_configured_root(config_file: Path) -> None:
    """out_dir 必须落在配置指定的目录下，绝不能在仓库根乱建目录。"""
    resolved = load_config([config_file], overrides={"runtime": {"run_name": "exp1"}})
    assert resolved.out_dir.name == "exp1"
    assert resolved.out_dir.parent.name == "outputs"


def test_load_config_applies_overrides_after_files(config_file: Path) -> None:
    """覆盖参数的优先级必须高于配置文件，否则命令行调参毫无意义。"""
    resolved = load_config([config_file], overrides=parse_overrides(["train.epochs=9"]))
    assert resolved.train.epochs == 9
    assert resolved.overrides == {"train": {"epochs": 9}}


def test_load_config_merges_extra_paths_between_files_and_overrides(
    config_file: Path, write_config: Any, base_config_payload: dict[str, Any]
) -> None:
    """实验覆盖文件必须插在中间层：base < extra < CLI，否则实验不可复现。"""
    extra_payload = {"train": {"epochs": 5, "batch_size": 4}}
    extra = write_config(extra_payload, "experiment.yaml")
    resolved = load_config([config_file], extra_paths=[extra], overrides={"train": {"epochs": 9}})
    assert resolved.train.epochs == 9
    assert resolved.train.batch_size == 4
    assert resolved.sources == (str(config_file), str(extra))


def test_load_config_sources_are_recorded_for_traceability(config_file: Path) -> None:
    """报告里必须能回答"这次实验到底读了哪几个 yaml"。"""
    resolved = load_config([config_file])
    assert len(resolved.sources) == 1
    assert resolved.sources[0].endswith("config.yaml")


def test_load_config_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """配置路径写错是高频事故，必须给专门的异常类型与提示。"""
    with pytest.raises(ConfigFileNotFoundError, match="找不到配置文件"):
        load_config([tmp_path / "definitely_missing.yaml"])


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """顶层是列表的 YAML 无法当成配置，必须立刻报错而不是层层报 KeyError。"""
    bad = tmp_path / "list.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="顶层必须是 mapping"):
        load_config([bad])


def test_load_config_rejects_broken_yaml(tmp_path: Path) -> None:
    """YAML 语法错误要转成项目自己的异常，方便 CLI 统一处理退出码。"""
    bad = tmp_path / "broken.yaml"
    bad.write_text("train: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config([bad])


def test_load_config_defaults_to_repo_config_when_no_path_given() -> None:
    """不传路径时应该读 configs/config.yaml 而不是空配置。"""
    if not REAL_CONFIG.exists():
        pytest.skip(f"真实配置尚未落地：{REAL_CONFIG}")
    resolved = load_config()
    assert resolved.sources == (str(REAL_CONFIG),)
    assert resolved.train.epochs >= 1


@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="configs/config.yaml 尚未落地")
def test_real_repo_config_is_valid_and_matches_schema_version() -> None:
    """真实配置文件必须能通过严格校验，否则全组的实验都跑不起来。"""
    resolved = load_config([REAL_CONFIG])
    assert resolved.config.schema_version == CONFIG_SCHEMA_VERSION
    assert resolved.dataset.name in resolved.datasets
    assert resolved.label_space
    assert resolved.train.epochs >= 1


# --- 严格校验：未知键与类型 --------------------------------------------------
@pytest.mark.parametrize(
    "dotted",
    ["bogus_key", "train.epochz", "runtime.seedd", "assess.threshold", "model.image_sizes"],
)
def test_unknown_key_raises(
    config_file: Path, base_config_payload: dict[str, Any], write_config: Any, dotted: str
) -> None:
    """拼错的键绝不能被静默忽略 —— 这是"结果对不上"的头号根因。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), dotted, 1)
    path = write_config(payload, "unknown.yaml")
    with pytest.raises(ConfigError, match="未知配置键"):
        load_config([path])


def test_unknown_key_via_override_raises(config_file: Path) -> None:
    """CLI 覆盖同样走严格校验，不能因为来自命令行就放行。"""
    with pytest.raises(ConfigError, match="未知配置键"):
        load_config([config_file], overrides=parse_overrides(["train.epochz=3"]))


@pytest.mark.parametrize(
    "dotted",
    ["train.epochs", "train.batch_size", "runtime.seed", "runtime.num_workers", "assess.smooth_window"],
)
def test_non_numeric_int_field_raises(
    base_config_payload: dict[str, Any], write_config: Any, dotted: str
) -> None:
    """把 "many" 写进整数字段必须报错；能 int() 的字符串也不该被接受。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), dotted, "many")
    path = write_config(payload, "badtype.yaml")
    with pytest.raises(ConfigError):
        load_config([path])


def test_wrong_container_type_raises(base_config_payload: dict[str, Any], write_config: Any) -> None:
    """``eval.splits`` 必须是列表；写成字符串会让 for 循环逐字符迭代。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "eval.splits", "test")
    path = write_config(payload, "badlist.yaml")
    with pytest.raises(ConfigError, match="应为列表"):
        load_config([path])


def test_bool_field_accepts_yaml_true_false(base_config_payload: dict[str, Any], write_config: Any) -> None:
    """bool 字段必须接受 YAML 的 true/false，同时拒绝任意字符串。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "runtime.deterministic", "false")
    loaded = load_config([write_config(payload, "bool.yaml")])
    assert loaded.runtime.deterministic is False

    bad = _set_nested(copy.deepcopy(base_config_payload), "runtime.deterministic", "maybe")
    with pytest.raises(ConfigError, match="应为 bool"):
        load_config([write_config(bad, "bool_bad.yaml")])


def test_section_must_be_mapping(base_config_payload: dict[str, Any], write_config: Any) -> None:
    """把整段写成标量（``train: 3``）时必须报错，而不是构造出半截配置。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "train", 3)
    with pytest.raises(ConfigError):
        load_config([write_config(payload, "scalar_section.yaml")])


# --- 严格校验：取值范围与跨字段 ----------------------------------------------
@pytest.mark.parametrize("seed", [-1, -100])
def test_negative_seed_raises(base_config_payload: dict[str, Any], write_config: Any, seed: int) -> None:
    """负种子在 numpy 里会直接抛错，必须在配置阶段就拦住。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "runtime.seed", seed)
    with pytest.raises(ConfigError, match="seed"):
        load_config([write_config(payload, "seed.yaml")])


@pytest.mark.parametrize(
    ("train", "val", "test"),
    [(0.5, 0.2, 0.2), (0.8, 0.15, 0.15), (0.7, 0.15, 0.2)],
)
def test_split_ratios_must_sum_to_one(
    base_config_payload: dict[str, Any], write_config: Any, train: float, val: float, test: float
) -> None:
    """比例之和不是 1 会让一部分视频既不属于 train 也不属于 test，指标失去意义。"""
    payload = _set_nested(
        copy.deepcopy(base_config_payload), "split", {"train": train, "val": val, "test": test}
    )
    with pytest.raises(ConfigError, match="split 比例之和"):
        load_config([write_config(payload, "split.yaml")])


def test_split_ratio_exact_sum_is_accepted(base_config_payload: dict[str, Any], write_config: Any) -> None:
    """合法的比例必须通过（防止上面的校验写成"永远报错"）。"""
    payload = _set_nested(
        copy.deepcopy(base_config_payload), "split", {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    assert load_config([write_config(payload, "split_ok.yaml")]).split.train == 0.8


def test_dataset_name_must_exist_in_datasets(base_config_payload: dict[str, Any], write_config: Any) -> None:
    """``dataset.name`` 指向不存在的档案会让数据层拿到空根目录。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "dataset.name", "not_registered")
    with pytest.raises(ConfigError, match="未在 datasets 中定义"):
        load_config([write_config(payload, "ds.yaml")])


def test_explicit_empty_datasets_section_raises(
    base_config_payload: dict[str, Any], write_config: Any
) -> None:
    """**显式**把 datasets 写成空映射时必须拒绝启动。

    注意区分两种情况（这是刻意的设计，不要"顺手统一"掉）：
        * 完全不写 datasets   -> 用 AppConfig 的默认档案（配置自洽，可校验通过）；
        * 写了但为空 ``{}``   -> 视为"作者想清空"，直接报错，
          否则训练会在跑到一半时才发现没有数据路径。
    """
    payload = copy.deepcopy(base_config_payload)
    payload["datasets"] = {}
    with pytest.raises(ConfigError, match="datasets 段不能为空"):
        load_config([write_config(payload, "emptydatasets.yaml")])


@pytest.mark.parametrize("version", [0, CONFIG_SCHEMA_VERSION - 1, CONFIG_SCHEMA_VERSION + 1])
def test_schema_version_mismatch_raises(
    base_config_payload: dict[str, Any], write_config: Any, version: int
) -> None:
    """结构版本不一致说明配置文件与代码不是同一代，必须按迁移说明升级而非硬跑。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "schema_version", version)
    with pytest.raises(ConfigError, match="schema_version"):
        load_config([write_config(payload, "schema.yaml")])


@pytest.mark.parametrize("size", [200, 100, 512, 224 + 1])
def test_image_size_must_be_in_whitelist(
    base_config_payload: dict[str, Any], write_config: Any, size: int
) -> None:
    """非 32 倍数尺寸在 YOLO 系列上行为不稳定，因此只允许白名单取值。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "model.image_size", size)
    with pytest.raises(ConfigError, match="image_size"):
        load_config([write_config(payload, "size.yaml")])


@pytest.mark.parametrize("size", [128, 224, 320])
def test_image_size_whitelist_accepts_valid_values(
    base_config_payload: dict[str, Any], write_config: Any, size: int
) -> None:
    """白名单本身也要被验证，避免误把它收紧成空集。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "model.image_size", size)
    assert load_config([write_config(payload, "size_ok.yaml")]).model.image_size == size


@pytest.mark.parametrize("mode", ["bogus", "seconds2", ""])
def test_invalid_duration_check_raises(
    base_config_payload: dict[str, Any], write_config: Any, mode: str
) -> None:
    """时长判定口径只允许 seconds/ratio/none，写错会让判定逻辑走进未定义分支。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), "assess.duration_check", mode)
    with pytest.raises(ConfigError, match="duration_check"):
        load_config([write_config(payload, "dur.yaml")])


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("model.normalize", "bogus"),
        ("model.dropout", 1.0),
        ("train.precision", "int8"),
        ("train.mode", "video"),
        ("train.optimizer", "lion"),
        ("train.scheduler", "plateau"),
        ("runtime.tracking", "mlflow"),
        ("split.group_key", "frame_index"),
        ("model.temporal.kind", "transformer"),
        ("data.prep.fps", 0.0),
        ("data.prep.jpeg_quality", 101),
        ("assess.min_confidence", 1.5),
        ("assess.step_duration_ratio", 2.0),
        ("train.augment.crop_scale", [0.0, 1.0]),
    ],
)
def test_out_of_range_or_unknown_enum_values_raise(
    base_config_payload: dict[str, Any], write_config: Any, dotted: str, value: Any
) -> None:
    """所有枚举/区间字段都必须拒绝非法值，避免"配置生效但语义未定义"。"""
    payload = _set_nested(copy.deepcopy(base_config_payload), dotted, value)
    with pytest.raises(ConfigError):
        load_config([write_config(payload, "enum.yaml")])


# --- AppConfig 默认值 -------------------------------------------------------
def test_app_config_from_empty_mapping_uses_documented_defaults() -> None:
    """空 mapping 代表"全用默认值"，默认值指向当前 PSKUS 主线。"""
    config = AppConfig.from_mapping({})
    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.dataset.name == "pskuss"
    assert config.label_space_name() == "pskuss"
    assert config.model.image_size == 224
    assert config.split.train + config.split.val + config.split.test == pytest.approx(1.0)


def test_app_config_defaults_are_self_consistent_and_validatable() -> None:
    """默认构造必须**自洽**：``AppConfig().validate()`` 应当直接通过。

    这条不变量很重要 —— 测试、notebook、二次开发都会直接 `AppConfig()`，
    如果默认配置自身不合法，每个人都要先手工塞一个 datasets 段，
    很容易漏掉并得到难懂的报错。
    """
    config = AppConfig()
    config.validate()
    assert config.datasets, "默认应当带一套数据集档案"
    assert config.dataset.name in config.datasets
    assert config.label_space_name() == "pskuss"


def test_app_config_from_mapping_accepts_complete_payload(config_file: Path) -> None:
    """带 datasets 档案的完整 mapping 构造后必须能通过校验。"""
    resolved = load_config([config_file])
    resolved.config.validate()
    assert resolved.datasets["kaggle"]["root"] == "data/raw/kaggle"


def test_app_config_to_dict_is_plain_and_hashable_payload(config_file: Path) -> None:
    """to_dict 的结果必须能直接 json.dumps，用于落盘与指纹计算。"""
    payload = load_config([config_file]).config.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["dataset"]["label_space"] == "kaggle"


# --- 落盘与往返 -------------------------------------------------------------
def test_dump_config_writes_utf8_yaml_with_hash_and_full_config(
    config_file: Path, tmp_path: Path
) -> None:
    """每个 run 都必须有 resolved_config.yaml，用来回答"当时到底用了什么参数"。"""
    resolved = load_config([config_file])
    target = dump_config(resolved, tmp_path / "nested" / "resolved_config.yaml")
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert resolved.config_hash in text
    payload = yaml.safe_load(text)
    assert payload["config_hash"] == resolved.config_hash
    assert payload["config"]["train"]["epochs"] == 2
    assert payload["sources"] == [str(config_file)]


def test_dumped_config_round_trips_through_app_config(config_file: Path, tmp_path: Path) -> None:
    """落盘配置必须能被重新读回并通过校验，否则"复现实验"只是一句口号。"""
    resolved = load_config([config_file], overrides={"assess": {"allow_repeats": True}})
    target = dump_config(resolved, tmp_path / "resolved_config.yaml")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))

    restored = AppConfig.from_mapping(payload["config"])
    restored.validate()
    assert restored.to_dict() == resolved.config.to_dict()
    assert restored.assess.allow_repeats is True
    assert restored.train.epochs == resolved.train.epochs


def test_dump_config_writes_to_utf8_without_bom(config_file: Path, tmp_path: Path) -> None:
    """中文注释/名称必须原样保留：带 BOM 或转义会让审阅者看不懂。"""
    resolved = load_config([config_file])
    target = dump_config(resolved, tmp_path / "resolved.yaml")
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "handwash" in raw.decode("utf-8")
