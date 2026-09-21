"""异常体系契约测试（src/handwash/errors.py）。

规则（CONTRIBUTING R9）：库代码禁止裸 ``raise Exception`` / ``sys.exit`` / ``assert``
做输入校验，可预期失败一律抛这里的子类。测试要保证三件事：层次结构统一、提示信息
真的出现在 ``str(exc)`` 里、退出码区分得开 —— 否则 CLI 只能把所有错误都变成 "1"。
"""

from __future__ import annotations

import pytest

from handwash import errors
from handwash.errors import (
    BackendUnavailableError,
    CheckpointError,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    DataError,
    DataLeakageError,
    DatasetNotFoundError,
    EvaluationError,
    HandwashError,
    ManifestError,
    ModelError,
    ModelNotFoundError,
    ProtocolError,
    TrainingError,
    VideoDecodeError,
)

pytestmark = pytest.mark.unit


def test_every_public_error_subclasses_handwash_error() -> None:
    """最外层 CLI 只捕获 HandwashError；漏一个就会漏出难以理解的 traceback。"""
    for name in errors.__all__:
        obj = getattr(errors, name)
        assert issubclass(obj, HandwashError), name


def test_base_error_str_returns_message_without_hint() -> None:
    """没有建议时不能多出一个空行，报告里会很丑。"""
    exc = HandwashError("出了点问题")
    assert str(exc) == "出了点问题"
    assert exc.message == "出了点问题"
    assert exc.hint is None


def test_base_error_str_includes_hint_when_given() -> None:
    """提示必须出现在 str(exc) 里：组员看到的就是这一行，而不是去读源码。"""
    exc = HandwashError("配置有问题", hint="检查 train.epochs")
    text = str(exc)
    assert "配置有问题" in text
    assert "检查 train.epochs" in text
    assert exc.hint == "检查 train.epochs"


def test_error_exit_codes_are_distinct_per_layer() -> None:
    """退出码区分层级，CI 与脚本才能按类型决定是否重试/告警。"""
    assert HandwashError.exit_code == 1
    assert ConfigError.exit_code == 2
    assert DataError.exit_code == 3
    assert ModelError.exit_code == 4
    assert TrainingError.exit_code == 5
    assert EvaluationError.exit_code == 5
    assert ProtocolError.exit_code == 6


def test_exit_codes_are_inherited_by_subclasses() -> None:
    """子类不重复声明退出码，必须从父类继承到同一个码。"""
    assert ConfigFileNotFoundError.exit_code == ConfigError.exit_code == 2
    assert ConfigKeyError.exit_code == 2
    assert DatasetNotFoundError.exit_code == 3
    assert VideoDecodeError.exit_code == 3
    assert CheckpointError.exit_code == ModelError.exit_code == 4


@pytest.mark.parametrize(
    ("cls", "parent"),
    [
        (ConfigFileNotFoundError, ConfigError),
        (ConfigKeyError, ConfigError),
        (DatasetNotFoundError, DataError),
        (VideoDecodeError, DataError),
        (ManifestError, DataError),
        (DataLeakageError, DataError),
        (ModelNotFoundError, ModelError),
        (CheckpointError, ModelError),
        (BackendUnavailableError, ModelError),
    ],
)
def test_specific_errors_extend_the_right_family(cls: type, parent: type) -> None:
    """按族捕获（except DataError）是数据层脚本的常见写法，族关系不能错。"""
    assert issubclass(cls, parent)
    assert issubclass(parent, HandwashError)


def test_config_file_not_found_records_the_path() -> None:
    """报错必须包含"哪个文件"，否则一晚上都在猜路径拼错了哪里。"""
    exc = ConfigFileNotFoundError("configs/nope.yaml")
    assert "configs/nope.yaml" in str(exc)
    assert exc.path.name == "nope.yaml"
    assert exc.hint


def test_config_key_error_reports_expected_and_actual() -> None:
    """类型错误必须同时给出"期望什么、实际什么"，这是最快定位配置问题的方式。"""
    exc = ConfigKeyError("epochs", expected="int", got="many", where="AppConfig.train")
    text = str(exc)
    assert "epochs" in text
    assert "int" in text
    assert "many" in text
    assert "AppConfig.train" in text
    assert exc.key == "epochs"


def test_dataset_not_found_message_can_include_expected_location() -> None:
    """告诉使用者期望位置，才能判断是没下载还是放错目录。"""
    exc = DatasetNotFoundError("pskuss", expected_at="data/raw/pskuss")
    assert "pskuss" in str(exc)
    assert "data/raw/pskuss" in str(exc)
    assert exc.name == "pskuss"


def test_dataset_not_found_without_location_still_readable() -> None:
    """不传期望位置时不能留下 "期望位置：None"。"""
    exc = DatasetNotFoundError("metc")
    assert "None" not in str(exc)


def test_video_decode_error_records_path_and_reason() -> None:
    """视频解码失败最常见的原因是编码/文件损坏，两者都要写进消息。"""
    exc = VideoDecodeError("data/raw/demo.mp4", "moov atom not found")
    text = str(exc)
    assert "demo.mp4" in text
    assert "moov atom not found" in text


def test_data_leakage_error_previews_clip_ids() -> None:
    """防数据泄漏：报错要列出越界出现的视频，方便立刻修正划分。"""
    exc = DataLeakageError(["clip_c", "clip_a", "clip_b"])
    text = str(exc)
    assert "数据泄漏" in text
    assert "clip_a" in text
    assert exc.clip_ids == ["clip_c", "clip_a", "clip_b"]


def test_data_leakage_error_truncates_long_list_but_reports_count() -> None:
    """泄漏几百段时消息要截断（否则日志刷屏），但必须给出总数。"""
    clip_ids = [f"clip_{i:03d}" for i in range(12)]
    exc = DataLeakageError(clip_ids)
    text = str(exc)
    assert "12 段" in text
    assert "clip_011" not in text


def test_model_not_found_lists_available_names_sorted() -> None:
    """可用名字必须排序，方便和配置里的拼写逐字对照。"""
    exc = ModelNotFoundError("yolo26x-cls", available=["yolo26n-cls", "baseline-cnn"])
    text = str(exc)
    assert "yolo26x-cls" in text
    assert "baseline-cnn, yolo26n-cls" in text
    assert exc.name == "yolo26x-cls"


def test_model_not_found_without_available_names_is_still_clean() -> None:
    """没有已注册模型时不能留下空的"已注册："。"""
    exc = ModelNotFoundError("nothing")
    assert "已注册" not in str(exc)


def test_backend_unavailable_error_carries_install_hint() -> None:
    """可选后端缺失要让使用者知道装哪一行依赖。"""
    exc = BackendUnavailableError("torch", "pip install -e \".[torch]\"")
    text = str(exc)
    assert "torch" in text
    assert ".[torch]" in text
    assert exc.backend == "torch"
    assert exc.exit_code == ModelError.exit_code


def test_protocol_error_is_catchable_as_handwash_error() -> None:
    """业务判定错误同样要能被 CLI 统一处理成友好的退出码 6。"""
    exc = ProtocolError("标签序列为空，无法评估")
    assert isinstance(exc, HandwashError)
    assert exc.exit_code == 6


def test_simple_family_errors_have_usable_defaults() -> None:
    """ManifestError / TrainingError / EvaluationError 直接用基类的构造签名。"""
    assert str(ManifestError("manifest 缺 column 列")) == "manifest 缺 column 列"
    assert TrainingError("loss 变成 nan", hint="降低学习率").hint == "降低学习率"
    assert EvaluationError("y_true 与 y_pred 长度不一致").message == "y_true 与 y_pred 长度不一致"


def test_errors_are_raisable_and_catchable_by_base_class() -> None:
    """典型调用点：顶层只写 ``except HandwashError``，因此继承链必须成立。"""
    with pytest.raises(HandwashError):
        raise DataLeakageError(["clip_a"])
    try:
        raise ConfigError("坏配置", hint="看看 docs/CONFIG.md")
    except HandwashError as exc:
        assert "docs/CONFIG.md" in str(exc)
