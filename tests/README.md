# 测试说明（tests/）

本目录是项目的质量门禁。目标只有一句话：**改了代码以后，能在 10 秒内知道自己有没有
把别人依赖的契约改坏**。

## 1. 怎么跑

在仓库根目录执行（`pyproject.toml` 里已配置 `testpaths = ["tests"]`）：

```bash
# 默认：只跑单元测试（不需要数据集、不需要 GPU、秒级完成）
python -m pytest -m "not integration and not slow and not gpu"

# 跑某个文件 / 某个用例
python -m pytest tests/unit/core/test_protocol.py
python -m pytest tests/unit/core/test_protocol.py -k "missing"

# 全量（需要数据集与 GPU，正式出结果前跑）
python -m pytest

# 带覆盖率（合约层覆盖率应保持在较高水平）
python -m pytest -m "not integration and not slow and not gpu" --cov --cov-report=term-missing
```

等价快捷方式：`make test`、`make test-all`、`make test-cov`、`make check`。

## 2. 标记（marker）的含义

标记在 `pyproject.toml` 的 `[tool.pytest.ini_options].markers` 中注册，写错标记名
会因为 `--strict-markers` 直接失败（这是故意的，避免标记失去意义）。

| 标记 | 含义 | 依赖 |
| --- | --- | --- |
| `unit` | 纯函数/契约测试：无 IO、无数据集、无 GPU，必须秒级完成 | 只用 numpy / PyYAML / 标准库 |
| `integration` | 需要真实数据集，设置 `HANDWASH_DATA_ROOT` 后才有意义 | 数据集 |
| `slow` | 训练或推理级别，默认不跑 | 数分钟到数小时 |
| `gpu` | 需要 CUDA | GPU |

**规则：本目录下每个测试模块都必须在模块级写 `pytestmark = pytest.mark.unit`（或给每个
用例加 `@pytest.mark.unit`）。** 标记为 `unit` 的用例一旦需要数据集或 GPU，就等于把
"秒级反馈"这条最重要的开发体验毁掉了。

## 3. 目录结构

目录必须**镜像 `src/handwash/` 的分层**：

```
tests/
  conftest.py                 公共 fixture（临时配置、逐帧序列构造等）
  unit/
    core/                     对应 src/handwash/core/
      test_labels.py          标签空间与跨数据集映射
      test_config.py          YAML -> 冻结 dataclass 的严格校验
      test_metrics.py         评估指标口径
      test_protocol.py        WHO 六步完整性判定（业务核心）
      test_schema.py          跨层 DTO 的序列化往返
      test_registry.py        名字 -> 实现 的注册表
    test_paths.py             对应 src/handwash/paths.py
    test_errors.py            对应 src/handwash/errors.py
    test_logging.py           对应 src/handwash/logging.py
```

所有目录都有 `__init__.py`，这样不同子目录下的同名测试模块（例如将来
`tests/integration/test_paths.py`）不会互相覆盖。

## 4. 硬性规则

1. **新增模块必须同步新增测试。** 这是"改了契约没人知道"的唯一防线：没有测试的
   公开函数等于没有契约。评审时先看 `src/` 的 diff 里有没有对应的 `tests/` 变化。
2. **测试不允许写仓库里的任何目录。** 一切写操作走 `tmp_path` / `tmp_path_factory`；
   `PROJECT_ROOT` 下的 `outputs/`、`data/` 是组员共享的实验产物，被测试污染后极难排查。
3. **不依赖真实 `configs/` 文件。** 那些文件由其他组员并发维护，测试里自备最小 YAML；
   需要验证真实配置的用例用 `if not path.exists(): pytest.skip(...)` 兜住。
4. **需要随机性的用例必须显式固定种子**（`np.random.default_rng(seed)` / `seed=` 参数），
   禁止依赖全局随机状态。
5. **不使用 `assert` 做输入校验以外的用途**，也不要 mock 掉被测试的纯函数本身。
6. **中文注释与文档字符串，英文标识符。** 不使用 emoji。
7. **不要在测试里 `sys.path.append`。** 包通过 `pyproject.toml` 的
   `package-dir = {"" = "src"}` 安装（`pip install -e .`）；`tests/conftest.py`
   里的兜底插入只为"刚 clone 还没装包"的场景服务。

## 5. 契约层测试为什么写得这么细

`src/handwash/core/` 的每个公开结构都是全组共用的契约：

* 标签空间的**通道顺序**变了 -> 所有已训练的 checkpoint 与已发布的指标全部作废；
* 配置的**未知键被静默忽略** -> 两个人用"看起来一样"的参数跑出不同结果；
* 指标的**缺失类别口径**不一致 -> 论文里的 macro-F1 无法互相比较；
* 完整性判定的**阈值口径**不一致 -> 同一个视频两个人给出不同结论。

所以这些用例的断言普遍偏"死"（精确数值、精确顺序），这是刻意的：它们保护的是
**不可变的契约**，而不是实现细节。若确实要修改契约，请先走 `CONTRIBUTING.md` 的
RFC 流程，再同步改测试与 `CHANGELOG.md`。
