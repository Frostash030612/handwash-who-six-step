# ============================================================================
# Makefile —— 团队统一入口。所有常用动作都必须在这里有一条命令。
# ----------------------------------------------------------------------------
# 规则（详见 CONTRIBUTING.md R7）：
#   * 新增工作流时必须补对应 target 与 .PHONY 声明。
#   * target 内部禁止写"只在你电脑上成立"的绝对路径。
#   * 命令统一走 scripts/ 下的入口脚本，而不是 `python -m handwash.cli`：
#     脚本自带导入路径引导（scripts/_bootstrap.py），因此**没做
#     `pip install -e .` 也能跑**，新组员 clone 下来就能用。
#     两种写法完全等价；装好包之后也可以直接用 `handwash <子命令>`。
#   * Windows 组员请用 `make`（Git Bash / MSYS2），或直接跑：
#       pwsh scripts/setup_windows.ps1      （首次初始化）
#       python scripts/<脚本>.py --help     （各条命令）
# ============================================================================

SHELL := /bin/bash
PY    := python
PKG   := src/handwash
CONFIG ?= configs/config.yaml

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
.PHONY: help
help:  ## 显示所有可用命令
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- 环境 ------------------------------------------------------------------
.PHONY: env
env:  ## 用 conda 创建/更新项目环境
	conda env update -f environment.yml --prune

.PHONY: install
install:  ## 以可编辑模式安装本包（含全部 extras）
	$(PY) -m pip install -e ".[all]"

.PHONY: doctor
doctor:  ## 自检环境、数据路径、配置合法性
	$(PY) -m handwash.cli doctor --config $(CONFIG)

.PHONY: setup
setup: install hooks  ## 新组员一键初始化（装包 + 装 git hooks）

# --- 质量门禁（CI 与本文件必须完全一致）------------------------------------
# 注意 `-p no:cacheprovider`：让 pytest 不写 .pytest_cache。
# 在"只允许写特定位置"的受限环境（企业策略、沙箱、部分 CI 容器）里，缓存写入会失败
# 并刷出 PytestCacheWarning，看起来像测试挂了其实没有。关掉后 `make test` 到处都干净。
PYTEST := $(PY) -m pytest -p no:cacheprovider
PYTEST_FAST := -m "not integration and not slow and not gpu"

.PHONY: lint
lint:  ## ruff 静态检查
	$(PY) -m ruff check .

.PHONY: format
format:  ## ruff 自动格式化 + 修复
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

.PHONY: typecheck
typecheck:  ## mypy 类型检查（当前只覆盖 core/ 与 io/）
	$(PY) -m mypy

.PHONY: archcheck
archcheck:  ## 分层依赖体检：禁止越层 import、禁止散落魔数
	$(PY) scripts/check_structure.py

.PHONY: configcheck
configcheck:  ## 配置契约检查：所有 configs/*.yaml 必须能严格加载
	$(PY) scripts/check_config.py

.PHONY: doccheck
doccheck:  ## 文档体检：双语配对 + 链接不断链
	$(PY) scripts/check_docs.py

.PHONY: docs-sync
docs-sync:  ## 补齐/更新双语文档的语言切换行（幂等）
	$(PY) scripts/sync_doc_locales.py

.PHONY: test
test:  ## 跑单元测试（秒级，不需要数据、不需要 GPU）
	$(PYTEST) $(PYTEST_FAST)

.PHONY: test-all
test-all:  ## 跑全部测试（需要数据集）
	$(PYTEST)

.PHONY: test-cov
test-cov:  ## 带覆盖率报告
	$(PYTEST) $(PYTEST_FAST) --cov --cov-report=term-missing

.PHONY: check
check: lint archcheck configcheck doccheck typecheck test  ## 提交前本地必跑（与 CI 完全一致）

.PHONY: hooks
hooks:  ## 安装 pre-commit 钩子
	$(PY) -m pre_commit install

# --- 数据 ------------------------------------------------------------------
.PHONY: data-kaggle
data-kaggle:  ## 准备 Kaggle 洗手数据集（快速原型；需先按 docs/DATA.md 下载）
	$(PY) scripts/prepare_data.py --config configs/data/kaggle.yaml

.PHONY: data-pskus
data-pskus:  ## 准备 PSKUS 主数据集（需先手工下载到 data/raw/pskuss）
	$(PY) scripts/prepare_data.py --config configs/data/pskuss.yaml

.PHONY: data-metc
data-metc:  ## 准备 METC 跨场景数据集
	$(PY) scripts/prepare_data.py --config configs/data/metc.yaml

.PHONY: data-inspect
data-inspect:  ## 只统计原始数据规模与标签分布，不写盘（全量抽帧前先跑这个）
	$(PY) scripts/prepare_data.py --inspect

.PHONY: manifest
manifest:  ## 只重建 manifest 与 train/val/test 划分
	$(PY) scripts/prepare_data.py --config $(CONFIG) --stage all

# --- 训练 / 评估 / 推理 ----------------------------------------------------
.PHONY: train
train:  ## 训练模型（默认 config）
	$(PY) scripts/train_model.py --config $(CONFIG)

.PHONY: train-smoke
train-smoke:  ## 冒烟训练：合成数据 + 2 个 epoch + CPU，验证代码链路（无需数据集）
	$(PY) scripts/train_model.py --config configs/experiments/smoke.yaml

.PHONY: evaluate
evaluate:  ## 评估某个 checkpoint（默认取 <out_dir>/models/best.pt）
	$(PY) scripts/evaluate_model.py --config $(CONFIG)

.PHONY: infer
infer:  ## 对单段视频做逐帧推理（VIDEO=<路径>）
	$(PY) scripts/run_inference.py --config $(CONFIG) --video $(VIDEO)

.PHONY: assess
assess:  ## 对单段视频做 WHO 完整性评估（VIDEO=<路径>）
	$(PY) scripts/run_assess.py --config $(CONFIG) --video $(VIDEO)

.PHONY: assess-folder
assess-folder:  ## 评估整个目录的自采视频
	$(PY) scripts/run_assess.py --config $(CONFIG) --folder data/external/self_recorded/full

# --- 清理 ------------------------------------------------------------------
.PHONY: clean
clean:  ## 清理所有缓存（不动数据与权重）
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage .tmp
	rm -rf pytest-cache-files-*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

.PHONY: clean-outputs
clean-outputs:  ## 清空 outputs/ 下的实验产物（危险：会删掉训练结果）
	@read -p "确定要删除 outputs/ 下所有实验产物吗？[y/N] " ok && [ "$$ok" = "y" ]
	find outputs -mindepth 1 -maxdepth 1 ! -name '.gitkeep' ! -name 'README.md' -exec rm -rf {} +
