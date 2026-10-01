> [English](/docs/RULES_CARD.md) | **中文**

# 快速上手与修改规则速查卡

> 这一页是 [`CONTRIBUTING.md`](/docs/zh/CONTRIBUTING.md) 的压缩版。
> **卡片和正文冲突时以正文为准**；卡片忘记更新属于文档缺陷，请提 PR 补上。
> 建议打印出来贴在桌上。

---

## 一、三条命令起步

```bash
conda env create -f environment.yml && conda activate handwash
pip install -e ".[all]" && python -m pre_commit install
python scripts/doctor.py                 # 自检；全绿后再往下
python scripts/train_model.py --config configs/experiments/smoke.yaml   # 30 秒冒烟
```

> 没做 `pip install -e .` 也能跑：`scripts/` 下的脚本自带导入路径引导。

---

## 二、我该改哪一级？（第一个问题永远是这个）

| 我要动的 | 级别 | 需要什么 |
| --- | --- | --- |
| 新建实验配置、跑实验、记录结果 | **L3 自由** | 只要 CI 通过 |
| 新建模型文件 / 数据集适配器 / 脚本 | **L3 自由** | 只要 CI 通过 |
| 修改已有函数的**实现** | **L2 受限** | 1 名负责人 review |
| 修改 `tests/`、`docs/`、`configs/experiments/` | **L2/L3** | 通常自由 |
| 改**函数签名 / 返回结构 / 配置键名 / 判定规则** | **L1 冻结** | **先写 RFC**，后改代码 |
| 改 `core/` 下任何东西的语义、改标签空间顺序、改划分规则 | **L1 冻结** | **先写 RFC**，后改代码 |

**口诀：加文件是 L3，改实现是 L2，改接口是 L1。**

L1 文件清单（动它们之前先停一下）：
```
src/handwash/core/labels.py  schema.py  config.py  protocol.py  registry.py
src/handwash/errors.py  paths.py  logging.py
src/handwash/io/split.py
configs/config.yaml
```

---

## 三、十条铁律

| # | 规则 |
| --- | --- |
| 1 | 只改自己负责的模块；跨模块先开 issue 说一声 |
| 2 | UTF-8 + LF + 4 空格（YAML/JSON 2 空格） |
| 3 | 可调参数写进 `configs/`，代码里不留魔数 |
| 4 | 新依赖同时改 `pyproject.toml` **和** `environment.yml` |
| 5 | `core` 不 import torch；`io` 不 import models（分层依赖） |
| 6 | 数据、训练权重、输出不进 Git；根目录 `exp.pt` 是演示例外 |
| 7 | 新工作流同时加进 `Makefile` 与 `scripts/` |
| 8 | 不用 `print`（CLI 除外），不用 `sys.exit`（库代码） |
| 9 | 抛 `HandwashError` 子类，消息写清"哪个字段/期望/实际" |
| 10 | 洗手步骤字符串只出现在 `core/labels.py` |

---

## 四、四个"绝对不能"

1. **不能让同一段原始视频跨 train/test**（数据泄漏 → 所有结论作废）。
   流程固定为：**扫描 → 按视频划分 → 抽帧**，顺序不可颠倒。
2. **不能重排标签空间里的类别顺序**（旧 checkpoint 的输出会静默错位）。
   只能追加到末尾。
3. **不能在业务代码里写判定规则**（漏步/顺序/时长只在 `core/protocol.py`）。
4. **不能用 `git push --force` 覆盖别人的提交**（`main` 分支禁止强推）。

---

## 五、提交前跑什么

```bash
python -m pre_commit run --all-files     # 快：格式 + 结构 + 快速单测
make check                               # 全：与 CI 完全一致
```

不装 pre-commit 也行，但 PR 里必须贴出 `make check` 的输出。

---

## 六、PR 检查清单（复制到 PR 描述）

```markdown
- [ ] 我改的文件属于我负责的模块，或已获得负责人同意
- [ ] 我没有修改 L1 冻结文件（若有，RFC 链接：____）
- [ ] 新增参数都进了 configs/，没有新增魔数
- [ ] 新增/修改的模块都有对应测试
- [ ] `make check` 通过（贴输出）
- [ ] 跑过受影响的最小流程（贴命令与结果）
- [ ] 若改了模型/数据，已把实验记进 docs/EXPERIMENTS.md（含 config_hash）
- [ ] 若影响历史结果，已列出"需要重跑的实验"
- [ ] 没有提交数据、训练权重或输出；根目录 `exp.pt` 是唯一权重例外
```

---

## 七、看到报错先查这里

| 报错 | 第一反应 |
| --- | --- |
| `ConfigError: 出现未知配置键` | 键名拼错，或你加了新参数没写进 `core/config.py` |
| `DataLeakageError` | 同一原始视频跨了 split；检查 `split.group_key` |
| 准确率 > 0.98 | 几乎一定是泄漏，不是模型好 |
| 准确率异常低 | `normalize` 与 `arch` 不匹配（YOLO 用 `zero_one`） |
| `ModuleNotFoundError: handwash` | 没 `pip install -e .`，且没用 `scripts/` 入口 |
| Windows DataLoader 报错 | `runtime.num_workers: 0`（写在 `configs/local.yaml`） |
| 太长的 traceback | 先看最后一行，再看 `handwash doctor` 的输出 |

---

## 八、每次跑实验都要记的东西

```
run_name          outputs/<run_name>/
config_hash       resolved_config.yaml 里的 config_hash
数据集与划分      split_report.json
指标              eval/eval_*.json + confusion_matrix_*.png
一句话结论        写进 docs/EXPERIMENTS.md
```

**没有 `config_hash` 的结果不允许出现在报告里。**

---

## 九、我要加一个新东西（最短路径）

| 想做 | 步骤 |
| --- | --- |
| **新模型** | `models/xxx.py` 用 `@register_model("名")` 注册 → `factory.py` 的 `register_all()` 加 import → `configs/models/xxx.yaml` → 形状测试 |
| **新数据集** | `core/labels.py` 的 `_DATASET_ALIASES` 补别名（追加）→ `configs/data/xxx.yaml` → `configs/config.yaml` 的 `datasets` 登记 → `docs/DATA.md` |
| **新实验** | `cp configs/config.yaml configs/experiments/expNN_描述.yaml`，只写与默认不同的键 |
| **新指标** | `core/metrics.py` **追加**函数 → 接入 `pipelines/evaluate.py` → 手算例子做单测 |
| **新命令** | `cli.py` 加子命令 → `Makefile` 加 target → `scripts/` 加薄封装（三处同步） |
| **改阈值** | 只改 `configs/` 的 `assess` 段（**不要改代码**） |

---

## 十、给自己一个"看得见的验收"

答辩前必须能现场演示这三件事：

```bash
# 1) 一段完整规范的洗手 -> 判定完整、顺序正确
python scripts/run_assess.py --video <完整视频>.mp4

# 2) 故意漏一步 -> 明确指出漏了第几步
python scripts/run_assess.py --video <漏步视频>.mp4

# 3) 故意换序 -> 明确指出顺序异常
python scripts/run_assess.py --video <换序视频>.mp4
```

三段视频的准备与标注规范见 [`SELF_RECORDING.md`](/docs/SELF_RECORDING.md)。

---

**最后一句：规则存在的唯一目的是让"任何人 clone 下来都能复现任何人的结果"。**
如果某条规则妨碍了正确的事，别偷偷绕过 —— 在 PR 里说明理由，让大家一起改规则。
