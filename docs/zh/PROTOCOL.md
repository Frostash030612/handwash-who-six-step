> [English](/docs/PROTOCOL.md) | **中文**

# 完整性判定口径（PROTOCOL）

本文件定义"漏步 / 顺序异常 / 时长不足"的**精确含义**。
报告里的每一个数字都必须能在这里找到对应的定义和阈值来源。

实现位置：`src/handwash/core/protocol.py`
阈值位置：`configs/config.yaml` 的 `assess` 段（**不要在代码里写阈值**）

---

## 1. 输入与输出

```
输入：逐帧标签序列 labels[]（Step）、逐帧置信度 confidences[]、有效帧率 fps
输出：ProtocolReport {
        is_complete, is_in_order,
        step_sequence, statistics[], violations[],
        total_wash_duration_s, total_duration_s, overall_score, notes[]
      }
```

处理链：**平滑 → 分段 → 三类判定 → 汇总打分**

---

## 2. 平滑（消除单帧跳变）

`assess.smooth_window`（默认 9 帧，5 fps 时约 1.8 秒）

- 以当前帧为中心的**多数投票**滑窗；窗口在序列两端自动收缩（不做零填充，
  否则边界会被无根据地"投票"给某个类别）。
- 平票时优先保持"上一帧结果"（时间连贯性优先于瞬时置信度）。
- 置信度低于 `assess.min_confidence`（默认 0.4）的帧**不参与投票**；
  整个窗口都不可信时沿用上一帧标签。

> 为什么不用"简单多数 + 固定窗口"：实测中第 4 步与第 2 步（都涉及手背）
> 容易互相跳变，时间连贯性约束能显著减少这类抖动。

---

## 3. 分段（把帧序列切成动作片段）

`assess.min_segment_frames`（默认 5 帧）+ `assess.min_segment_s`（默认 1.0 秒）

- 连续同标签的帧构成一个片段。
- 两个条件**都**要满足才算有效片段。
- 太短的片段**不丢弃**，而是并入时间上最近的相邻片段（保持时间轴无空洞）。
  若全部片段都太短，退回"整段一个主标签"，保证后续统计不出现空结果。

> 为什么不直接删掉短片段：删了会让总时长凭空缩短，
> "时长不足"的结论就会被这个删除动作本身制造出来。

---

## 4. 三类判定

### 4.1 漏步（missing）

```
detected(step) := 该步的累计时长 >= assess.min_segment_s
is_complete    := |{六步中 detected 的}| >= 6 - assess.missing_tolerance
```

- 默认 `missing_tolerance: 0` —— 漏一步就算不完整。
- 每个缺失步骤生成一条 `severity="error"` 的 violation。

**注意**：`detected` 用的是**时长阈值**而不是"出现过一帧"。
理由：一闪而过的误判不应被当成"做了这一步"。

### 4.2 顺序异常（out_of_order）

```
步骤序列 := collapse_repeats(片段标签序列中的六步)   # 先去掉 other/unknown/水龙头，再合并连续重复
对步骤序列，统计所有逆序对 (前, 后)：
    若 order(后) < order(前)  → 一条 out_of_order
```

- 只看 WHO 六步；`faucet_on/off`、`other`、`unknown` 不参与顺序判定。
- **1 → 2 → 1**（回头补做）会被判为乱序并单独说明，而不是静默忽略 ——
  "顺序异常"本身就是本项目要检出的目标之一。
- `assess.order_check: false` 可关闭顺序判定（用于对比实验）。

### 4.3 时长不足（insufficient_duration）

`assess.duration_check` 三选一：

| 取值 | 判定条件 | 适用场景 |
| --- | --- | --- |
| `seconds` | 该步时长 ≥ `min_step_duration_s`（默认 3.0 s） | 已知动作节奏的数据集 |
| `ratio`（默认） | 该步时长 ≥ `step_duration_ratio` × (总搓洗时间 ÷ 应做步骤数) | **推荐**：WHO 未要求六步均分时间 |
| `none` | 不判定 | 只关心漏步/顺序时 |

外加一条**整体**判定：

```
若 total_wash_duration_s < assess.min_total_duration_s  → 一条 error
（默认 40.0 s，依据 WHO 建议的完整洗手 40—60 秒）
```

> **为什么默认用 `ratio`**：WHO 指南只要求整个流程 40—60 秒，
> **没有**规定每步必须平均分配。用绝对秒数会把"某步做得快但流程完整"
> 误判为不合格。比例口径以"平均应得份额"为基准，更贴近指南原意。
> 报告中必须写明使用了哪一种口径。

### 4.4 重复步骤（repeated）

某一步在步骤序列中出现多次 → `severity="info"`。
`assess.allow_repeats: true` 时不再报告。

- 同一步被 `other` / `unknown` 短暂打断（停顿、手移出画面）后继续，合并为一次，
  **不算**重复；只有中间插入了另一个 WHO 步骤（如 5 → 6 → 5）才算。
- 同理，2 → 停顿 → 2 → 1 只记一条乱序，不会因停顿重复报告。

### 4.5 可选的水龙头事件检查

`assess.require_faucet_events: true` 时，平滑后的序列必须同时出现 `faucet_on` 和
`faucet_off`；缺少任一事件会生成一条警告。这些补充警告**不改变** `is_complete` 和
`overall_score`，两者仍只描述 WHO 六个搓洗步骤。模型标签空间必须包含这两个事件类别。
PSKUS 没有 `faucet_on` 标注，因此对应配置必须关闭该选项。

---

## 5. 综合得分（overall_score，0—1）

```
score = 0.5 × 六步覆盖率
      + 0.3 × (顺序正确 ? 1 : 0)
      + 0.2 × min(1, 总搓洗时间 / reference_total_duration_s)
```

- 权重**固定在代码里**（`core/protocol.py::_overall_score`），不放进配置 ——
  否则各组员用不同权重会得到不可比的分数。
- 分母 `reference_total_duration_s` 是可配置的（默认 50 s）。
- 得分只用于"同一套阈值下的横向比较"，**不是临床指标**，报告中必须如此表述。

---

## 6. 阈值调整的正确方式

**改配置，不改代码。**

```bash
# 单次实验
python -m handwash.cli assess --video demo.mp4 assess.min_total_duration_s=30

# 某个数据集的长期设定 -> 写进 configs/data/<name>.yaml 的 assess 段
```

如果要改的是**判定规则本身**（例如"允许回头补做不算乱序"），
那属于 L1 改动：先写 RFC（见 `docs/ARCHITECTURE.md` 第 5 节），再改
`core/protocol.py`，并同步更新本文件与 `CHANGELOG.md`。

---

## 7. 报告里必须交代的四件事

1. **口径**：`duration_check` 用的是 `seconds` 还是 `ratio`，阈值各是多少；
2. **帧率**：完整性判定基于抽帧后的有效帧率（`data.prep.fps`），不是原始 fps；
3. **模型**：用哪个 checkpoint（`config_hash` + epoch），是逐帧还是时序推理；
4. **失败案例**：至少给 2 个模型判错的例子，说明是视觉困难（遮挡/相似动作）
   还是流程困难（乱序/过快），这一节往往最能体现工作的深度。

---

## 8. 已知局限（主动写进报告，比被问到更好）

- **只做规范性分析，不做医学判断**：不评价洗手是否达到消毒标准。
- **依赖机位**：双手不清晰时六步区分度大幅下降（与公开数据集机位差异大时尤其明显）。
- **顺序判定的歧义**：真实洗手存在"回头补做"，框架默认判为乱序；
  若认为这不合理，应通过 RFC 修改规则并在报告中说明。
- **总时长阈值是参考值**：WHO 的 40—60 秒是完整流程建议，不是硬性标准，
  因此框架把它作为 `error` 级别的提示而非"不合格"判定。
