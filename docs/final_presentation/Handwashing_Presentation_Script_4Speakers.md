# Camera-Based Handwashing Assessment — Speaker Script（四人分工版）

**Group 43** · Shen Ziyi（沈梓忆）· Wang Lepeng（王乐芃）· Xu Wenzhe（徐文哲）· Zhu Jianyu（朱建宇）

**Target:** 13:00 total — 8:00 slide narration plus 5:00 recorded demonstration. Play the demonstration at 04:20 and resume narration at 09:20.

本稿只做一件事：把现有 11 页讲稿按**最初的分工**拆到四个人头上。英文口播与数字一个字没改，只加了每段的讲者与时间边界。

## 分工总表

| 段落 | 讲者 | 负责模块 | 页面 | 口播时间 | 有效口播 |
|---|---|---|---|---|---|
| PART A | Xu Wenzhe 徐文哲 | 产品 / 交付（开场与收尾） | 1 · 2 · 10 · 11 | 00:00–01:05、11:10–13:00 | 2:55 |
| PART B | Shen Ziyi 沈梓忆 | 数据 / 评估 | 3 · 4 | 01:05–02:30 | 1:25 |
| PART C | Wang Lepeng 王乐芃 | 模型 / 训练 | 5 · 6 | 02:30–04:20 | 1:50 |
| PART D | Zhu Jianyu 朱建宇 | 演示 / 报告规则 / 结果 | 7 · 8 · 9 | 04:20–11:10 | 1:50（另含 5:00 视频播放） |

合计口播 2:55 + 1:25 + 1:50 + 1:50 = 8:00，加 5:00 演示视频 = 13:00。

## 分工依据

依据 10 月 6 日首次汇报的四人分工（`演讲稿-4人版-打印稿-修正.docx`）：Xu 负责产品、演示与交付；Shen 负责数据与评估；Wang 负责模型与训练；Zhu 负责时序模型与报告规则。按最终 11 页的内容类型对应过来：

- 封面与项目范围 → A（开场）
- 数据与数据驱动的可行性分析 → B（数据）
- 架构与动作确认机制 → C（模型）
- 演示与量化结果 → D（演示与评估）
- 本地部署取舍与结论 → A（收尾）

## 使用说明

1. **硬规矩①**：每段开口先说姓名和自己负责的模块，每段第一句已经写好，照说即可。
2. **点名交接**：上一段最后一句点名下一位（"Ziyi, over to you."），下一位开头接"Thanks, ..."。交接句合计约 15 秒，已包含在各段时长里，排练时按实际语速再核一次。
3. **不要照念页面**：页面是证据，口播只讲主线。每页控制在标注的秒数内。
4. **数字口径**：只念自己那段的数字，被追问别人的数字就转给对应的人。
5. 如需调整分工（例如把 Slide 4 改给 C 讲模型设计，或把 Slide 6 改给 D 讲报告规则），只改上面这张表和各段的讲者行，口播不用动。

# PART A · Xu Wenzhe（徐文哲）— 开场与收尾

**负责模块：** 产品 / 交付。开场两页定调子和范围，收尾两页讲部署取舍与结论。

## Slide 1 — Camera-Based Handwashing Assessment

**Xu Wenzhe｜00:00–00:25（25 s）**

> Hello, we are Group forty-three. Our project uses a local camera to assess six-step handwashing. It recognizes hand gestures, reconstructs their order, and measures recognized washing time. We will explain the model, demonstrate the application, and discuss the results and deployment trade-offs.

*Presenter cue: Start the presentation timer. Introduce the team without reading every name. 报组名，点一下四位成员，别逐个念名字。*

Source: Final project report, Abstract and Section 1.

## Slide 2 — What the system assesses

**Xu Wenzhe｜00:25–01:05（40 s）**

> A correct gesture prediction is only one part of the task. A person can perform recognizable actions while skipping a step, changing the order, or stopping too early. Our application therefore evaluates the complete procedure. It recognizes six actions and a background class, builds the observed action sequence, and checks total recognized washing time. Completion requires all six steps in the standard order and at least forty seconds of recognized actions. Background time is excluded. This is an assessment of visible procedure, rather than a direct measurement of cleanliness.

*Presenter cue: Point to the action examples, then to the order and duration requirements. 这页是全场的"问题定义"，把 40 秒门槛和"评估流程而非洁净度"这句话说清楚。*

Source: Final project report, Sections 1, 2.2 and 6.

# PART B · Shen Ziyi（沈梓忆）— 数据与评估

**负责模块：** 数据 / 评估。两页讲清楚数据从哪来、为什么可信，以及为什么用自录数据做最终评估。

## Slide 3 — The self-recorded dataset

**Shen Ziyi｜01:05–01:55（50 s）**

> Our target-scene dataset contains six thousand nine hundred and eighty-seven sampled frames from thirty-two recordings. Sampling at about five frames per second captures motion variation and transitions. The recordings include standard sequences, reordered or missing actions, and isolated examples of steps four and five. We split by video, so neighboring frames stay in the same partition. Fourteen videos support training, eight support validation, and ten form the reported test set. We distinguish definite action labels from uncertain transition frames. Those uncertain frames remain explicit in the annotations instead of becoming background labels by default.

*Presenter cue: Emphasize frame annotation and the split by video. Avoid describing adjacent frames as independent samples. 重点在"按视频划分"和"过渡帧不硬塞成背景"。*

Source: Final project report, Tables 2 and 3. dataset/README.md.

## Slide 4 — Why we model temporal boundaries

**Shen Ziyi｜01:55–02:30（35 s）**

> We first studied the public PSKUS dataset to check feasibility. On that preliminary comparison, macro F1 near transitions increased from eighty-three point three for YOLO to ninety point one with GRU, and ninety-one point zero with early boundary awareness. This supported using temporal context and transition supervision. We then trained the final systems on our self-recorded data. The final model set contains no PSKUS-trained weights. The public-data experiment explains the design choice, while our own scene supplies the final evaluation.

*Presenter cue: The plot shows preliminary transition-region macro F1, which differs from the final output-boundary F1 metric. 这是"公开数据只用来定设计"的边界说明，别让老师以为最终模型是 PSKUS 训的。*

Source: Final project report, Section 2.1 and Table 1.

# PART C · Wang Lepeng（王乐芃）— 模型与训练

**负责模块：** 模型 / 训练。两页讲系统怎么搭、动作怎么被确认，最后把画面交给演示视频。

## Slide 5 — YOLO + Adaptive Boundary-Aware GRU

**Wang Lepeng｜02:30–03:30（60 s）**

> Here is the complete architecture. YOLO extracts visual features from each sampled frame. A unidirectional GRU carries information forward through the sequence. Two heads share this temporal representation. The classification head estimates the current action, while the boundary head estimates where an action changes. Boundary-guided decoding produces candidate segments, and adaptive confirmation decides which segments belong in the final chain. This separates frame recognition from confirmation of an action instance. The recurrent state also supports incremental processing during a camera session. We compare this complete method with standalone YOLO, ordinary YOLO plus GRU, and VideoMAE. Each baseline answers a different question about visual recognition, temporal context, or video-window modeling.

*Presenter cue: Trace the diagram from the frame thumbnail through the two heads to the final chain. Explain the flow without reading layer dimensions. 顺着图讲一句话流程，不要念层尺寸。*

Source: Final project report, Section 3 and Figure 1. docs/MODEL_CARD.md.

## Slide 6 — Adaptive action confirmation

**Wang Lepeng｜03:30–04:20（50 s）**

> Brief prediction changes can create extra actions even when most frames are correct. We smooth labels over about one second, then group adjacent labels into segments. Adaptive confirmation compares each segment's duration and confidence with the recording's own statistics. An unusually short and uncertain segment can remain unconfirmed. This replaces the fixed three-second rule. With fewer than four comparable segments, duration rejection is skipped. The example removes an extra step four while preserving the observed order. We will now play a five-minute recording of the application in use.

*Presenter cue: Finish the introduction before 04:20, then start the prepared demonstration video. 最后一句说完立刻起视频，别停顿。*

Source: Final project report, Section 3.3 and Algorithm 1.

# PART D · Zhu Jianyu（朱建宇）— 演示与结果

**负责模块：** 演示 / 报告规则 / 结果。视频播放 5 分钟，视频结束后讲两页量化结果。

## Slide 7 — Recorded system demonstration

**Zhu Jianyu｜04:20–09:20（300 s · 视频播放）**

> （此处播放预先录制的 5 分钟演示视频，无现场口播。）

*Presenter cue: 04:20 开始播放，播到 09:20 自然结束，不要抢话。视频播完立刻接 Slide 8 的第一句。*

Source: Final project report, Section 6. src/handwash/self_scene_camera.py, src/handwash/self_scene_assessment.py.

## Slide 8 — Recognition and boundary results

**Zhu Jianyu｜09:20–10:25（65 s）**

> That was our system demonstration. We now turn to the quantitative results. The quantitative comparison uses six hundred and thirty-two aligned valid time centers from ten videos. Accuracy and macro F1 use the same reference labels for every model. Boundary F1 measures predicted changes using one-to-one matching with transition tolerance. Our method reaches ninety-six point five two percent accuracy, ninety-three point seven one macro F1, and ninety-four point four boundary F1. Relative to ordinary GRU, the largest improvement is boundary F1, which rises by six point eight one percentage points. VideoMAE achieves the highest values on all three measures. We report that clearly. These are complete-system results under their actual resolutions and training protocols, so they do not establish a controlled architecture-only advantage.

*Presenter cue: Read the proposed row and the boundary improvement. Do not read every table cell. VideoMAE 更强这件事要主动说，再补一句"这是完整系统比较，不是纯架构消融"。*

Source: evaluation/metrics.csv and final project report, Table 4.

## Slide 9 — Complete action chains

**Zhu Jianyu｜10:25–11:10（45 s）**

> Frame scores do not fully describe sequence quality. YOLO matches only two of the ten complete reference chains, ordinary GRU matches six, and both our final pipeline and VideoMAE match all ten. The examples show that a correct prediction can describe an incorrect procedure. Our model preserves a missing step and a reordered sequence instead of repairing them into the standard order. Only our pipeline uses adaptive confirmation on the dense five-frame-per-second stream. These results therefore compare complete methods, including their decoding, rather than an isolated change to the neural network.

*Presenter cue: Point to the four bars, then distinguish prediction correctness from procedure correctness. 这一页是"逐帧分数不等于流程正确"的核心证据。*

Source: evaluation/action_chains.csv and final project report, Tables 4 and 5.

# PART A（续）· Xu Wenzhe（徐文哲）— 部署取舍与结论

**负责模块：** 产品 / 交付。收尾两页：为什么选这个模型落地，以及边界和下一步。

## Slide 10 — The local deployment choice

**Xu Wenzhe｜11:10–11:55（45 s）**

> We retain our model for the local camera application because deployment also depends on size and temporal input requirements. It has about ten point six four million parameters, compared with eighty-six point two three million for VideoMAE. Its recurrent path consumes a frame and an existing state. VideoMAE uses a sixteen-frame window, requiring about three seconds of initial context at our sampling rate. Centered predictions also use future frames. VideoMAE can run locally and stream delayed predictions, but its workflow differs from immediate incremental feedback. The parameter comparison supports our choice without claiming a measured speed or energy advantage.

*Presenter cue: Present VideoMAE as the accuracy reference. Do not claim it is impossible to deploy or that parameter ratio equals speed ratio. 只说参数量与输入方式，不要说"更快的倍数"。*

Source: provenance/deployment/architecture_timing.json and final project report, Table 6.

## Slide 11 — Conclusions and next evaluation

**Xu Wenzhe｜11:55–13:00（65 s）**

> The project delivers a working camera application, annotated data, four comparison models, and reproducible evaluation records. Our selected pipeline combines visual recognition with learned boundaries and adaptive action confirmation. It achieves all ten reference chains in the reviewed evaluation and supports local feedback on both order and recognized duration. The scope of that evidence matters. We have inspected these recordings repeatedly, including correcting labels after reviewing predictions. The results therefore describe internal validation in the recorded scene. The next evaluation should freeze labels and settings, then use new people, camera positions, and lighting. It should also measure sustained throughput on the same hardware. Our contribution is a complete, inspectable assessment system with a clear deployment rationale. Thank you.

*Presenter cue: Finish at 13:00. Keep the final slide visible as the recording ends. 最后一段是"证据边界"，说得平，不心虚也不夸大。*

Source: Final project report, Section 7. docs/FINAL_RELEASE.md and provenance/README.md.

如果超时，按原有删减顺序砍：先砍 Slide 10 的参数量细节，再砍 Slide 6 的"fewer than four comparable segments"那句，Slide 9 的完整链例子尽量保留。
