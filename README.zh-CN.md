> [English](/README.md) | **中文**

# 六步洗手动作识别与完整性评估

**最终技术版本：2026-10-22。** 本地摄像头识别“其他＋六个洗手动作”，生成动作链，
结合累计洗手时长输出结果。模型开发已定稿，后续以交付整理为主。

## 最终方案

- **摄像头主方案：YOLO＋自适应边界感知 GRU。**
- **结构对照：YOLO、YOLO＋GRU。**
- **离线精度参照：使用通用 Kinetics 预训练的 VideoMAE Base。**
- **数据：32 段自采视频。** 开发阶段 14 段训练、8 段验证，另有 10 段测试。
  保留的时序头训练脚本在选轮数后使用全部 22 段开发视频重训，详见
  [最终版来源说明](docs/FINAL_RELEASE.zh-CN.md)。
- **判定：六步齐全且顺序正确，预测洗手总时长至少 40 秒。** 不要求每个动作分别达到某个秒数。

本版不使用 PSKUS 数据或其训练权重。仓库中的公开数据实验属于历史阶段，不能替代当前自采权重。

## 启动

使用 Python 3.12，在仓库根目录执行：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/final_release/requirements-camera.txt -e ".[compare]"
python scripts/run_self_scene_camera.py --runtime-dir /path/to/runtime config=configs/experiments/self_scene_camera.yaml
```

从本地最终包提供 `runtime/models/`，具体文件见[摄像头部署说明](docs/SELF_SCENE_CAMERA.zh-CN.md)。
单独克隆 GitHub 不包含最终自采权重与视频；根目录旧 `exp.pt` 只是历史演示权重。

打开 http://127.0.0.1:8876/，授权摄像头，选择一个模型，预览、开始、结束。
一次只运行所选模型，结束时汇总缓存预测；刷新完成后的页面会清空旧显示，不删除本地录像。
入口优先使用 Apple MPS，否则 CPU；未提供 CUDA 选择。
本地 ZIP 已配套安装与启动脚本，详见[最终技术交付说明](docs/FINAL_RELEASE.zh-CN.md)。

## 已保存的内部对比

10 段已反复查看的视频，632 个共同有效时间点：

| 模型 | 准确率 | 宏 F1 | 输出边界 F1 | 最终动作链 |
| --- | ---: | ---: | ---: | ---: |
| YOLO | 95.25% | 93.84% | 76.92% | 2/10 |
| YOLO＋GRU | 96.04% | 93.58% | 87.59% | 6/10 |
| YOLO＋自适应边界感知 GRU | 96.52% | 93.71% | 94.40% | 10/10 |
| VideoMAE | 99.53% | 99.02% | 97.48% | 10/10 |

只有边界模型的最终动作链使用自适应确认。不同输入、训练过程及后处理共同影响结果，
本表不代表严格结构消融或新场景泛化。详见[完整口径与限制](docs/SELF_SCENE_FOUR_MODEL_COMPARISON.zh-CN.md)。

VideoMAE 内部精度最高，能在本机流式输出，但首窗约需 3 秒、中心标签约需 1.4 秒未来画面，
另有计算耗时。因此保留为离线精度参照，不能表述成“无法在本机部署”。主模型是当前选定的摄像头方案，
不宣称它在全部指标上最好。

## 成果入口

- [最终技术交付说明](docs/FINAL_RELEASE.zh-CN.md)
- [权重与解码指纹](docs/final_release/model_manifest.json)
- [七类标签](docs/final_release/label_map.json)
- [最终指标表](docs/final_release/metrics.csv)
- [逐视频动作链](docs/final_release/action_chains.csv)
- [实验记录索引](docs/EXPERIMENTS.md)
- [作业提交件](deliverables/README.md)

GitHub 保存代码和报告；本地最终技术 ZIP 另含权重、32 段原视频、采样帧、标注及预测导出。
不包含账号信息、摄像头历史会话或缓存。最终报告、PPT、展示视频和个人报告另行完成。

## 历史阶段

以下内容保留为历史来源，不是本版还需执行的任务：
[公开数据对比](docs/FAIR_VIDEO_COMPARISON.zh-CN.md)、
[旧实施计划](docs/IMPLEMENTATION_PLAN.zh-CN.md)、
[课程计划](docs/PROJECT_PLAN_4_WEEK.md)。
工程约定见[架构](docs/ARCHITECTURE.md)、[配置](docs/CONFIG.md)、[贡献规则](CONTRIBUTING.md)。
项目只做记录动作的模型评估，不作疾病诊断。
