# 自采模型摄像头页面

## 使用方式

1. 选择 YOLO、YOLO + GRU、YOLO + 自适应边界感知 GRU、VideoMAE 中的一个模型。
2. 开启摄像头预览，调整位置和光线，保持双手完整入镜。
3. 点击“开始识别”。模型准备好后开始计时；采集期间锁定模型选择。
4. 完成后点击“结束并查看结果”。等待已收到的最后几帧推理完成后，直接汇总缓存的预测。

界面按准备、识别、结果分区；宽屏左右排列摄像头与状态，窄屏自动纵向排列。
当前动作编号条只表示当前分类，不表示六步已经完成。结果中的动作链才是整段录像的预测顺序。
运行详情默认折叠，保留设备、采样帧数、推理耗时和返回延迟。

## 启动

在仓库根目录，使用已经安装模型依赖的 Python 环境：

```sh
pip install -e ".[compare]"
make self-scene-camera
```

也可使用脚本或 CLI，两者功能一致：

```sh
python scripts/run_self_scene_camera.py --runtime-dir outputs/self_scene_camera config=configs/experiments/self_scene_camera.yaml
python -m handwash.cli self-scene-camera --runtime-dir outputs/self_scene_camera config=configs/experiments/self_scene_camera.yaml
```

打开 `http://127.0.0.1:8876/`。端口可用 `--port` 更改。
已有运行目录可以通过 `--runtime-dir /path/to/existing/runtime` 直接复用，无需复制录像和权重。
Makefile 对应参数为 `SELF_SCENE_CAMERA_DIR=/path/to/existing/runtime`。

当前实现优先使用 Apple MPS，否则回退 CPU；未在此入口实现 CUDA 设备选择。
一次只服务一个采集会话。网页刷新不会替用户重新开启摄像头；若后台会话仍未结束，可点击结束保存已收到的画面。

## 本机权重目录

此入口仅使用显式提供的自采权重，不回退到仓库旧演示权重或 PSKUS 模型。
运行目录包含：

```text
outputs/self_scene_camera/
  models/
    yolo_best.pt
    yolo_gru.pt
    boundary_gru.pt
    videomae/
      config.json
      preprocessor_config.json
      model.safetensors
  cache/        # 首次使用时创建
  sessions/     # 录像采样、逐帧预测和报告
```

YOLO 两种时序头的 `exp_sha256` 必须与 YOLO 权重匹配。边界头需带有训练时保存的解码参数。
VideoMAE 必须是本轮自采训练的七类检查点，类别映射为 `0=other, 1=step1, …, 6=step6`，16 帧窗口。
模型缺失时禁用对应选项；缺少 VideoMAE 依赖只影响选择 VideoMAE 后的加载，不预先加载其他模型。

训练对应的输入约定保持为 YOLO 1088、VideoMAE 224。YOLO 预处理先等比例缩放至长边 1080，再复制边缘填充为正方形；VideoMAE 使用与自采训练一致的直接 224 缩放及处理器归一化。
视频采样约 5 fps，单次最长 120 秒。模型权重、用户录像、推理缓存、账号信息均不随源码提交。

## 判定规则

- 配置 `assess.min_total_duration_s` 默认 **40 秒**，前端说明同步读取后端配置。
- 仅累计分类为动作 1–6 的时间；“其他”和没有预测覆盖的时间不计入洗手时长。
- **六步齐全、顺序为 1→2→3→4→5→6、累计洗手达到阈值**才判达标。
- 不单独要求每个动作达到某个时长。配置中的 `duration_check: none` 表示不启用旧流程的逐动作时长规则；本入口始终只考核总洗手时长。
- 未知时间可能影响是否达标时，显示时长证据不足，不自动把未知时间当作洗手。
- 自适应边界解码与动作链只用于边界感知模型；YOLO、普通 GRU 和 VideoMAE 使用各自分类序列。
- 短片段即使未进入边界模型的确认动作链，仍可贡献其预测洗手时间。这是“链的确认”与“总时长统计”两个不同指标。

40 秒是本项目当前判定配置。页面给出的是模型对该录像的判断；没有人工真值时，不生成准确率或 F1。
实时累计值是临时估计，最终报告按完整缓存汇总。VideoMAE 约需 3 秒形成首个窗口，之后每 0.8 秒更新；窗口中心标签约有 1.4 秒未来上下文需求，未覆盖的两端时间保留为未知。

## 本轮实现记录

- 单模型选择真正控制后端加载与推理，不仅隐藏其他模型结果。
- 停止后不重跑整段 YOLO / VideoMAE，复用采集时的概率缓存。
- `/status` 每 500 ms 轮询；选项文字、可用状态没有变化时不写入 DOM，避免打开原生下拉菜单时闪烁。
- 新增 `self-scene-camera` 入口，保留原 `camera` 命令及其界面，避免旧协议与新接口互相覆盖。
- 服务仅绑定本机 loopback。请求限制为本机来源、JPEG 大小和图像尺寸上限；帧时间必须递增。
- 本轮美化不重新训练权重、不调整已有模型解码，不产生新的模型精度结论。

此前本机功能检查已覆盖四个模型分别加载和推理、30 秒不足与 40 秒达标等时长场景；这一轮只做界面及部署整合，未重新运行模型对照实验或测试套件。当前页面已目视检查。完成后刷新页面会清空上次结果显示；本地录像文件保留。


## VideoMAE accuracy and live feedback

In the project's internal matched-point comparison, VideoMAE ranked first: 99.53% frame accuracy and 99.02% macro F1 on 632 aligned test points (10 clips). This is a small, repeatedly inspected internal test set, so these scores describe that comparison and do not establish generalization to new users or rooms.

That accuracy comes with a temporal delay. VideoMAE consumes a 16-frame window sampled at 5 fps, emits a new window prediction every 0.8 seconds, and places its label at the window center. It waits for about 3 seconds of frames before the first result and uses about 1.4 seconds of future frames relative to the label. A prediction can therefore arrive periodically while still describing an action from the recent past. It is useful as an offline accuracy benchmark, but it is not suitable for immediate live camera feedback. The smaller YOLO-family models provide the project's responsive live path.

最终权重、训练来源与交付包见 [最终技术版本](FINAL_RELEASE.zh-CN.md)。
