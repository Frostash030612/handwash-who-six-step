# 本地运行输出

本目录只保存可重新生成的本地运行产物，例如网页演示报告、逐帧预测和训练时的特征缓存；不提交 Git。

当前可交付成果不依赖这里的历史文件：

- `exp.pt`：当前七分类 YOLO 帧模型；
- `exp_temporal_head.pt`：当前 GRU 时序头；
- `data/external/pskus_demo_validation/`：8 条公开验证视频。

需要重新生成演示报告时，运行：

```bash
conda activate handwash-demo
python scripts/run_camera.py --demo-exp
```
