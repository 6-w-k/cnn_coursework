# 五组正式实验

五组为 `baseline`、`depthwise`、`se`、`gelu`、`mish`。模型实现位于 `cnn/models.py`，正式训练入口为 `scripts/train.py`，统一配置为 `experiments/plan.json`。

所有模型统一采用 CIFAR-10 适配版 ResNet-18：3×3、步长 1 输入卷积，无初始最大池化，四个阶段各两个残差块，通道数为 64/128/256/512，最后全局平均池化并输出 10 类。

## 单项结构修改

| 组别 | 修改 |
|---|---|
| baseline | 原始残差块与 ReLU |
| depthwise | 仅将残差分支的 16 个 3×3 标准卷积各替换为 3×3 逐通道卷积＋1×1 逐点卷积；保留原有 BN 和激活位置、输入端及投影快捷分支 |
| se | 8 个残差块分别在第二个 BN 后、残差相加前加入 SE；压缩比 16 |
| gelu | 将主干全部 17 处 ReLU 改成 GELU |
| mish | 将主干全部 17 处 ReLU 改成 Mish |

## 公平实验设置

- 固定随机种子 42，每组先运行一个种子，不作跨种子显著性或误差条声明。
- 分层划分训练集 45000 张、验证集 5000 张，每类分别为 4500/500 张。原始测试集 10000 张只在训练结束后评估最佳验证权重。
- 划分索引保存到 `splits/seed42.json`，每组记录同一文件 SHA-256。
- 统一从头训练 200 轮，训练批次 128，验证/测试批次 256。
- SGD：学习率 0.1、momentum 0.9、weight decay 0.0005；余弦学习率衰减。
- 数据增强统一为随机裁剪、随机水平翻转；归一化使用代码中的同一组均值和标准差。
- 使用 FP32、确定性算法及固定随机源。完整记录库版本与环境；不承诺不同软硬件环境之间逐位一致。
- 以验证集准确率选择权重，准确率相同时选择验证损失更小者。
- 卷积与全连接层 MACs 按真实前向计算统计，FLOPs 定义为 2×MACs；不包含 BN、激活、池化、偏置及逐元素操作。因此 GELU/Mish 的理论卷积 MACs 与基线相同，不代表实际耗时相同。

## 四卡调度

容器已扩展为可见四张 A10，仍使用原 `kang-workspace` 工作卷和 kang 私有虚拟环境。前四组分别占用 GPU 0/1/2/3；Mish 在队列中等待，由最先完成任务的 GPU 接续执行。每张卡同一时间只有一个本项目训练进程。

调度器会检查 GPU 占用，遇到其他计算任务时等待，不终止其他进程。GPU 0 上已有的小型桌面会话由管理员检查确认后按指定 PID 放行，不视作训练作业。该放行参数仅适用于当前服务器，换机器时需重新确认。

## 启动与查看

启动前先运行真实样本验证（不产生正式测试准确率）：

```bash
cd /home/kang/workspace/cnn_coursework
source /home/kang/workspace/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/verify_experiments.py
```

调度器的 `--run-dir` 必须是新的目录；训练脚本与运行配置的哈希必须与验证通过的版本一致：

```bash
python scripts/launch_experiments.py --run-dir runs/your_run_name --allow-background-pid 8579
python scripts/training_status.py runs/your_run_name
```

正式后台启动信息、实际 PID 和目录由本次启动记录提供。`8579` 是本次已检查的宿主机桌面进程 PID，不应在其他服务器照搬。

## 输出与恢复

每组保存 `config.json`、`environment.json`、`model_statistics.json`、`history.csv`、`status.json`、`best.pt`、`last.pt` 和标准输出日志。训练完成后生成 `metrics.json`、`test_predictions.npz`、混淆矩阵和每类 precision/recall/F1。

总目录保存 `queue.json`、配置、源码快照及 SHA-256。已完成结果自动汇总为 `comparison.json` / `comparison.csv`。指标文件中的百分比以 `_pct` 命名，F1/precision/recall 为 0～1 的比例。

训练脚本会逐轮原子保存权重、优化器、学习率调度器和随机状态。用相同配置加 `--resume` 可从 `last.pt` 恢复；调度器也支持 `--resume`。恢复逻辑在两数据加载进程条件下与不中断训练的最终模型权重作逐位比较验证。

## 场景应用入口

```bash
python scripts/predict.py runs/your_run_name/baseline/best.pt data/sample_png/train_00000_frog.png
```

该入口只针对 CIFAR-10 的十类图像，返回前三个类别和模型概率。使用短测权重时输出会明确标记 `training_smoke_checkpoint: true`，不能作为正式应用效果。
