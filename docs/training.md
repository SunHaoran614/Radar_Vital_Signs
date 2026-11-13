# 训练、评估与适配

安装后使用以下入口查看实际参数：

```bash
radar-train --help
radar-evaluate --help
radar-adapt --help
python -m src.training.evaluate_signal_baselines --help
```

训练必须显式指定源数据集、窗口导出目录和新输出目录，例如：

```bash
radar-train --model tcn --datasets FTU \
  --export-dir /path/to/source_windows \
  --output-dir /path/to/new_source_run
```

模型工厂支持CycleFormer、TCN、Transformer、DLinear/NLinear及其他保留骨干。通用模板不自动复现历史冻结实验；采用者需要明确数据版本、组划分、前端配置及随机种子。

`radar-adapt`基于已训练源权重读取目标train的无标签雷达，不输出目标HR指标。源与目标必须具有匹配的表征及传输特征契约；不同导出版本不能因为张量形状相同而混用。目标测试集只在方法固定后用于评估。

通用伪标签适配与[固定DAR推理](inference.md)是不同路径，不能把前者输出权重直接加载进后者。固定DAR权重的完整历史训练过程保存在本地归档中。

评估应同时报告同一BPM预测的绝对误差和记录内变化误差，提供无雷达基线、受试者级统计和每个lag的有效点对数。`src/evaluation/dar_regression.py`按原始窗口ID精确配对，禁止把质量筛选后的邻行当作相邻原始窗口。PhysDrive参考未认证时不得把重新评分当成可信验证。
