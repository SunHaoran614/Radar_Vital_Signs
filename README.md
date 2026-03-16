# Radar Vital Signs 文档总览

本文件是项目唯一主文档。

## 1. 项目目标

在毫米波雷达生命体征检测任务中，构建可复用、可迁移的方法体系，重点解决：

- 跨平台：AWR1642 与 BGT60TR13C
- 跨场景：实验室与车载
- 跨时长：短时样本与长时连续监测

## 2. 当前代码范围

已实现：

- `src/data/loaders/base_loader.py`
- `src/data/loaders/ftu_loader.py`
- `src/data/loaders/physdrive_loader.py`
- `src/data/loaders/bgt60_loader.py`

数据结构细节见 [DATASETS.md](DATASETS.md)。

## 3. 未来整体项目规划（按实验包执行）

### 包A：数据统一与质量分析

要做的实验：

1. 三数据集读取一致性检查（shape、dtype、时间轴、标签字段）。
2. 域差异量化（特征分布图、MMD/KL、信号强度统计）。
3. 数据可用性基线（缺失率、异常样本率、可训练样本数）。

交付：`reports/A_data_audit.md`、统一数据字典、可复现分析脚本。

### 包B：基线模型建立

要做的实验：

1. 传统信号处理基线（频谱峰值法/滤波链路）。
2. 深度学习基线（1D-CNN、LSTM、CNN-LSTM）。
3. 源域内性能上界与直接迁移下界。

交付：`reports/B_baselines.md`、基线训练配置、基线权重。

### 包C：跨域方法对比

要做的实验：

1. 特征层域适应：DANN、MMD、CORAL。
2. 数据层策略：噪声增强、幅度扰动、相位扰动、Mixup。
3. 组合策略：增强+DANN、增强+MMD、DANN+MMD。

交付：`reports/C_domain_adaptation.md`、方法对比表、最优组合配置。

### 包D：鲁棒性与稳定性验证

要做的实验：

1. 极值心率区间测试（低心率/高心率）。
2. 干扰场景测试（运动、说话、遮挡、多人）。
3. 长时漂移测试（分段统计、漂移曲线、恢复能力）。

交付：`reports/D_robustness.md`、失败样本库、误差归因报告。

### 包E：消融与论文材料

要做的实验：

1. 消融：移除增强、移除域损失、替换骨干网络。
2. 统计显著性检验（配对检验/置信区间）。
3. 论文图表与核心结论固化。

交付：`reports/E_ablation_and_paper.md`、论文图表源文件。

## 4. 实验矩阵（可实施）

说明：每行是一个可直接执行的实验单元，按 `ExpID` 建配置文件和结果目录。

| ExpID | 训练域 | 测试域 | 方法 | 主要输入 | 主要输出 | 核心指标 | 通过标准 | 实验目的（备注） |
|---|---|---|---|---|---|---|---|---|
| E01 | 4TU | 4TU | 传统信号基线 | 原始ADC + 参考心率 | HR/BR估计 | MAE, RMSE | MAE <= 6 bpm | 建立传统方法同域性能上界 |
| E02 | 4TU | 4TU | 深度基线(CNN-LSTM) | 统一特征张量 | 预测序列 | MAE, r | MAE <= E01 | 建立深度模型同域基线并与E01对照 |
| E03 | 4TU | PhysDrive | 直接迁移(无适配) | E02模型 | 跨域预测 | MAE, 降幅率 | 记录为跨域下界 | 量化4TU到车载场景的域偏移 |
| E04 | 4TU | BGT60 | 直接迁移(无适配) | E02模型 | 跨域预测 | MAE, 降幅率 | 记录为跨域下界 | 量化跨雷达平台域偏移 |
| E05 | 4TU | PhysDrive | DANN | 统一特征 + 域标签 | 跨域预测 | MAE, r | MAE较E03下降 >= 20% | 验证对抗域适应有效性 |
| E06 | 4TU | PhysDrive | MMD | 统一特征 | 跨域预测 | MAE, r | MAE较E03下降 >= 20% | 验证分布对齐损失有效性 |
| E07 | 4TU | PhysDrive | CORAL | 统一特征 | 跨域预测 | MAE, r | MAE较E03下降 >= 15% | 提供轻量域对齐对照方法 |
| E08 | 4TU | PhysDrive | 增强 + DANN | 增强样本 + 域标签 | 跨域预测 | MAE, r | MAE较E03下降 >= 30% | 验证组合策略是否达到主目标提升 |
| E09 | BGT60 | 4TU | 最优方法反向迁移 | 最优配置 | 跨域预测 | MAE, r | 验证双向有效 | 排除单向迁移偶然性，验证可迁移性 |
| E10 | 最优模型 | 各域极端子集 | 鲁棒性测试 | 低/高心率与干扰片段 | 分场景误差 | 场景MAE波动 | 波动率 <= 25% | 验证模型在困难场景的稳定性 |
| E11 | 最优模型 | BGT60长时数据 | 长时稳定性 | 连续长时序列 | 漂移曲线 | 分段MAE, 漂移量 | 漂移不过阈值 | 评估长时监测漂移与可用性 |
| E12 | 最优模型 | 跨域测试集 | 消融实验 | 去除单模块配置 | 性能差异 | Delta MAE | 证明关键模块贡献 | 解释性能来源并支撑论文结论 |

## 5. 评估指标（精简且可落地）

### 5.1 主指标

- MAE（bpm）
- RMSE（bpm）
- Pearson r

### 5.2 跨域指标

- 跨域性能下降率：`(MAE_target - MAE_source) / MAE_source`
- 相对改进率：`(MAE_baseline - MAE_method) / MAE_baseline`

### 5.3 鲁棒性指标

- 场景波动率：不同干扰场景 MAE 的变异系数。
- 长时漂移：按时间窗口统计 MAE 斜率。

### 5.4 工程指标

- 单样本推理时延（ms）
- 显存占用（MB）
- 可复现实验数（成功复现实验/总实验）

建议项目目标：最优跨域方法相对“直接迁移”基线，MAE 改进 >= 30%。

## 6. 环境（Conda）

```bash
# CPU
conda env create -f environment.yml
conda activate radar

# GPU
conda env create -f environment.gpu.yml
conda activate radar-gpu
```

## 7. 文档边界

- 保留：`docs/README.md`、`docs/DATASETS.md`
- 规划、实验矩阵、指标定义统一维护在本文件
