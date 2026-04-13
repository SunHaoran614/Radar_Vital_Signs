# Radar Vital Signs 文档总览

本文件是项目主入口文档。

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

数据结构细节见 [docs/DATASETS.md](docs/DATASETS.md)。

### 2.1 近期统一导出更新（2026-04-11）

- 更新文件：`src/data/loaders/export_all_datasets.py`
- 统一目标：
  - 三数据集导出统一字段：`radar`, `time`, `heart_rate`, `respiration_rate`。
  - 统一雷达特征域：将 FTU/BGT60 从 `(frames, rx, chirps, samples)` 转换到 PhysDrive 风格 `(frames, doppler, angle, range)`，默认对齐为 `8x16x8`。
  - PhysDrive 标签对齐：从 `ecg/respiration` 估计 `heart_rate/respiration_rate`，并补齐 `time`（当前按 `20 Hz` 对齐）。
  - FTU/BGT60 的 `range` 维选择已更新：由“逐帧 top-k”改为“全样本能量中心 + 连续窗口”，提升跨帧语义稳定性。
- 说明：
  - PhysDrive 在当前公开包中是 processed mmWave，不是原始 ADC；FTU/BGT60 为原始或近原始数据，因此统一前必须走特征变换。
  - 详细背景与参数来源已同步到 `docs/DATASETS.md`。

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

1. 经典域适应基线：DANN、MMD、CORAL（作为主流对照组）。
2. Source-Free 路线：Source-Free plain、Source-Free + WPL。
3. 本项目升级版：Source-Free + WPL + 时序修正（回归任务替代Reverse-kNN）。

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
本版已合并重复项，突出“经典DA基线 -> Source-Free升级 -> 鲁棒性/消融”主线。

| ExpID | 训练域 | 测试域 | 方法 | 主要输入 | 主要输出 | 核心指标 | 通过标准 | 实验目的（备注） |
|---|---|---|---|---|---|---|---|---|
| E01 | 4TU | 4TU | 传统信号基线 | 原始ADC + 参考心率 | HR/BR估计 | MAE, RMSE | MAE <= 6 bpm | 建立信号处理上界与可解释参考 |
| E02 | 4TU | 4TU | 深度基线(CNN-LSTM) | 统一特征张量 | 预测序列 | MAE, r | 不劣于E01 | 建立可迁移的源域监督模型 |
| E03 | 4TU | PhysDrive + BGT60 | 直接迁移(无适配) | E02模型 | 跨域预测 | MAE, 降幅率 | 记录下界 | 统一作为跨域下界（合并原E03/E04） |
| E04 | 4TU | PhysDrive + BGT60 | 经典DA基线组 | DANN / MMD / CORAL | 跨域预测 | MAE, r | 至少1种优于E03 | 合并原E05/E06/E07，保留主流对照 |
| E05 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free plain | 源模型 + 目标无标签 | 跨域预测 | MAE, r | 接近E04最优 | 验证“无源数据适配”可行性 |
| E06 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free + WPL | E05 + 置信度加权伪标签 | 跨域预测 | MAE, r | 优于E05 | 验证WPL抑制伪标签噪声有效 |
| E07 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free + WPL + 时序修正 | E06 + Temporal Correction | 跨域预测 | MAE, r, 漂移量 | 优于E06或更稳 | 对应WPL-SFUDA思想的本项目版本 |
| E08 | 最优模型 | 反向迁移(BGT60->4TU) | 方向一致性验证 | 最优配置 | 跨域预测 | MAE, r | 不出现明显反向失效 | 排除单向迁移偶然性（原E09简化） |
| E09 | 最优模型 | 极端/干扰/长时子集 | 鲁棒性与稳定性 | 场景子集 + 长时序列 | 分场景误差 + 漂移曲线 | 场景波动率, 漂移斜率 | 波动率<=25%，漂移可控 | 合并原E10/E11，统一鲁棒性评估 |
| E10 | 最优模型 | 跨域测试集 | 消融与统计检验 | 去除WPL/时序修正/前端模块 | Delta MAE, CI | 关键模块贡献显著 | 合并原E12并补统计显著性 |

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

## 6. 方法新颖性与参考文献

### 6.1 经典基线（必须保留）

- `DANN / MMD / CORAL` 仍是跨域任务中的主流强基线，适合本项目作为第一阶段可复现对照。
- 新颖性有限：更适合做“可靠起点”，不建议作为论文唯一创新点。

经典论文：

- DANN (ICML 2015 / JMLR 2016): https://proceedings.mlr.press/v37/ganin15 / https://jmlr.org/papers/v17/15-239.html
- DAN (MMD, ICML 2015): https://proceedings.mlr.press/v37/long15
- Deep CORAL (ECCV-W 2016): https://arxiv.org/abs/1607.01719

### 6.2 建议重点重写的升级方向（面向本项目）

- `Source-Free UDA`（更贴近真实部署）：
  - Radar HAR 代表案例：WPL-SFUDA (Pattern Recognition, 2026): https://www.sciencedirect.com/science/article/pii/S0031320325005266
  - 价值：目标域无标签，且不需要访问源域原始数据，符合隐私约束和工程落地需求。
- `对比学习 + 域对齐`（通常比纯 DANN 更稳）：
  - mmWave gait 代表案例：GaitSADA (2023): https://arxiv.org/abs/2301.13384
  - 价值：先学习更有判别性的表征，再进行分布对齐，在低标注场景更容易获得稳定提升。
- `领域相关迁移学习`（跨模态/跨设备）：
  - 与生理监测更接近：PSG -> FMCW radar 迁移 (2026): https://www.mdpi.com/2306-5354/13/3/283
  - 领域直系早期工作：IR-UWB + ECG 的 SADA (2018): https://www.sciencedirect.com/science/article/pii/S1746809418301927
  - 价值：能直接回答“跨设备、跨传感模态”下生命体征标签如何迁移的问题。
- `鲁棒前端（物理先验）+ 轻量适配`：
  - Pi-ViMo (mmWave vital signs, 2023): https://arxiv.org/abs/2303.13816
  - 价值：先做生理信号提纯，再做域适配，通常比端到端硬对齐更稳，更适合噪声和场景扰动明显的数据。

### 6.4 与本项目的适配建议

- 当前规划中的 `DANN/MMD/CORAL` 保留为基线层。
- 若要提升论文创新性，建议升级到：`基线DA + Source-Free/TTA + 雷达物理先验前端`。


## 7. 文档边界

- 保留：`docs/DATASETS.md`
- 规划、实验矩阵、指标定义统一维护在本文件

## 8. 文档同步约定

- 凡是涉及加载器输出字段、雷达轴语义、采样率/时间轴、导出格式的代码变更，需同步更新：
  - `docs/DATASETS.md`（数据事实与接口语义）
  - `README.md`（方案级摘要与影响范围）
- 提交前至少检查一次“代码实现 vs 文档描述”一致性，避免后续实验配置偏差。


## 9. docker
```bash
docker run -it --gpus all \
  --network host \
  --name radar_dev \
  -v /home/sunny/code/Radar_Vital_Signs:/Radar_Vital_Signs \
  -v /mnt/d/Dataset:/Radar_Vital_Signs/Dataset \
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  -e HTTP_PROXY=$HTTP_PROXY \
  -e HTTPS_PROXY=$HTTPS_PROXY \
  -e no_proxy=$no_proxy \
  -e NO_PROXY=$NO_PROXY \
  -w /Radar_Vital_Signs \
  radar_vital_signs bash
```

- 镜像地址：docker pull crpi-ojnb84j7hma95ay2.cn-shanghai.personal.cr.aliyuncs.com/hsun97282/radar:latest
