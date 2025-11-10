# 数据集说明文档

## 概述

本项目使用两个公开的雷达生命体征检测数据集进行算法开发和性能评估。每个数据集都有其独特的特点和应用场景，涵盖了从极端生理场景到车载驾驶环境的多样化应用。

**数据集位置**: `Dataset/` 文件夹

- **4TU.ResearchD**: 极端生理场景毫米波雷达数据集
- **PhysDrive**: 车载驾驶员多模态生理监测数据集

## 1. 4TU.ResearchD - 极端生理场景毫米波雷达数据集

### 数据集简介

**完整名称**: Comprehensive mmWave FMCW Radar Dataset for Vital Sign Monitoring Embracing Extreme Physiological Scenarios

这是一个专门针对极端生理场景的毫米波FMCW雷达生命体征监测数据集，由4TU.ResearchData平台发布。该数据集的独特之处在于包含了多种极端生理状态（如运动后心率升高）的测量数据，为算法在挑战性场景下的性能评估提供了宝贵资源。

### 数据集特点

- **雷达类型**: 毫米波FMCW (Frequency Modulated Continuous Wave) 雷达
- **雷达型号**: AWR1642BOOST (Texas Instruments)
- **工作频率**: 60 GHz频段
- **受试者数量**: 10名参与者 (Participant 1-10)
- **测试场景**:
  - **距离场景 (Distance Scenario)**: 4种距离 (40cm, 80cm, 120cm, 160cm)
  - **方向场景 (Orientation Scenario)**: 4种身体朝向 (Front side, Back, Left side, Right side)
  - **角度场景 (Angle Scenario)**: 3种雷达角度 (0°, 30°, 45°)
  - **运动后场景 (Elevated)**: 运动后心率升高状态 (Participant 2, 3, 4, 6)
- **每个场景重复次数**: 4次重复测量 (1, 2, 3, 4)
- **数据采集时长**: 每次测量约60秒
- **参考设备**: Polar H10 心率传感器
- **数据集特色**:
  - 包含运动后心率升高的极端生理场景
  - 包含哮喘患者的生理数据
  - 包含冥想练习者的生理数据
  - 多种生理特征的参与者,提供多样化的生理状态数据

### 数据集结构

```
Dataset/4TU.ResearchD/
├── Radar data/                          # 雷达原始数据
│   ├── Participant 1/
│   │   ├── 1. Distance Scenario/        # 距离场景
│   │   │   ├── 40 cm/
│   │   │   │   ├── 1/                   # 第1次重复
│   │   │   │   │   ├── data_LogFile.txt        # 数据采集日志
│   │   │   │   │   ├── data_Raw_0.bin          # 原始雷达ADC数据
│   │   │   │   │   └── data_Raw_LogFile.csv    # 原始数据日志
│   │   │   │   ├── 2/                   # 第2次重复
│   │   │   │   ├── 3/                   # 第3次重复
│   │   │   │   └── 4/                   # 第4次重复
│   │   │   ├── 80 cm/
│   │   │   │   ├── 1/, 2/, 3/, 4/
│   │   │   ├── 120 cm/
│   │   │   │   ├── 1/, 2/, 3/, 4/
│   │   │   └── 160 cm/
│   │   │       ├── 1/, 2/, 3/, 4/
│   │   ├── 2. Orientation Scenario/     # 方向场景
│   │   │   ├── 80 cm- Front side/       # 正面朝向
│   │   │   │   ├── 1/, 2/, 3/, 4/
│   │   │   ├── 80 cm- back/             # 背面朝向
│   │   │   │   ├── 1/, 2/, 3/, 4/
│   │   │   ├── 80 cm- left side/        # 左侧朝向
│   │   │   │   ├── 1/, 2/, 3/, 4/
│   │   │   └── 80 cm- Right side/       # 右侧朝向
│   │   │       ├── 1/, 2/, 3/, 4/
│   │   └── 3. Angle Scenario/           # 角度场景
│   │       ├── 0 deg- 80 cm/            # 0度角
│   │       │   ├── 1/, 2/, 3/, 4/
│   │       ├── 30 deg- 80 cm/           # 30度角
│   │       │   ├── 1/, 2/, 3/, 4/
│   │       └── 45 deg- 80 cm/           # 45度角
│   │           ├── 1/, 2/, 3/, 4/
│   │
│   ├── Participant 2/
│   │   ├── 1. Distance Scenario/
│   │   ├── 2. Orientation Scenario/
│   │   ├── 3. Angle Scenario/
│   │   └── 4. Elevated/                 # 运动后高心率场景
│   │       └── 80 cm/
│   │           ├── 1/, 2/, 3/, 4/
│   │
│   ├── Participant 3/
│   │   ├── 1. Distance Scenario/
│   │   ├── 2. Orientation Scenario/
│   │   ├── 3. Angle Scenario/
│   │   └── 4. Elevated/
│   │
│   ├── Participant 4/
│   │   ├── 1. Distance Scenario/
│   │   ├── 2. Orientation Scenario/
│   │   ├── 3. Angle Scenario/
│   │   └── 4. Elevated/
│   │
│   ├── Participant 5/
│   │   ├── 1. Distance Scenario/
│   │   ├── 2. Orientation Scenario/
│   │   └── 3. Angle Scenario/
│   │
│   ├── Participant 6/
│   │   ├── 1. Distance Scenario/
│   │   ├── 2. Orientation Scenario/
│   │   ├── 3. Angle Scenario/
│   │   └── 4. Elevated/
│   │
│   └── Participant 7-10/
│       ├── 1. Distance Scenario/
│       ├── 2. Orientation Scenario/
│       └── 3. Angle Scenario/
│
├── HR_Ref_Values/                       # 心率参考值
│   ├── Participant 1/
│   │   ├── 1. Distance Scenario/
│   │   │   ├── 40 cm/
│   │   │   │   ├── R1.CSV               # 第1次重复的心率参考值
│   │   │   │   ├── R2.CSV               # 第2次重复的心率参考值
│   │   │   │   ├── R3.CSV               # 第3次重复的心率参考值
│   │   │   │   └── R4.CSV               # 第4次重复的心率参考值
│   │   │   ├── 80 cm/
│   │   │   │   ├── R1.CSV, R2.CSV, R3.CSV, R4.CSV
│   │   │   ├── 120 cm/
│   │   │   └── 160 cm/
│   │   ├── 2. Orientation Scenario/
│   │   │   ├── 80 cm- Front side/
│   │   │   │   ├── R1.CSV, R2.CSV, R3.CSV, R4.CSV
│   │   │   ├── 80 cm- back/
│   │   │   ├── 80 cm- left side/
│   │   │   └── 80 cm- Right side/
│   │   └── 3. Angle Scenario/
│   │       ├── 0 deg- 80 cm/
│   │       │   ├── R1.CSV, R2.CSV, R3.CSV, R4.CSV
│   │       ├── 30 deg- 80 cm/
│   │       └── 45 deg- 80 cm/
│   │
│   └── Participant 2-10/
│       └── (相同结构,Participant 2,3,4,6 额外包含 4. Elevated/)
│
├── BR_Ref_Values/                       # 呼吸率参考值
│   └── BR_ref_human.ods                 # 呼吸率参考数据表(OpenDocument格式)
│
└── 数据集_包含极端生理场景Comprehensiv mmWave FMCW Radar Dataset for Vital Sign Monitoring Embracing Extreme Physiological Scenarios.pdf
```

### 数据文件说明

1. **雷达数据文件** (位于 `Radar data/` 文件夹):
   - `data_Raw_0.bin`: 原始雷达ADC数据,包含雷达接收的原始信号
   - `data_LogFile.txt`: 数据采集日志文件,包含时间戳和雷达配置信息
   - `data_Raw_LogFile.csv`: 原始数据日志CSV格式,记录采集过程的详细信息

2. **参考值文件**:
   - **心率参考值** (`HR_Ref_Values/` 文件夹):
     - `R1.CSV - R4.CSV`: 每次重复测量的心率参考值（来自Polar H10心率传感器）
     - CSV格式,包含时间序列心率数据,单位为bpm (beats per minute)
   - **呼吸率参考值** (`BR_Ref_Values/` 文件夹):
     - `BR_ref_human.ods`: 呼吸率参考值汇总表(OpenDocument格式),包含所有参与者的呼吸率数据

### 雷达原始数据格式详解

#### 1. 雷达配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **频率范围** | 77-81 GHz | 起始频率77 GHz, 结束频率81 GHz |
| **带宽** | 4 GHz | 频率扫描带宽 |
| **ADC采样点** | 250 | 每个chirp的ADC采样数 |
| **采样率** | 6250 ksps | ADC采样率 |
| **接收天线** | 4 | RX1, RX2, RX3, RX4 |
| **发射天线** | 2 | TX1, TX2 |
| **帧数** | 1200 | 每次测量的总帧数 |
| **Chirp数/帧** | 128 | 每帧包含的chirp数量 |
| **帧周期** | 50 ms | 帧率 20 fps |
| **Chirp时长** | 50 µs | 单个chirp的持续时间 |
| **距离分辨率** | 3.75 cm | c/(2×BW) = 3×10⁸/(2×4×10⁹) |

#### 2. 数据立方体结构

**原始数据维度**：
```
[RX天线, ADC采样, 帧, Chirp] = [4, 250, 1200, 128]
```

**实际使用维度**（合并帧和Chirp）：
```
[RX天线, ADC采样, 总Chirp数] = [4, 250, 153600]
其中: 153600 = 1200帧 × 128 chirps/帧
```

#### 3. 数据文件格式

**文件大小**：
```
总大小 = 4(RX) × 250(ADC) × 1200(帧) × 128(Chirp) × 4(字节)
      = 614,400,000 字节 ≈ 600 MB
```

**数据类型**：
- **存储格式**: 16位有符号整数 (int16)
- **IQ复数**: 每个采样点包含I(实部)和Q(虚部)
- **字节分配**: I(2字节) + Q(2字节) = 4字节/采样

**数据排列顺序** (基于DCA1000 LVDS Lane格式)：

DCA1000使用**2个LVDS Lane**输出数据：
```
Lane 1 → I分量 (实部)
Lane 2 → Q分量 (虚部)
```

**实际存储顺序**（按RX通道交错，每个RX的IQ相邻）：
```
对于每个ADC采样点（每8个int16为一组）:
  [RX0_I, RX0_Q, RX1_I, RX1_Q, RX2_I, RX2_Q, RX3_I, RX3_Q]

完整序列:
  [ADC0: RX0_I, RX0_Q, RX1_I, RX1_Q, RX2_I, RX2_Q, RX3_I, RX3_Q]
  [ADC1: RX0_I, RX0_Q, RX1_I, RX1_Q, RX2_I, RX2_Q, RX3_I, RX3_Q]
  ...
  [ADC249: RX0_I, RX0_Q, RX1_I, RX1_Q, RX2_I, RX2_Q, RX3_I, RX3_Q]
  然后进入下一个chirp...
```

**索引规律**：
- RX0: I在索引 [0::8], Q在索引 [1::8]
- RX1: I在索引 [2::8], Q在索引 [3::8]
- RX2: I在索引 [4::8], Q在索引 [5::8]
- RX3: I在索引 [6::8], Q在索引 [7::8]

#### 4. 数据读取方法

**读取步骤**：

1. **读取二进制文件**：使用int16类型读取data_Raw_0.bin文件
2. **重塑数据**：将原始数据重塑为 [总采样数, 8] 的二维数组
3. **分离IQ分量**：根据索引规律提取每个RX通道的I和Q分量
4. **组合复数**：将I和Q组合成复数数据
5. **重塑为数据立方体**：最终得到 [4, 250, 153600] 的三维数组

**关键点**：
- 每8个int16值对应一个ADC采样点的所有RX数据
- 每个RX通道的I和Q分量相邻存储
- 需要按照索引规律正确提取各通道数据

**参考实现**：详见 `src/data/loaders/ftu_loader.py`

#### 5. 信号处理流程

**标准处理步骤**：

1. **Range FFT** (距离维FFT)
   - 对每个chirp的250个ADC采样做FFT
   - 生成距离-多普勒图
   - 输出维度: [4, 125, 153600] (只保留正频率)

2. **目标检测**
   - 在距离维找到最大能量对应的bin
   - 该bin对应人体胸部位置

3. **相位提取**
   - 提取目标距离bin的相位信息
   - 相位变化反映胸部微小位移

4. **相位解包**
   - 消除2π跳变
   - 得到连续的相位信号

5. **位移计算**
   - 公式: 位移 = 相位 × λ / (4π)
   - 波长: λ = c/f ≈ 3.85 mm (中心频率78 GHz)

6. **生命体征提取**
   - **心率**: 带通滤波 0.8-2.5 Hz，对应48-150 bpm
   - **呼吸率**: 带通滤波 0.1-0.6 Hz，对应6-36 bpm
   - 通过FFT找峰值频率，乘以60转换为bpm

### 应用场景

- 极端生理状态下的生命体征监测
- 不同距离和角度下的雷达性能评估
- 运动后恢复期的心率监测
- 鲁棒性算法开发和测试
- 多场景下的算法泛化能力验证
- 非接触式生命体征监测研究

### 数据标注

- **心率 (HR)**: 每分钟心跳次数，包含正常和运动后升高状态
  - 正常心率范围: 60-100 bpm
  - 运动后心率: 可达 120-150 bpm
- **呼吸率 (BR)**: 每分钟呼吸次数
  - 正常呼吸率范围: 12-20 次/分钟
- **参考信号**: 来自专业医疗设备的同步测量数据
- **时间同步**: 雷达数据与参考信号时间对齐

### 数据集统计

- **总参与者数**: 10人
- **有运动后数据的参与者**: 4人 (Participant 2, 3, 4, 6)
- **总测量场景数**: 约 120+ 个场景组合
- **总数据文件数**: 约 1000+ 个测量文件

## 2. PhysDrive - 车载驾驶员多模态生理监测数据集

### 数据集简介

**完整名称**: PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring

PhysDrive 是一个专门针对车载驾驶场景的多模态远程生理测量数据集。该数据集的独特之处在于真实模拟了不同驾驶环境（道路类型、时间段、天气条件）下的生理信号采集，为驾驶员监测和疲劳检测研究提供了丰富的真实场景数据。

### 数据集特点

- **采集场景**: 真实车载驾驶环境
- **传感器类型**:
  - **毫米波雷达 (mmWave)**: 主要传感器,用于非接触式生理监测
  - **RGB 摄像头**: 可见光视频采集
  - **红外摄像头 (IR)**: 红外视频采集
  - **接触式生理传感器**: ECG、呼吸带、血氧仪等作为Ground Truth
- **受试者编码**: 48个不同的受试者-场景组合
- **场景维度**:
  - **车辆类型**:
    - A: Segment-A0 轿车
    - B: Segment-B 轿车
    - C: Segment-C SUV
  - **性别**:
    - M: 男性 (Male)
    - F: 女性 (Female)
  - **时间/天气**:
    - Z: 正午 (Noon)
    - H: 黄昏和清晨 (Dusk & Early morning)
    - W: 午夜 (Midnight)
    - Y: 雨天和阴天 (Rainy & Cloudy day)
  - **道路状态** (仅在视频数据中):
    - A: 平坦无阻碍道路 (Flat & Unobstructed Road)
    - B: 平坦拥堵道路 (Flat & Congested Road)
    - C: 颠簸拥堵道路 (Bumpy & Congested Road)
  - **驾驶员状态** (仅在视频数据中):
    - S: 静止 (Stationary)
    - T: 说话 (Talking)

### 数据集结构

```
Dataset/PhysDrive/
├── mmWave/                              # 毫米波雷达数据（已处理）
│   ├── AFH1/                            # A=Segment-A0, F=女性, H=黄昏/清晨, 1=第一次采集
│   │   ├── AFH1_00/                     # 编号00的数据段
│   │   │   ├── mmwave.mat               # 裁剪后的雷达信号 (n_doppler=8, n_angle=16, n_range=8)
│   │   │   ├── ecg.mat                  # ECG 心电信号
│   │   │   └── resp.mat                 # 呼吸信号
│   │   ├── AFH1_05/
│   │   ├── AFH1_10/
│   │   └── ... (多个数据段)
│   ├── AFH2/                            # 第二次采集
│   ├── AFW1/                            # W=午夜
│   ├── AFY1/                            # Y=雨天/阴天
│   ├── AFZ2/                            # Z=正午
│   ├── AMH1/                            # M=男性
│   ├── BFH1/                            # B=Segment-B轿车
│   ├── BMH1/
│   ├── CFH1/                            # C=SUV
│   ├── CMH1/
│   └── ... (共48个受试者-场景组合)
│
├── RGB and IR (one subject sample)/     # RGB和红外视频数据（样本）
│   ├── AMH1/
│   │   ├── AS/                          # A=平坦无阻碍, S=静止
│   │   ├── AT/                          # T=说话
│   │   ├── IR.mp4                       # 红外视频
│   │   ├── RGB.mp4                      # RGB视频
│   │   ├── Recording_Physiological_Data.csv  # 所有生理数据及时间戳
│   │   ├── Label/
│   │   │   ├── HR.mat                   # 滤波后的心率
│   │   │   ├── BVP.mat                  # 滤波后的血容积脉搏
│   │   │   ├── RESP.mat                 # 滤波后的呼吸信号
│   │   │   ├── ECG.mat                  # 滤波后的ECG信号
│   │   │   └── SPO2.mat                 # 血氧饱和度
│   │   └── STMap/
│   │       └── STMap_RGB.png            # 从RGB视频提取的时空图
│   └── ... (其他受试者样本)
│
├── README.md                            # 数据集说明文档
└── 数据集_驾驶环境PhysDrive A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring.pdf
```

### 命名规则详解

**mmWave 文件夹命名**: `[车型][性别][时间/天气][采集次数]`
- 第1位: A/B/C (车型)
  - A = Segment-A0 轿车
  - B = Segment-B 轿车
  - C = Segment-C SUV
- 第2位: M/F (性别)
  - M = 男性
  - F = 女性
- 第3位: Z/H/W/Y (时间/天气)
  - Z = 正午
  - H = 黄昏/清晨
  - W = 午夜
  - Y = 雨天/阴天
- 第4位: 1/2/3/4 (采集次数)

**示例**:
- `AFH1`: Segment-A0轿车, 女性驾驶员, 黄昏/清晨, 第1次采集
- `CMZ2`: SUV, 男性驾驶员, 正午, 第2次采集
- `BMW1`: Segment-B轿车, 男性驾驶员, 午夜, 第1次采集

**数据段命名**: `[场景ID]_[段编号]`
- 例如: `AFH1_00`, `AFH1_05`, `AFH1_10` 等
- 段编号为两位或三位数字,表示时间段索引

### 数据文件说明

1. **毫米波雷达数据** (`mmwave.mat`):
   - **维度**: (n_doppler=8, n_angle=16, n_range=8)
   - **格式**: MATLAB .mat 文件
   - **内容**: 已经过裁剪和预处理的雷达信号
   - **包含信息**:
     - 多普勒维度 (8): 速度信息
     - 角度维度 (16): 空间角度信息
     - 距离维度 (8): 距离信息
   - **数据类型**: 复数矩阵

2. **生理参考信号** (Ground Truth):
   - `ecg.mat`: 心电图信号 (ECG)
   - `resp.mat`: 呼吸信号 (Respiration)
   - `HR.mat`: 心率 (Heart Rate) - 仅在完整数据中
   - `BVP.mat`: 血容积脉搏 (Blood Volume Pulse) - 仅在完整数据中
   - `SPO2.mat`: 血氧饱和度 (SpO2) - 仅在完整数据中

3. **视频数据** (仅提供一个受试者样本 AMH1):
   - `RGB.mp4`: 可见光视频
   - `IR.mp4`: 红外视频
   - `STMap_RGB.png`: 从RGB视频提取的时空图
   - `Recording_Physiological_Data.csv`: 所有生理数据及时间戳
   - `Label/` 文件夹: 包含滤波后的生理信号

### 应用场景

- 车载驾驶员生理状态监测
- 疲劳驾驶检测
- 多模态融合算法开发
- 不同环境条件下的鲁棒性测试
- 非接触式生理信号测量研究
- 驾驶员注意力监测
- 压力和情绪状态评估

### 数据标注

- **生理参数**:
  - HR (心率): 每分钟心跳次数
  - RR (呼吸率): 每分钟呼吸次数
  - HRV (心率变异性): 心跳间隔变化
  - BVP (血容积脉搏): 血管容积变化
  - SPO2 (血氧饱和度): 血液氧饱和度百分比
- **环境标签**: 车型、时间段、天气、道路状态
- **驾驶员状态**: 静止、说话
- **时间戳**: 所有传感器数据同步时间戳

### 数据集统计

- **总场景数**: 48个不同的受试者-场景组合
- **mmWave数据段**: 每个场景包含多个时间段(约20-120个段)
- **总数据段数**: 约 3000+ 个数据段
- **每段时长**: 变化,通常为几秒到几十秒
- **采样率**: 根据传感器类型不同而异

