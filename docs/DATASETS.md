# 数据集说明文档

**所属项目**: 面向跨域场景的毫米波雷达生命体征检测关键技术研究

## 概述

本项目使用三个雷达生命体征检测数据集进行跨域场景关键技术研究。包括两个公开数据集和一个自采集数据集，每个数据集都有其独特的特点和应用场景，涵盖了从极端生理场景到车载驾驶环境，以及长时连续监测的多样化应用，为跨域场景研究提供了丰富的数据支撑。

**数据集位置**: `Dataset/` 文件夹

- **4TU.ResearchD**: 极端生理场景毫米波雷达数据集（公开）
- **PhysDrive**: 车载驾驶员多模态生理监测数据集（公开）
- **BGT60TR13C_Indoor**: 室内长时连续监测数据集（自采集）

## 1. 4TU.ResearchD - 极端生理场景毫米波雷达数据集

### 数据集简介

**完整名称**: Comprehensive mmWave FMCW Radar Dataset for Vital Sign Monitoring Embracing Extreme Physiological Scenarios

这是一个专门针对极端生理场景的毫米波FMCW雷达生命体征监测数据集，由4TU.ResearchData平台发布。该数据集的独特之处在于包含了多种极端生理状态（如运动后心率升高）的测量数据，为算法在挑战性场景下的性能评估提供了宝贵资源。

### 数据集特点

- **雷达类型**: 毫米波FMCW (Frequency Modulated Continuous Wave) 雷达
- **雷达型号**: AWR1642BOOST (Texas Instruments)
- **工作频率**: 77-81 GHz
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
- **雷达型号**: 未公开
- **雷达配置参数**（来源：数据集论文第5页）:
  - **有效带宽**: 2.6 GHz
  - **虚拟天线阵列**: 12个天线
  - **距离分辨率**: ~6 cm
  - **角度分辨率**: 14°
  - **帧率**: 20 fps
  - **每帧Chirp数**: 64 chirps
  - **速度分辨率**: 7.6 cm/s
- **传感器类型**:
  - **毫米波雷达 (mmWave)**: 主要传感器,用于非接触式生理监测
  - **RGB 摄像头**: 可见光视频采集
  - **红外摄像头 (IR)**: 红外视频采集
  - **接触式生理传感器**: ECG、呼吸带、血氧仪等作为Ground Truth
- **受试者数量**: 48名驾驶员
- **会话数量**: 48个不同的驾驶会话
- **数据时长**: 每个样本约30秒（600帧）
- **场景维度**:
  - **路段类型**:
    - A: Segment-A0 (平坦无障碍道路)
    - B: Segment-B (平坦拥堵道路)
    - C: Segment-C SUV (颠簸拥堵道路)
  - **性别**:
    - M: 男性 (Male)
    - F: 女性 (Female)
  - **时间/天气**:
    - Z: 正午 (Noon)
    - H: 黄昏和清晨 (Dusk & Early morning)
    - W: 午夜 (Midnight)
    - Y: 雨天和阴天 (Rainy & Cloudy day)
  - **重复次数**: 1或2
  - **道路状态** (仅在视频数据中):
    - A: 平坦无阻碍道路 (Flat & Unobstructed Road)
    - B: 平坦拥堵道路 (Flat & Congested Road)
    - C: 颠簸拥堵道路 (Bumpy & Congested Road)
  - **驾驶员状态** (仅在视频数据中):
    - S: 静止 (Stationary)
    - T: 说话 (Talking)
- **数据集特色**:
  - 真实车载驾驶环境数据
  - 多样化驾驶条件（路段、时间、天气）
  - 包含运动干扰（说话、转头等）
  - 自然光照变化
  - 车辆振动和路面颠簸

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

**会话ID命名**: `[路段][性别][时间/天气][重复次数]` (4个字符)

| 位置 | 字符 | 含义 | 可能值 |
|------|------|------|--------|
| **第1位** | A/B/C | 路段类型 | A=平坦无障碍<br>B=平坦拥堵<br>C=颠簸拥堵(SUV) |
| **第2位** | M/F | 性别 | M=男性<br>F=女性 |
| **第3位** | Z/H/W/Y | 时间/天气 | Z=中午<br>H=黄昏/清晨<br>W=午夜<br>Y=雨天/阴天 |
| **第4位** | 1/2 | 重复次数 | 1或2 |

**示例**:
- `AFH1`: A(平坦无障碍) + F(女性) + H(黄昏/清晨) + 1(第1次)
- `CMZ2`: C(颠簸拥堵) + M(男性) + Z(中午) + 2(第2次)
- `BMW1`: B(平坦拥堵) + M(男性) + W(午夜) + 1(第1次)

**样本ID命名**: `[会话ID]_[样本编号]`
- 例如: `AFH1_00`, `AFH1_05`, `AFH1_10`, `AFH1_118`
- 样本编号为两位或三位数字,表示时间段索引
- 每个会话包含多个样本（如AFH1有35个样本）

### 雷达数据格式详解

#### 1. 雷达配置参数

**PDF文档明确提供的参数**（来源：论文第5页）：

| 参数 | 值 | 说明 |
|------|-----|------|
| **有效带宽** | 2.6 GHz | Effective bandwidth |
| **虚拟天线阵列** | 12个天线 | Virtual array of 12 antennas |
| **距离分辨率** | ~6 cm | Range resolution (around 6 cm) |
| **角度分辨率** | 14° | Angular resolution |
| **帧率** | 20 fps | Transmits 20 frames in one second |
| **Chirp数/帧** | 64 | Each frame has 64 chirps |
| **速度分辨率** | 7.6 cm/s | Velocity resolution |

**从配置参数计算得出**：

| 参数 | 值 | 计算方法 |
|------|-----|---------|
| **帧周期** | 50 ms | 1/20秒 |
| **理论距离分辨率** | ~5.77 cm | c/(2×BW) = 3×10⁸/(2×2.6×10⁹) |

**数据处理后的维度**（来源：论文第6页）：

| 参数 | 值 | 说明 |
|------|-----|------|
| **Range bins** | 8 | 距离维度bins数量 |
| **Doppler bins** | 8 | 速度维度bins数量 |
| **Angle bins** | 16 | 角度维度bins数量 |
| **速度范围** | ±60.8 cm/s | Doppler bins覆盖的速度范围 |

**未公开的参数**：
- ❌ 雷达设备型号
- ❌ 具体工作频率范围（起始/结束频率）
- ❌ ADC采样点数
- ❌ ADC采样率
- ❌ 发射/接收天线的具体配置

**注意**：
- 以上所有参数均来自官方论文，无推测内容
- 数据集提供的是已处理的RDA数据，而非原始ADC数据

#### 2. 数据立方体结构

**已处理数据维度**：
```
[时间帧, 实部/虚部, Doppler, Angle, Range] = [600, 2, 8, 16, 8]
```

**维度详解**：

| 维度 | 大小 | 含义 | 说明 |
|------|------|------|------|
| **Dim 0** | 600 | 时间帧 | 30秒 × 20fps = 600帧 |
| **Dim 1** | 2 | 实部/虚部 | [0]=实部, [1]=虚部 |
| **Dim 2** | 8 | Doppler bins | 速度维度，覆盖±60.8 cm/s范围 |
| **Dim 3** | 16 | Angle bins | 角度维度，覆盖前方人体存在区域 |
| **Dim 4** | 8 | Range bins | 距离维度，覆盖驾驶员可能的距离范围 |

**原始雷达配置到处理后数据的映射**：
```
原始配置:
- 64 chirps/frame  →  Doppler-FFT  →  8 Doppler bins
- 12虚拟天线       →  Angle-FFT   →  16 Angle bins
- ADC采样点(未知)  →  Range-FFT   →  8 Range bins
```

**数据处理说明**:
- 原始雷达配置: 64 chirps/frame, 12虚拟天线
- 经过Range-Angle-Doppler FFT处理后得到当前维度
- **8个Doppler bins**: 覆盖心肺运动速度范围（±60.8 cm/s，无大幅度身体运动）
- **16个Angle bins**: 覆盖前方驾驶员位置的角度范围
- **8个Range bins**: 覆盖驾驶员可能的距离范围
- **注意**: 这些bins数量是雷达系统配置的分辨率，包含完整探测空间信息，不是从更多bins中提取的子集

#### 3. 数据文件格式

**文件大小**：
```
总大小 = 600(帧) × 2(实/虚) × 8(Doppler) × 16(Angle) × 8(Range) × 8(字节)
      = 9,830,400 字节 ≈ 9.4 MB
```

**数据类型**：
- **存储格式**: 64位浮点数 (float64)
- **数值范围**: 约 -51 到 +42
- **复数表示**: 实部和虚部分离存储在Dim 1
- **文件格式**: MATLAB .mat 文件

**数据状态**：
- **已处理**: Range-Doppler-Angle (RDA) 格式
- **无需FFT**: 可直接用于特征提取和深度学习
- **时间同步**: 与ECG、呼吸信号时间对齐

**与4TU数据集的关键差异**:

| 特性 | 4TU.ResearchD | PhysDrive |
|------|---------------|-----------|
| **数据状态** | 原始ADC数据 | 已处理RDA数据 |
| **文件格式** | `.bin` (二进制) | `.mat` (MATLAB) |
| **复数表示** | IQ交织 | 实部/虚部分离 |
| **数据类型** | int16 | float64 |
| **文件大小** | ~600 MB/次 | ~9.4 MB/样本 |
| **处理复杂度** | 高（需要3次FFT） | 低（直接使用） |
| **分辨率配置** | 精细测量（250×128×~8） | 实时监测（8×16×8） |
| **适用场景** | 算法开发、信号处理研究 | 深度学习、实时应用 |

#### 4. Range-Doppler-Angle (RDA) 数据解释

**什么是RDA数据？**

Range-Doppler-Angle数据是毫米波雷达经过三次FFT处理后的三维数据立方体，包含目标的三个关键维度信息：

| 维度 | 含义 | 获取方式 | 物理意义 | PhysDrive配置 |
|------|------|---------|---------|--------------|
| **Range** | 距离 | Range-FFT | 目标到雷达的距离 | 8 bins (~6 cm分辨率) |
| **Doppler** | 速度 | Doppler-FFT | 目标的径向速度（运动方向） | 8 bins (7.6 cm/s分辨率) |
| **Angle** | 角度 | Angle-FFT | 目标相对雷达的方位角 | 16 bins (14°分辨率) |

**数据处理流程对比**:

```
4TU数据集（原始ADC数据）:
原始ADC数据 (int16)
    ↓
IQ解交织 → 复数信号
    ↓
Range-FFT → Range维度
    ↓
Doppler-FFT → Range-Doppler
    ↓
Angle-FFT → Range-Doppler-Angle

PhysDrive数据集（已处理RDA数据）:
已处理的RDA数据 (float64)
    ↓
直接使用！✨
```

**PhysDrive的优势**:
- ✅ 无需复杂的FFT处理
- ✅ 直接可用于特征提取和深度学习
- ✅ 包含距离、速度、角度三维信息
- ✅ 适合实时处理应用

**各维度的应用**:
- **Range维度**: 定位驾驶员在车内的位置
- **Doppler维度**: 检测呼吸和心跳引起的微小运动（关键维度）
- **Angle维度**: 区分驾驶员和其他反射目标

#### 5. 数据读取方法

**读取步骤**:
1. 使用 `scipy.io.loadmat` 加载MAT文件
2. 提取 `'mmwave'` 键的数据
3. 验证形状为 `(600, 2, 8, 16, 8)`
4. 转换为复数格式: `real + 1j * imag`

**参考实现**: 详见 `src/data/loaders/physdrive_loader.py`

### 参考数据格式

1. **ECG数据** (`ecg.mat`):
   - **数据形状**: `(1, 600)`
   - **数据类型**: float64
   - **含义**: 心电图信号

2. **呼吸数据** (`resp.mat`):
   - **数据形状**: `(1, 600)`
   - **数据类型**: float64
   - **含义**: 呼吸信号

3. **时间对齐**:
   - 雷达数据、ECG、呼吸信号都是600帧
   - 所有模态数据已时间同步

4. **其他参考信号** (仅在完整数据中):
   - `HR.mat`: 心率 (Heart Rate)
   - `BVP.mat`: 血容积脉搏 (Blood Volume Pulse)
   - `SPO2.mat`: 血氧饱和度 (SpO2)

### 视频数据

**仅提供一个受试者样本 (AMH1)**:
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

- **总驾驶员数**: 48人
- **总会话数**: 48个会话
- **会话分布**:
  - 路段A (平坦无障碍): 16个会话
  - 路段B (平坦拥堵): 16个会话
  - 路段C (颠簸拥堵): 16个会话
  - 男性驾驶员: 24个会话
  - 女性驾驶员: 24个会话
  - 各时间/天气条件: 各12个会话
- **样本数量**: 每个会话包含多个样本（如AFH1有35个样本）
- **总样本数**: 1685个样本
- **每样本时长**: 约30秒（600帧）
- **帧率**: 20 fps

### 数据来源与参考

- **数据集论文**: "PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring"
- **数据集网站**: https://github.com/WJULYW/PhysDrive-Dataset
- **Kaggle数据**: https://www.kaggle.com/datasets/xiaoyang274/physdrive
- **雷达配置参数来源**: 数据集论文第5页（3.2 Data Processing - Apparatus部分）
- **数据处理说明来源**: 数据集论文第6页（3.2 Data Processing - Data Preprocessing部分）
- **注意**:
  - 论文中提供了部分雷达配置参数（带宽、天线、分辨率等）
  - 但未公开雷达设备型号和具体工作频率范围
  - 本文档仅记录论文中明确提供的参数，不包含推测内容

---

## 3. BGT60TR13C_Indoor - 室内长时连续监测数据集（自采集）

### 数据集简介

**完整名称**: BGT60TR13C Indoor Long-term Vital Sign Monitoring Dataset

这是一个使用Infineon BGT60TR13C雷达芯片在室内环境下采集的生命体征监测数据集。该数据集的特点是包含了长时连续监测数据（2小时），为研究长期稳定性、昼夜节律变化、以及连续监测算法提供了重要支撑。

### 数据集特点

- **雷达类型**: 60 GHz FMCW雷达
- **雷达芯片**: BGT60TR13C (Infineon Technologies)
- **工作频率**: 58-63.5 GHz（来源：JSON配置文件）
- **受试者数量**: 8名受试者
- **测试场景**:
  - **短时测量**: 8名受试者 × 4种距离 = 32个场景（每次约10分钟）
  - **长时监测**: 1个场景（2小时连续监测）
- **测试环境**: 室内静态场景
- **测试距离**: 4种距离（0.3m, 0.6m, 0.9m, 1.2m）
- **参考设备**: Bigrun心率带（蓝牙心率监测设备）
- **数据采集时长**:
  - 短时测量：每次10分钟
  - 长时监测：连续2小时
- **数据集特色**:
  - 长时连续监测数据，可研究生理参数的时变特性
  - 多距离测试，覆盖不同距离场景
  - 室内真实环境，包含环境噪声和干扰
  - 高质量参考心率和呼吸率数据（Bigrun心率带）

### 雷达配置参数

**来源**: `BGT60TR13C_settings_20250423-163757.json`

| 参数 | 值 | 说明 |
|------|-----|------|
| **起始频率** | 58.0 GHz | start_frequency_Hz: 58,000,000,000 |
| **结束频率** | 63.5 GHz | end_frequency_Hz: 63,500,000,000 |
| **带宽** | 5.5 GHz | 频率扫描带宽 |
| **中心频率** | 60.75 GHz | (58.0 + 63.5) / 2 |
| **ADC采样点/chirp** | 512 | num_samples_per_chirp |
| **Chirp数/帧** | 16 | num_chirps_per_frame |
| **ADC采样率** | 2.0 MHz | sample_rate_Hz: 2,000,000 |
| **Chirp重复时间** | 0.400 ms | chirp_repetition_time_s |
| **帧重复时间** | 33.33 ms | frame_repetition_time_s |
| **帧率** | 30 fps | 1 / frame_repetition_time_s |
| **接收天线** | 3 | RX1, RX2, RX3 |
| **发射天线** | 1 | TX1 |
| **MIMO模式** | off | 未启用MIMO |
| **IF增益** | 30 dB | if_gain_dB |
| **高通滤波器** | 20 kHz | hp_cutoff_Hz |
| **抗混叠滤波器** | 500 kHz | aaf_cutoff_Hz |

**计算参数**:

| 参数 | 计算公式 | 值 |
|------|---------|-----|
| **距离分辨率** | c/(2×BW) | 2.73 cm |
| **理论最大距离** | c×fs/(4×BW) | ~27 km |
| **实际最大距离** | - | ~5 m (受限于信噪比) |

### 数据集结构

```
Dataset/BGT60TR13C/
├── BGT60TR13C_settings_20250423-163757.json  # 雷达配置文件
│
├── Radar_Data/                      # 雷达数据目录
│   ├── Participant1/                # 受试者1
│   │   ├── 0.3m/                    # 距离0.3米
│   │   │   └── radar_raw_data.npy   # 雷达原始数据 (~844 MB)
│   │   ├── 0.6m/                    # 距离0.6米
│   │   │   └── radar_raw_data.npy
│   │   ├── 0.9m/                    # 距离0.9米
│   │   │   └── radar_raw_data.npy
│   │   └── 1.2m/                    # 距离1.2米
│   │       └── radar_raw_data.npy
│   ├── Participant2/ ... Participant8/  # 受试者2-8
│   │   └── (同样的4个距离目录)
│   └── Long_duration/               # 长时监测（2小时）
│       └── radar_raw_data.npy       # 2小时连续数据 (~9.9 GB)
│
└── HR_Ref_Data/                     # 心率参考数据目录
    ├── Participant1/
    │   ├── 0.3m/
    │   │   └── HR_ref.csv           # 心率和呼吸率参考值
    │   ├── 0.6m/
    │   │   └── HR_ref.csv
    │   ├── 0.9m/
    │   │   └── HR_ref.csv
    │   └── 1.2m/
    │       └── HR_ref.csv
    ├── Participant2/ ... Participant8/
    │   └── (同样的4个距离目录)
    └── Long_duration/
        └── HR_ref.csv               # 2小时连续心率参考

```

### 雷达数据格式详解

#### 1. 雷达数据文件

**radar_raw_data.npy**:
- **文件格式**: NumPy `.npy` 文件
- **数据类型**: `uint16` (16位无符号整数)
- **采样方式**: **实采样** (Real Sampling)
- **数据形状**:
  - 短时测量 (10分钟): `(18000, 3, 16, 512)`
  - 长时监测 (2小时): `(216000, 3, 16, 512)`

**维度详解**:

| 维度 | 大小 | 含义 | 说明 |
|------|------|------|------|
| **Dim 0** | 18000 / 216000 | 帧数 | 10分钟×30fps / 2小时×30fps |
| **Dim 1** | 3 | 天线数 | RX1, RX2, RX3 |
| **Dim 2** | 16 | Chirp数 | 每帧16个chirps |
| **Dim 3** | 512 | **实数采样点** | 每个chirp 512个实数值 |

**文件大小**:
```
短时测量: 18000 × 3 × 16 × 512 × 2字节 = 884,736,128 字节 ≈ 843.75 MB
长时监测: 216000 × 3 × 16 × 512 × 2字节 = 10,616,832,000 字节 ≈ 9.89 GB
```

**数据特征**:
- **存储格式**: uint16 (0-65535)
- **数值范围**: 约 [559, 3276]
- **典型均值**: ~1865 (接近ADC中点2048)
- **数据排列**: 连续的实数序列，**不是IQ交织**
- **DC偏移**: 需要减去DC偏移 (2048或实际均值)

**重要说明**:
- ✅ 这是**实采样**数据，512个采样点就是512个实数值
- ❌ **不是**IQ复采样，不需要IQ解交织
- ✅ 使用`np.fft.rfft()`进行Range-FFT，输出257个复数值

#### 2. 参考心率文件

**HR_ref.csv**:
- **文件格式**: CSV文件
- **采样率**: 1 Hz (每秒1个采样点)
- **数据点数**:
  - 短时测量 (10分钟): 600行
  - 长时监测 (2小时): 7200行

**列定义**:

| 列名 | 含义 | 单位 | 说明 |
|------|------|------|------|
| `Time` | 时间戳 | 秒 | 从1开始的秒数 |
| `HR (bpm)` | 心率 | 次/分钟 | Bigrun心率带测量值 |
| `RR (bpm)` | 呼吸率 | 次/分钟 | 呼吸频率 |

**示例数据**:
```csv
Time,HR (bpm),RR (bpm)
1,55,17
2,55,17
3,55,17
4,55,17
5,57,17
```

#### 3. JSON配置文件

**BGT60TR13C_settings_20250423-163757.json**:

完整的雷达配置参数，包含：
- **设备信息**: 设备名称、固件版本等
- **FMCW配置**: 频率、带宽、采样率等
- **天线配置**: RX/TX天线选择
- **滤波器配置**: 高通、抗混叠滤波器

**关键字段**:
```json
{
  "device_config": {
    "fmcw_single_shape": {
      "start_frequency_Hz": 58000000000,
      "end_frequency_Hz": 63500000000,
      "num_samples_per_chirp": 512,
      "num_chirps_per_frame": 16,
      "sample_rate_Hz": 2000000,
      "frame_repetition_time_s": 0.03333333507180214,
      "rx_antennas": [1, 2, 3],
      "tx_antennas": [1],
      "mimo_mode": "off"
    }
  },
  "device_info": {
    "device_name": "BGT60TR13C",
    "firmware_version": "2.6.0"
  }
}
```

详见文件: `Dataset/BGT60TR13C/BGT60TR13C_settings_20250423-163757.json`

### 数据处理建议

#### 1. 数据加载

```python
import numpy as np

# 加载雷达数据
radar_data = np.load('radar_raw_data.npy')  # (18000, 3, 16, 512)

# 转换数据类型
radar_data = radar_data.astype(np.float32)

# DC偏移校正
dc_offset = radar_data.mean()  # 或使用2048
radar_data = radar_data - dc_offset
```

#### 2. Range-FFT处理

```python
# 使用实数FFT (因为是实采样数据)
range_fft = np.fft.rfft(radar_data, axis=-1)  # (18000, 3, 16, 257)

# 注意: rfft输出257个复数值 (512/2 + 1)
# 这是实数FFT的特性，只保留正频率部分
```

#### 3. 与4TU数据集的处理差异

| 处理步骤 | 4TU (复采样) | BGT60TR13C (实采样) |
|---------|-------------|-------------------|
| **IQ解交织** | ✅ 需要 | ❌ 不需要 |
| **Range-FFT** | `np.fft.fft()` | `np.fft.rfft()` |
| **输出点数** | 250个复数 | 257个复数 |
| **频谱范围** | 负频率到正频率 | 仅正频率 |

### 应用场景

- **长时连续监测研究**: 2小时数据可用于研究生理参数的时变特性
- **算法稳定性测试**: 验证算法在长时间运行下的稳定性
- **跨受试者泛化**: 8名受试者提供个体差异数据
- **室内环境适应**: 真实室内环境的噪声和干扰
- **跨雷达平台验证**: 与4TU（AWR1642）、PhysDrive对比，验证跨平台泛化能力
- **昼夜节律研究**: 长时数据可观察心率的自然变化

### 数据质量

- **参考设备精度**: Bigrun心率带精度±1 bpm
- **时间同步**: 雷达数据与参考心率时间戳同步
- **数据完整性**: 所有测量均完整记录，无中断
- **环境控制**: 室内环境，温度、湿度相对稳定
- **受试者配合度**: 所有受试者在测量期间保持静止

### 数据集统计

- **总受试者数**: 8人
- **短时测量**: 32个场景（8人 × 4距离，每次10分钟）
- **长时测量**: 1个场景（2小时连续）
- **测试距离**: 4种（0.3m, 0.6m, 0.9m, 1.2m）
- **总数据时长**: 约5.3小时（短时32×10分钟 + 长时2小时）
- **总数据量**: 约37 GB（短时32×844MB + 长时9.9GB）
- **心率范围**: 55-90 bpm（静息状态）
- **呼吸率范围**: 12-20 bpm
- **采集时间**: 2025年4月（根据JSON文件时间戳）

### 与其他数据集的对比

| 特性 | 4TU.ResearchD | PhysDrive | BGT60TR13C |
|------|---------------|-----------|------------|
| **雷达型号** | AWR1642 | 未公开 | BGT60TR13C |
| **频率** | 77-81 GHz | 未公开 | 58-63.5 GHz |
| **带宽** | 4 GHz | 2.6 GHz | **5.5 GHz** |
| **ADC采样点** | 250 | - | 512 |
| **Chirp数/帧** | 128 | 64 | 16 |
| **天线配置** | 4 RX × 2 TX | 12虚拟天线 | 3 RX × 1 TX |
| **帧率** | 20 fps | 20 fps | **30 fps** |
| **距离分辨率** | 3.75 cm | ~6 cm | **2.73 cm** |
| **采样方式** | **复采样 (IQ)** | 已处理RDA | **实采样** |
| **数据格式** | 原始ADC (.bin) | 已处理RDA (.mat) | 原始ADC (.npy) |
| **数据类型** | int16 | float64 | uint16 |
| **IQ排列** | 交织 | 分离 | **无（实采样）** |
| **数据形状** | (4,250,153600) | (600,2,8,16,8) | (18000,3,16,512) |
| **FFT类型** | 复数FFT | - | **实数FFT** |
| **Range bins** | 250 | 8 | **257** (rfft输出) |
| **受试者数** | 10 | 48 | 8 |
| **测试距离** | 4种 | - | **4种** |
| **场景类型** | 极端生理 | 车载驾驶 | 室内静态 |
| **最长时长** | 60秒 | 约30秒/样本 | **2小时** |
| **环境** | 实验室 | 车载 | 室内 |
| **干扰** | 受控 | 运动/说话/振动 | 受控 |
| **参考设备** | Polar H10 | 多模态 | Bigrun |
| **参考数据** | 心率+呼吸率 | ECG+呼吸 | 心率+呼吸率 |
| **特色** | 极端场景 | 真实驾驶环境 | **长时监测+多距离** |
| **单次数据量** | ~600 MB | ~9.3 MB | ~844 MB (短时) |
| **总样本数** | 456 | 1685 | 33 (32短时+1长时) |
| **总数据量** | **~267 GB** | **~15.3 GB** | **~37 GB** |

### 数据使用建议

1. **短时数据（32个场景）**:
   - 用于跨受试者泛化性测试（8人）
   - 用于跨距离泛化性测试（4种距离）
   - 用于与4TU、PhysDrive的跨数据集对比
   - 用于快速算法验证

2. **长时数据（2小时）**:
   - 用于算法长期稳定性测试
   - 用于研究心率的时变特性和昼夜节律
   - 用于测试时间自适应算法
   - 用于分析算法的漂移和退化

3. **跨平台研究**:
   - BGT60TR13C (实采样) vs AWR1642 (复采样) 的域适应
   - 不同雷达芯片的信号特性对比
   - 跨平台泛化算法开发
   - 实采样与复采样数据的处理差异研究

---

## 数据集对比总结

| 数据集 | 主要特点 | 适用研究方向 |
|--------|----------|-------------|
| **4TU.ResearchD** | 极端生理场景、复采样、多场景 | 鲁棒性、极端场景处理、复数信号处理 |
| **PhysDrive** | 车载环境、已处理RDA、多模态 | 多模态融合、实际应用、深度学习 |
| **BGT60TR13C** | 长时监测、实采样、多距离 | 长期稳定性、跨雷达域适应、实数信号处理 |

**三个数据集的协同作用**：
- **4TU**: 提供极端场景的挑战性数据，复采样IQ数据
- **PhysDrive**: 提供真实应用场景的复杂数据，已处理RDA数据
- **BGT60TR13C**: 提供长时稳定性和跨平台验证数据，实采样数据

**面向跨域场景的研究价值**：
- **跨数据集泛化**：4TU → PhysDrive → BGT60TR13C
- **跨雷达平台**：AWR1642 (77-81 GHz) → BGT60TR13C (58-63.5 GHz)
- **跨采样方式**：复采样 (4TU) → 已处理 (PhysDrive) → 实采样 (BGT60TR13C)
- **跨应用场景**：极端生理 → 车载驾驶 → 室内静态
- **跨时间尺度**：短时（60秒）→ 中时（数分钟）→ 长时（2小时）
- **跨距离场景**：固定距离 (4TU) → 车内距离 (PhysDrive) → 多距离 (BGT60TR13C)
- **跨环境条件**：实验室 → 车载 → 真实室内

这些跨域特性为"面向跨域场景的毫米波雷达生命体征检测关键技术研究"提供了全面的数据支撑。

---

## 数据加载器对比

### 关键差异总结

| 维度 | 4TU.ResearchD | PhysDrive | BGT60TR13C |
|------|---------------|-----------|------------|
| **数据级别** | 原始ADC | 已处理RDA | 原始ADC |
| **采样方式** | IQ复采样 | 已FFT处理 | **实采样** ⚠️ |
| **输出维度** | 3D (RX, Chirp, Range) | 5D (Frame, TX, Doppler, Angle, Range) | 4D (Frame, RX, Chirp, Sample) |
| **数据类型** | 复数 (complex) | 复数/实数可选 | **实数 (float32)** |
| **需要FFT** | ✅ 是 | ❌ 否（已处理） | ✅ 是（用rfft） |
| **参考数据** | 单个值 | 时间序列 | 时间序列 |
| **时间维度** | 无（单次测量） | 600帧 | 18000/216000帧 |

#### 加载器输入输出

**FTUDataLoader (4TU.ResearchData)**:
```python
# 输入
load_radar_data(
    participant_id: int,    # 1-10
    scenario: str,          # 'Distance', 'Orientation', 'Angle', 'Elevated'
    distance: str,          # 如 '80cm', '100cm'
    repeat: int            # 1-4
)

# 输出
radar_data: (4, 128, 250) complex128  # RX × Chirp × Range (复数)
ref_data: {'heart_rate': float, 'respiration_rate': float}  # 单个值
```

**PhysDriveDataLoader**:
```python
# 输入
load_radar_data(
    session_id: str,        # 如 'AFM1Z'
    sample_id: int,         # 样本编号
    return_complex: bool    # 是否返回复数（默认True）
)

# 输出
radar_data: (600, 2, 8, 16, 8) complex128  # Frame × TX × Doppler × Angle × Range
# 或 (600, 2, 8, 16, 8, 2) float64 (最后维度: [real, imag])
ref_data: {'ecg': (600,), 'resp': (600,)}  # 时间序列
```

**BGT60TR13CDataLoader**:
```python
# 输入
load_radar_data(
    participant_id: int,        # 1-8
    distance: str,              # '0.3m', '0.6m', '0.9m', '1.2m'
    measurement_type: str,      # 'short' (10分钟) 或 'long' (2小时)
    apply_dc_correction: bool   # 是否DC校正（默认True）
)

# 输出
radar_data: (18000, 3, 16, 512) float32  # Frame × RX × Chirp × Sample (实数)
# 或 (216000, 3, 16, 512) float32 (长时测量)
ref_data: {'time': array, 'heart_rate': array, 'respiration_rate': array}  # 时间序列
```

---

### 数据处理流程差异

#### 4TU.ResearchD 处理流程

```
原始ADC数据 (int16, IQ交织)
    ↓
IQ解交织 (按RX通道分离)
    ↓
转换为复数 (I + jQ)
    ↓
Range-FFT (np.fft.fft, axis=-1)
    ↓
Doppler-FFT (np.fft.fft, axis=1)
    ↓
生命体征提取
```

**关键代码**:
```python
# IQ解交织
rx0_i = raw_data[0::8].astype(np.float32)
rx0_q = raw_data[1::8].astype(np.float32)
rx0_complex = rx0_i + 1j * rx0_q

# Range-FFT (复数FFT)
range_fft = np.fft.fft(rx0_complex, axis=-1)  # 输出: 250个复数
```

---

#### PhysDrive 处理流程

```
已处理RDA数据 (float64, real/imag分离)
    ↓
转换为复数 (可选)
    ↓
直接使用 (已经是Range-Doppler-Angle)
    ↓
生命体征提取
```

**关键代码**:
```python
# 数据已经过Range-FFT、Doppler-FFT、Angle-FFT
# 直接转换为复数使用
radar_complex = radar_data[..., 0] + 1j * radar_data[..., 1]
# 形状: (600, 2, 8, 16, 8) - 已经是频域数据
```

---

#### BGT60TR13C 处理流程

```
原始ADC数据 (uint16, 实采样) ⚠️
    ↓
DC偏移校正 (减去均值)
    ↓
Range-FFT (np.fft.rfft, axis=-1) ⚠️ 注意用rfft!
    ↓
Doppler-FFT (np.fft.fft, axis=2)
    ↓
生命体征提取
```

**关键代码**:
```python
# DC校正
radar_data = radar_data - radar_data.mean()

# Range-FFT (实数FFT) ⚠️ 重要!
range_fft = np.fft.rfft(radar_data, axis=-1)  # 输出: 257个复数 (512//2+1)

# Doppler-FFT (复数FFT)
doppler_fft = np.fft.fft(range_fft, axis=2)  # 输出: 16个复数
```

---

### ⚠️ 最重要的区别

**BGT60TR13C是唯一使用实采样的数据集！**

| 数据集 | 采样方式 | 采样点数 | FFT类型 | FFT输出 |
|--------|---------|---------|---------|---------|
| **4TU** | IQ复采样 | 250个IQ对 | `np.fft.fft()` | 250个复数 |
| **PhysDrive** | 已处理 | - | 无需FFT | - |
| **BGT60TR13C** | **实采样** ⚠️ | **512个实数** | `np.fft.rfft()` ⚠️ | **257个复数** |

**常见错误**:
```python
# ❌ 错误: 对BGT60TR13C使用fft()
range_fft = np.fft.fft(radar_data, axis=-1)  # 输出512个复数 (错误!)

# ✅ 正确: 对BGT60TR13C使用rfft()
range_fft = np.fft.rfft(radar_data, axis=-1)  # 输出257个复数 (正确!)
```

**原因**:
- **实采样**: 只有实部，没有虚部（不是IQ对）
- **rfft**: Real FFT，专门用于实数输入，输出N/2+1个复数
- **fft**: 用于复数输入，输出N个复数

---

### 数据加载器使用建议

| 数据集 | 适用场景 | 注意事项 |
|--------|---------|---------|
| **4TU** | 极端生理场景、算法鲁棒性测试 | 需要IQ解交织，注意索引规律 |
| **PhysDrive** | 车载应用、多模态融合 | 数据已处理，可直接用于深度学习 |
| **BGT60TR13C** | 长时监测、跨雷达平台验证 | **必须用rfft**，注意DC校正 |

