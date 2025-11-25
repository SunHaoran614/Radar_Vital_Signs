"""BGT60TR13C数据加载器使用示例"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders.bgt60_loader import BGT60TR13CDataLoader


def example_basic_usage():
    """示例1: 基本使用"""
    print("=" * 80)
    print("示例1: 基本数据加载")
    print("=" * 80)
    
    # 初始化加载器
    loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
    
    # 加载短时雷达数据
    radar_data = loader.load_radar_data(
        participant_id=1,
        distance='0.3m',
        measurement_type='short',
        apply_dc_correction=True
    )
    
    # 加载参考数据
    ref_data = loader.load_reference_data(
        participant_id=1,
        distance='0.3m',
        measurement_type='short'
    )
    
    print(f"\n雷达数据形状: {radar_data.shape}")
    print(f"  - 帧数: {radar_data.shape[0]}")
    print(f"  - 天线数: {radar_data.shape[1]}")
    print(f"  - Chirp数: {radar_data.shape[2]}")
    print(f"  - 采样点数: {radar_data.shape[3]}")
    
    print(f"\n参考数据:")
    print(f"  - 时间点数: {len(ref_data['time'])}")
    print(f"  - 心率范围: {ref_data['heart_rate'].min():.0f}-{ref_data['heart_rate'].max():.0f} bpm")
    print(f"  - 呼吸率范围: {ref_data['respiration_rate'].min():.0f}-{ref_data['respiration_rate'].max():.0f} bpm")


def example_dataset_exploration():
    """示例2: 数据集探索"""
    print("\n" + "=" * 80)
    print("示例2: 数据集探索")
    print("=" * 80)
    
    loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
    
    # 获取数据集信息
    info = loader.get_dataset_info()
    print(f"\n数据集: {info['dataset_name']}")
    print(f"受试者数: {info['total_participants']}")
    print(f"短时测量数: {info['total_short_measurements']}")
    print(f"长时测量: {'是' if info['has_long_measurement'] else '否'}")
    
    # 列出所有受试者
    participants = loader.list_participants()
    print(f"\n可用受试者: {participants}")
    
    # 查看每个受试者的测量
    print("\n各受试者的可用测量:")
    for participant_id in participants[:3]:  # 只显示前3个
        measurements = loader.list_available_measurements(participant_id)
        print(f"  Participant{participant_id}:")
        print(f"    短时: {measurements['short']}")
        print(f"    长时: {measurements['long']}")


def example_config_inspection():
    """示例3: 配置参数查看"""
    print("\n" + "=" * 80)
    print("示例3: 雷达配置参数")
    print("=" * 80)
    
    loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
    
    # 获取雷达参数
    params = loader.get_radar_params()
    
    print("\n雷达配置:")
    print(f"  起始频率: {params['start_frequency_Hz'] / 1e9:.1f} GHz")
    print(f"  结束频率: {params['end_frequency_Hz'] / 1e9:.1f} GHz")
    print(f"  带宽: {params['bandwidth_Hz'] / 1e9:.1f} GHz")
    print(f"  采样点/chirp: {params['num_samples_per_chirp']}")
    print(f"  Chirp数/帧: {params['num_chirps_per_frame']}")
    print(f"  帧率: {params['frame_rate_fps']:.1f} fps")
    print(f"  接收天线: {params['rx_antennas']}")
    print(f"  发射天线: {params['tx_antennas']}")
    
    # 计算距离分辨率
    c = 3e8  # 光速
    bandwidth = params['bandwidth_Hz']
    range_resolution = c / (2 * bandwidth)
    print(f"\n计算参数:")
    print(f"  距离分辨率: {range_resolution * 100:.2f} cm")


def example_data_processing():
    """示例4: 数据处理"""
    print("\n" + "=" * 80)
    print("示例4: 数据处理示例")
    print("=" * 80)
    
    loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
    
    # 加载数据
    radar_data = loader.load_radar_data(1, '0.3m', 'short')
    
    # 选择第一个天线的数据
    antenna_0_data = radar_data[:, 0, :, :]  # (18000, 16, 512)
    
    print(f"\n天线0数据形状: {antenna_0_data.shape}")
    
    # 对第一帧进行Range-FFT (实数FFT)
    first_frame = antenna_0_data[0, :, :]  # (16, 512)
    range_fft = np.fft.rfft(first_frame, axis=-1)  # (16, 257)
    
    print(f"Range-FFT结果形状: {range_fft.shape}")
    print(f"  - Chirp数: {range_fft.shape[0]}")
    print(f"  - Range bins: {range_fft.shape[1]} (实数FFT输出)")
    
    # 计算幅度谱
    range_profile = np.abs(range_fft)
    print(f"\nRange profile统计:")
    print(f"  最大值: {range_profile.max():.2f}")
    print(f"  均值: {range_profile.mean():.2f}")


def example_long_duration():
    """示例5: 长时数据加载"""
    print("\n" + "=" * 80)
    print("示例5: 长时数据加载")
    print("=" * 80)
    
    loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
    
    # 检查是否有长时数据
    info = loader.get_dataset_info()
    if not info['has_long_measurement']:
        print("\n长时数据不可用")
        return
    
    print("\n加载长时数据 (2小时)...")
    print("注意: 这可能需要一些时间...")
    
    # 加载长时雷达数据
    radar_data = loader.load_radar_data(
        participant_id=1,  # 长时数据不使用此参数
        distance='0.3m',   # 长时数据不使用此参数
        measurement_type='long'
    )
    
    print(f"\n长时雷达数据形状: {radar_data.shape}")
    print(f"  总帧数: {radar_data.shape[0]:,}")
    print(f"  时长: {radar_data.shape[0] / 30 / 60:.1f} 分钟")
    
    # 加载长时参考数据
    ref_data = loader.load_reference_data(
        participant_id=1,
        distance='0.3m',
        measurement_type='long'
    )
    
    print(f"\n长时参考数据:")
    print(f"  时间点数: {len(ref_data['time']):,}")
    print(f"  心率范围: {ref_data['heart_rate'].min():.0f}-{ref_data['heart_rate'].max():.0f} bpm")


if __name__ == '__main__':
    example_basic_usage()
    example_dataset_exploration()
    example_config_inspection()
    example_data_processing()
    example_long_duration()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)

