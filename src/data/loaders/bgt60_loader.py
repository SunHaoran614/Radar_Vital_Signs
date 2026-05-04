"""BGT60TR13C数据集加载器"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import json
from .base_loader import BaseDataLoader


class BGT60TR13CDataLoader(BaseDataLoader):
    """BGT60TR13C数据集加载器。
    
    该加载器负责读取BGT60TR13C数据集的NumPy文件和CSV参考文件，
    并将其转换为标准格式供后续处理使用。
    
    数据格式说明：
        - 雷达数据: .npy文件，实采样原始ADC数据
        - 参考心率: .csv文件，包含心率和呼吸率时间序列
        - 配置文件: .json文件，包含雷达配置参数
    
    Attributes:
        NUM_SAMPLES (int): ADC采样点数，默认512
        NUM_RX (int): 接收天线数，默认3
        NUM_CHIRPS (int): 每帧chirp数，默认16
        FRAME_RATE (int): 帧率，默认30 fps
        SHORT_DURATION_FRAMES (int): 短时测量帧数，默认18000 (10分钟)
        LONG_DURATION_FRAMES (int): 长时测量帧数，默认216000 (2小时)
        DISTANCES (list): 测试距离列表
    
    Example:
        >>> loader = BGT60TR13CDataLoader('Dataset/BGT60TR13C')
        >>> radar_data = loader.load_radar_data(1, '0.3m')
        >>> ref_data = loader.load_reference_data(1, '0.3m')
        >>> print(f"Radar shape: {radar_data.shape}")
        Radar shape: (18000, 3, 16, 512)
    """
    
    # 雷达配置常量
    NUM_SAMPLES = 512
    NUM_RX = 3
    NUM_CHIRPS = 16
    FRAME_RATE = 30
    SHORT_DURATION_FRAMES = 18000  # 10分钟 × 30fps
    LONG_DURATION_FRAMES = 216000  # 2小时 × 30fps
    
    # 测试距离
    DISTANCES = ['0.3m', '0.6m', '0.9m', '1.2m']
    
    def __init__(self, dataset_path: str):
        """初始化BGT60TR13C数据加载器。
        
        Args:
            dataset_path: 数据集根目录路径
        
        Raises:
            FileNotFoundError: 如果数据集路径不存在或结构不完整
        """
        super().__init__(dataset_path)
        self.radar_data_path = self.dataset_path / 'Radar_Data'
        self.hr_ref_path = self.dataset_path / 'HR_Ref_Data'
        self.config_file = self.dataset_path / 'BGT60TR13C_settings_20250423-163757.json'
        
        self._validate_dataset_structure()
        self._load_config()
    
    def _validate_dataset_structure(self) -> None:
        """验证数据集目录结构是否完整。
        
        Raises:
            FileNotFoundError: 如果必要的目录不存在
        """
        if not self.radar_data_path.exists():
            raise FileNotFoundError(f"雷达数据目录不存在: {self.radar_data_path}")
        
        if not self.hr_ref_path.exists():
            raise FileNotFoundError(f"参考数据目录不存在: {self.hr_ref_path}")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
    
    def _load_config(self) -> None:
        """加载JSON配置文件。"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            raise IOError(f"读取配置文件失败: {self.config_file}, 错误: {e}")
    
    def load_radar_data(
        self,
        participant_id: int,
        distance: str,
        measurement_type: str = 'short',
        apply_dc_correction: bool = False
    ) -> np.ndarray:
        """加载雷达原始数据。
        
        Args:
            participant_id: 受试者ID (1-8)
            distance: 测试距离 ('0.3m', '0.6m', '0.9m', '1.2m')
            measurement_type: 测量类型 ('short'=短时10分钟, 'long'=长时2小时)
            apply_dc_correction: 是否应用DC偏移校正，默认True
        
        Returns:
            雷达数据立方体，shape为(num_frames, num_rx, num_chirps, num_samples)
            - num_frames: 18000 (短时) 或 216000 (长时)
            - num_rx: 3 (接收天线数)
            - num_chirps: 16 (每帧chirp数)
            - num_samples: 512 (实数采样点数)
        
        Raises:
            ValueError: 如果参数无效
            FileNotFoundError: 如果数据文件不存在
            IOError: 如果文件读取失败
        """
        # 参数验证
        self._validate_parameters(participant_id, distance, measurement_type)
        
        # 构建文件路径
        file_path = self._build_radar_file_path(
            participant_id, distance, measurement_type
        )
        
        # 读取NumPy文件
        try:
            radar_data = np.load(file_path)
        except Exception as e:
            raise IOError(f"读取雷达数据失败: {file_path}, 错误: {e}")
        
        # 验证数据形状
        expected_frames = (
            self.SHORT_DURATION_FRAMES if measurement_type == 'short' 
            else self.LONG_DURATION_FRAMES
        )
        expected_shape = (expected_frames, self.NUM_RX, self.NUM_CHIRPS, self.NUM_SAMPLES)
        if radar_data.shape != expected_shape:
            raise ValueError(
                f"数据形状不匹配。期望: {expected_shape}, 实际: {radar_data.shape}"
            )
        
        # 转换数据类型并应用DC校正
        radar_data = radar_data.astype(np.float32)
        if apply_dc_correction:
            radar_data = self._apply_dc_correction(radar_data)
        
        return radar_data

    def load_reference_data(
        self,
        participant_id: int,
        distance: str,
        measurement_type: str = 'short'
    ) -> Dict[str, Any]:
        """加载参考心率和呼吸率数据。

        Args:
            participant_id: 受试者ID (1-8)
            distance: 测试距离 ('0.3m', '0.6m', '0.9m', '1.2m')
            measurement_type: 测量类型 ('short'=短时, 'long'=长时)

        Returns:
            包含以下键的字典:
                - 'time': 时间戳 (秒)
                - 'heart_rate': 心率时间序列 (bpm)
                - 'respiration_rate': 呼吸率时间序列 (bpm)

        Raises:
            FileNotFoundError: 如果参考文件不存在
            IOError: 如果文件读取失败
        """
        # 构建参考文件路径
        hr_file = self._build_hr_file_path(
            participant_id, distance, measurement_type
        )

        # 读取CSV文件
        try:
            df = pd.read_csv(hr_file)
        except Exception as e:
            raise IOError(f"读取参考数据失败: {hr_file}, 错误: {e}")

        # 验证列名
        expected_columns = ['Time', 'HR (bpm)', 'RR (bpm)']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(
                f"CSV文件缺少必要的列。期望: {expected_columns}, "
                f"实际: {list(df.columns)}"
            )

        return {
            'time': df['Time'].values,
            'heart_rate': df['HR (bpm)'].values,
            'respiration_rate': df['RR (bpm)'].values
        }

    def _validate_parameters(
        self,
        participant_id: int,
        distance: str,
        measurement_type: str
    ) -> None:
        """验证输入参数的有效性。

        Args:
            participant_id: 受试者ID
            distance: 测试距离
            measurement_type: 测量类型

        Raises:
            ValueError: 如果参数无效
        """
        if not 1 <= participant_id <= 8:
            raise ValueError(
                f"participant_id必须在1-8之间，得到: {participant_id}"
            )

        if distance not in self.DISTANCES:
            raise ValueError(
                f"distance必须是{self.DISTANCES}之一，得到: {distance}"
            )

        if measurement_type not in ['short', 'long']:
            raise ValueError(
                f"measurement_type必须是'short'或'long'，得到: {measurement_type}"
            )

    def _build_radar_file_path(
        self,
        participant_id: int,
        distance: str,
        measurement_type: str
    ) -> Path:
        """构建雷达数据文件路径。

        Args:
            participant_id: 受试者ID
            distance: 测试距离
            measurement_type: 测量类型

        Returns:
            文件完整路径

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if measurement_type == 'short':
            file_path = (
                self.radar_data_path /
                f'Participant{participant_id}' /
                distance /
                'radar_raw_data.npy'
            )
        else:  # long
            file_path = (
                self.radar_data_path /
                'Long_duration' /
                'radar_raw_data.npy'
            )

        if not file_path.exists():
            raise FileNotFoundError(f"雷达数据文件不存在: {file_path}")

        return file_path

    def _build_hr_file_path(
        self,
        participant_id: int,
        distance: str,
        measurement_type: str
    ) -> Path:
        """构建心率参考文件路径。

        Args:
            participant_id: 受试者ID
            distance: 测试距离
            measurement_type: 测量类型

        Returns:
            文件完整路径

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if measurement_type == 'short':
            file_path = (
                self.hr_ref_path /
                f'Participant{participant_id}' /
                distance /
                'HR_ref.csv'
            )
        else:  # long
            file_path = (
                self.hr_ref_path /
                'Long_duration' /
                'HR_ref.csv'
            )

        if not file_path.exists():
            raise FileNotFoundError(f"参考数据文件不存在: {file_path}")

        return file_path

    def _apply_dc_correction(self, radar_data: np.ndarray) -> np.ndarray:
        """应用DC偏移校正。

        Args:
            radar_data: 原始雷达数据

        Returns:
            校正后的雷达数据
        """
        # 计算DC偏移（全局均值）
        dc_offset = radar_data.mean()

        # 减去DC偏移
        return radar_data - dc_offset

    def list_participants(self) -> List[int]:
        """列出所有可用的受试者ID。

        Returns:
            受试者ID列表
        """
        participants = []
        for i in range(1, 9):
            participant_dir = self.radar_data_path / f'Participant{i}'
            if participant_dir.exists():
                participants.append(i)
        return participants

    def list_available_measurements(
        self,
        participant_id: int
    ) -> Dict[str, List[str]]:
        """列出指定受试者的所有可用测量。

        Args:
            participant_id: 受试者ID

        Returns:
            包含可用距离的字典

        Example:
            >>> loader.list_available_measurements(1)
            {'short': ['0.3m', '0.6m', '0.9m', '1.2m'], 'long': []}
        """
        available = {'short': [], 'long': []}

        # 检查短时测量
        participant_dir = self.radar_data_path / f'Participant{participant_id}'
        if participant_dir.exists():
            for distance in self.DISTANCES:
                distance_dir = participant_dir / distance
                radar_file = distance_dir / 'radar_raw_data.npy'
                if radar_file.exists():
                    available['short'].append(distance)

        # 检查长时测量
        long_dir = self.radar_data_path / 'Long_duration'
        if long_dir.exists() and (long_dir / 'radar_raw_data.npy').exists():
            available['long'] = ['long_duration']

        return available

    def get_config(self) -> Dict[str, Any]:
        """获取雷达配置参数。

        Returns:
            配置参数字典
        """
        return self.config

    def get_radar_params(self) -> Dict[str, Any]:
        """获取雷达关键参数。

        Returns:
            包含雷达参数的字典
        """
        device_config = self.config.get('device_config', {}).get('fmcw_single_shape', {})

        return {
            'start_frequency_Hz': device_config.get('start_frequency_Hz'),
            'end_frequency_Hz': device_config.get('end_frequency_Hz'),
            'bandwidth_Hz': (
                device_config.get('end_frequency_Hz', 0) -
                device_config.get('start_frequency_Hz', 0)
            ),
            'num_samples_per_chirp': device_config.get('num_samples_per_chirp'),
            'num_chirps_per_frame': device_config.get('num_chirps_per_frame'),
            'sample_rate_Hz': device_config.get('sample_rate_Hz'),
            'frame_repetition_time_s': device_config.get('frame_repetition_time_s'),
            'frame_rate_fps': (
                1.0 / device_config.get('frame_repetition_time_s', 1.0)
                if device_config.get('frame_repetition_time_s') else None
            ),
            'rx_antennas': device_config.get('rx_antennas'),
            'tx_antennas': device_config.get('tx_antennas'),
            'mimo_mode': device_config.get('mimo_mode'),
            'if_gain_dB': device_config.get('if_gain_dB')
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集详细信息。

        Returns:
            包含数据集信息的字典
        """
        # 统计受试者和测量
        participants = self.list_participants()
        total_short_measurements = 0

        for participant_id in participants:
            measurements = self.list_available_measurements(participant_id)
            total_short_measurements += len(measurements['short'])

        # 检查长时测量
        has_long_measurement = (
            self.radar_data_path / 'Long_duration' / 'radar_raw_data.npy'
        ).exists()

        return {
            'dataset_name': 'BGT60TR13C',
            'dataset_path': str(self.dataset_path),
            'total_participants': len(participants),
            'participants': participants,
            'total_short_measurements': total_short_measurements,
            'has_long_measurement': has_long_measurement,
            'available_distances': self.DISTANCES,
            'data_config': {
                'num_samples': self.NUM_SAMPLES,
                'num_rx': self.NUM_RX,
                'num_chirps': self.NUM_CHIRPS,
                'frame_rate': self.FRAME_RATE,
                'short_duration_frames': self.SHORT_DURATION_FRAMES,
                'long_duration_frames': self.LONG_DURATION_FRAMES,
                'sampling_type': 'Real Sampling',
                'data_format': 'Raw ADC (.npy)'
            },
            'radar_params': self.get_radar_params()
        }

