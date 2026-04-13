"""PhysDrive数据集加载器"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import scipy.io as sio

from .base_loader import BaseDataLoader


class PhysDriveDataLoader(BaseDataLoader):
    """PhysDrive数据集加载器。
    
    该加载器负责读取PhysDrive数据集的MATLAB文件，
    并将其转换为标准格式供后续处理使用。
    
    数据格式说明：
        - 雷达数据: .mat文件，已处理的Range-Doppler-Angle数据
        - 参考ECG: .mat文件，心电图信号
        - 参考呼吸: .mat文件，呼吸信号
    
    Attributes:
        NUM_FRAMES (int): 时间帧数，默认600
        NUM_DOPPLER (int): Doppler bins，默认8
        NUM_ANGLE (int): Angle bins，默认16
        NUM_RANGE (int): Range bins，默认8
        SEGMENT_MAP (dict): 路段类型映射
        GENDER_MAP (dict): 性别映射
        TIME_MAP (dict): 时间/天气映射
    
    Example:
        >>> loader = PhysDriveDataLoader('Dataset/PhysDrive')
        >>> radar_data = loader.load_radar_data('AFH1', 0)
        >>> ref_data = loader.load_reference_data('AFH1', 0)
        >>> print(f"Radar shape: {radar_data.shape}")
        Radar shape: (600, 8, 16, 8)
    """
    
    # 数据维度常量
    NUM_FRAMES = 600
    NUM_DOPPLER = 8
    NUM_ANGLE = 16
    NUM_RANGE = 8
    
    # 会话ID解析映射
    SEGMENT_MAP = {
        'A': 'Flat & Unobstructed',
        'B': 'Flat & Congested',
        'C': 'Bumpy & Congested'
    }
    
    GENDER_MAP = {
        'M': 'Male',
        'F': 'Female'
    }
    
    TIME_MAP = {
        'Z': 'Noon',
        'H': 'Dusk/Early Morning',
        'W': 'Midnight',
        'Y': 'Rainy/Cloudy'
    }
    
    def __init__(self, dataset_path: str):
        """初始化PhysDrive数据加载器。
        
        Args:
            dataset_path: 数据集根目录路径
        
        Raises:
            FileNotFoundError: 如果数据集路径不存在或结构不完整
        """
        super().__init__(dataset_path)
        self.mmwave_path = self.dataset_path / 'mmWave'
        
        self._validate_dataset_structure()
    
    def _validate_dataset_structure(self) -> None:
        """验证数据集目录结构是否完整。
        
        Raises:
            FileNotFoundError: 如果缺少必要的目录
        """
        if not self.mmwave_path.exists():
            raise FileNotFoundError(
                f"数据集目录结构不完整，缺少: {self.mmwave_path}"
            )
    
    def load_radar_data(
        self,
        session_id: str,
        sample_id: int,
        return_complex: bool = True,
        normalize_frames: bool = True,
    ) -> np.ndarray:
        """加载雷达数据。
        
        Args:
            session_id: 会话ID (如 'AFH1')
            sample_id: 样本ID (如 0 对应 AFH1_00)
            return_complex: 是否返回复数格式，默认True
            normalize_frames: 是否将帧数标准化到600（短样本补零、长样本截断），默认True
        
        Returns:
            雷达数据立方体:
                - 如果return_complex=True: shape为(600, 8, 16, 8)，复数类型
                - 如果return_complex=False: shape为(600, 2, 8, 16, 8)，实数类型
        
        Raises:
            ValueError: 如果参数无效
            FileNotFoundError: 如果数据文件不存在
            IOError: 如果文件读取失败
        """
        # 参数验证
        self._validate_session_id(session_id)
        
        # 构建文件路径
        file_path = self._build_sample_file_path(session_id, sample_id, 'mmwave.mat')
        
        # 读取MAT文件
        try:
            mat_data = sio.loadmat(file_path)
        except Exception as e:
            raise IOError(f"读取雷达数据失败: {file_path}, 错误: {e}")
        
        # 提取mmwave数据
        if 'mmwave' not in mat_data:
            raise ValueError(f"MAT文件中缺少'mmwave'键: {file_path}")
        
        mmwave = mat_data['mmwave']
        
        # 验证数据形状
        expected_tail = (2, self.NUM_DOPPLER, self.NUM_ANGLE, self.NUM_RANGE)
        if mmwave.ndim != 5 or mmwave.shape[1:] != expected_tail:
            expected_shape = (self.NUM_FRAMES,) + expected_tail
            raise ValueError(
                f"数据形状不匹配。期望: {expected_shape}, 实际: {mmwave.shape}"
            )

        # 仅帧数不一致时，可选择标准化到固定帧长
        if mmwave.shape[0] != self.NUM_FRAMES:
            if normalize_frames:
                mmwave = self._normalize_mmwave_frames(mmwave, target_frames=self.NUM_FRAMES)
            else:
                expected_shape = (self.NUM_FRAMES,) + expected_tail
                raise ValueError(
                    f"数据形状不匹配。期望: {expected_shape}, 实际: {mmwave.shape}"
                )
        
        # 转换为复数格式
        if return_complex:
            return self._convert_to_complex(mmwave)
        else:
            return mmwave

    def load_reference_data(
        self,
        session_id: str,
        sample_id: int
    ) -> Dict[str, Any]:
        """加载参考数据（ECG和呼吸）。

        Args:
            session_id: 会话ID (如 'AFH1')
            sample_id: 样本ID (如 0)

        Returns:
            包含以下键的字典:
                - 'ecg': ECG信号，shape为(600,)
                - 'respiration': 呼吸信号，shape为(600,)

        Raises:
            FileNotFoundError: 如果参考文件不存在
            IOError: 如果文件读取失败
        """
        # 构建文件路径
        ecg_file = self._build_sample_file_path(session_id, sample_id, 'ecg.mat')
        resp_file = self._build_sample_file_path(session_id, sample_id, 'resp.mat')

        # 读取ECG数据
        try:
            ecg_data = sio.loadmat(ecg_file)
            ecg = ecg_data['ecg'].flatten()  # (1, 600) -> (600,)
        except Exception as e:
            raise IOError(f"读取ECG数据失败: {ecg_file}, 错误: {e}")

        # 读取呼吸数据
        try:
            resp_data = sio.loadmat(resp_file)
            resp = resp_data['resp'].flatten()  # (1, 600) -> (600,)
        except Exception as e:
            raise IOError(f"读取呼吸数据失败: {resp_file}, 错误: {e}")

        return {
            'ecg': ecg,
            'respiration': resp
        }

    def parse_session_id(self, session_id: str) -> Dict[str, Any]:
        """解析会话ID，提取场景信息。

        Args:
            session_id: 会话ID (如 'AFH1')

        Returns:
            包含以下键的字典:
                - 'segment': 路段类型代码 ('A'/'B'/'C')
                - 'segment_name': 路段类型名称
                - 'gender': 性别代码 ('M'/'F')
                - 'gender_name': 性别名称
                - 'time': 时间/天气代码 ('Z'/'H'/'W'/'Y')
                - 'time_name': 时间/天气名称
                - 'repeat': 重复次数 (1/2/3/4)

        Raises:
            ValueError: 如果会话ID格式无效
        """
        # 验证长度
        if len(session_id) != 4:
            raise ValueError(
                f"会话ID必须是4个字符，得到: {session_id} (长度: {len(session_id)})"
            )

        # 解析每个字符
        segment = session_id[0]
        gender = session_id[1]
        time = session_id[2]
        repeat_str = session_id[3]

        # 验证路段类型
        if segment not in self.SEGMENT_MAP:
            raise ValueError(
                f"无效的路段类型: {segment}，必须是 {list(self.SEGMENT_MAP.keys())} 之一"
            )

        # 验证性别
        if gender not in self.GENDER_MAP:
            raise ValueError(
                f"无效的性别: {gender}，必须是 {list(self.GENDER_MAP.keys())} 之一"
            )

        # 验证时间/天气
        if time not in self.TIME_MAP:
            raise ValueError(
                f"无效的时间/天气: {time}，必须是 {list(self.TIME_MAP.keys())} 之一"
            )

        # 验证重复次数
        try:
            repeat = int(repeat_str)
        except ValueError:
            raise ValueError(f"无效的重复次数: {repeat_str}，必须是数字")

        return {
            'segment': segment,
            'segment_name': self.SEGMENT_MAP[segment],
            'gender': gender,
            'gender_name': self.GENDER_MAP[gender],
            'time': time,
            'time_name': self.TIME_MAP[time],
            'repeat': repeat
        }

    def list_sessions(self) -> List[str]:
        """列出所有可用的会话ID。

        Returns:
            会话ID列表，按字母顺序排序

        Example:
            >>> loader.list_sessions()
            ['AFH1', 'AFH2', 'AMH1', ...]
        """
        sessions = []
        for session_dir in self.mmwave_path.iterdir():
            if session_dir.is_dir():
                sessions.append(session_dir.name)
        return sorted(sessions)

    def list_samples(self, session_id: str) -> List[int]:
        """列出指定会话的所有样本ID。

        Args:
            session_id: 会话ID (如 'AFH1')

        Returns:
            样本ID列表，按数字顺序排序

        Example:
            >>> loader.list_samples('AFH1')
            [0, 5, 6, 10, 14, ...]

        Raises:
            FileNotFoundError: 如果会话不存在
        """
        session_path = self.mmwave_path / session_id

        if not session_path.exists():
            raise FileNotFoundError(f"会话不存在: {session_id}")

        samples = []
        for sample_dir in session_path.iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith(session_id + '_'):
                # 提取样本编号，如 'AFH1_00' -> 0, 'AFH1_118' -> 118
                sample_str = sample_dir.name.split('_')[1]
                try:
                    sample_num = int(sample_str)
                    samples.append(sample_num)
                except ValueError:
                    # 跳过无效的目录名
                    continue

        return sorted(samples)

    def filter_sessions(
        self,
        segment: Optional[str] = None,
        gender: Optional[str] = None,
        time: Optional[str] = None,
        repeat: Optional[int] = None
    ) -> List[str]:
        """根据条件筛选会话。

        Args:
            segment: 路段类型 ('A'/'B'/'C')，None表示不筛选
            gender: 性别 ('M'/'F')，None表示不筛选
            time: 时间/天气 ('Z'/'H'/'W'/'Y')，None表示不筛选
            repeat: 重复次数 (1/2/3/4)，None表示不筛选

        Returns:
            符合条件的会话ID列表

        Example:
            >>> loader.filter_sessions(gender='F', time='H')
            ['AFH1', 'AFH2', 'BFH1', 'BFH2', 'CFH1', 'CFH2']

        Raises:
            ValueError: 如果筛选条件无效
        """
        # 验证筛选条件
        if segment is not None and segment not in self.SEGMENT_MAP:
            raise ValueError(
                f"无效的路段类型: {segment}，必须是 {list(self.SEGMENT_MAP.keys())} 之一"
            )

        if gender is not None and gender not in self.GENDER_MAP:
            raise ValueError(
                f"无效的性别: {gender}，必须是 {list(self.GENDER_MAP.keys())} 之一"
            )

        if time is not None and time not in self.TIME_MAP:
            raise ValueError(
                f"无效的时间/天气: {time}，必须是 {list(self.TIME_MAP.keys())} 之一"
            )

        # 获取所有会话
        all_sessions = self.list_sessions()

        # 筛选
        filtered = []
        for session in all_sessions:
            try:
                info = self.parse_session_id(session)

                # 检查每个条件
                if segment is not None and info['segment'] != segment:
                    continue
                if gender is not None and info['gender'] != gender:
                    continue
                if time is not None and info['time'] != time:
                    continue
                if repeat is not None and info['repeat'] != repeat:
                    continue

                filtered.append(session)
            except ValueError:
                # 跳过无效的会话ID
                continue

        return filtered

    def _validate_session_id(self, session_id: str) -> None:
        """验证会话ID的有效性。

        Args:
            session_id: 会话ID

        Raises:
            ValueError: 如果会话ID无效
        """
        # 使用parse_session_id进行验证
        try:
            self.parse_session_id(session_id)
        except ValueError as e:
            raise ValueError(f"无效的会话ID '{session_id}': {e}")

    def _build_sample_file_path(
        self,
        session_id: str,
        sample_id: int,
        filename: str
    ) -> Path:
        """构建样本文件路径。

        Args:
            session_id: 会话ID (如 'AFH1')
            sample_id: 样本ID (如 0)
            filename: 文件名 (如 'mmwave.mat')

        Returns:
            文件完整路径

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        session_dir = self.mmwave_path / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"会话目录不存在: {session_dir}")

        # 优先尝试常见两位格式
        candidates = [session_dir / f"{session_id}_{sample_id:02d}" / filename]
        # 兼容三位及以上前导零目录（如 AFZ2_010）
        for d in session_dir.iterdir():
            if not d.is_dir() or not d.name.startswith(session_id + "_"):
                continue
            suffix = d.name.split("_", 1)[1]
            if suffix.isdigit() and int(suffix) == sample_id:
                candidates.append(d / filename)

        for p in candidates:
            if p.exists():
                return p

        # 保留最可读的默认路径用于报错
        raise FileNotFoundError(f"文件不存在: {candidates[0]}")

    def _normalize_mmwave_frames(
        self,
        mmwave: np.ndarray,
        target_frames: int,
    ) -> np.ndarray:
        """将 mmwave 的帧数标准化到 target_frames。"""
        cur = int(mmwave.shape[0])
        if cur == target_frames:
            return mmwave
        if cur > target_frames:
            return mmwave[:target_frames]

        # 帧数不足时尾部补零，保持数据语义最小侵入
        pad_shape = (target_frames - cur,) + tuple(mmwave.shape[1:])
        pad = np.zeros(pad_shape, dtype=mmwave.dtype)
        return np.concatenate([mmwave, pad], axis=0)

    def _convert_to_complex(self, mmwave: np.ndarray) -> np.ndarray:
        """将实部/虚部分离的数据转换为复数格式。

        Args:
            mmwave: 形状为(600, 2, 8, 16, 8)的数组
                   其中第2维: [0]=实部, [1]=虚部

        Returns:
            形状为(600, 8, 16, 8)的复数数组
        """
        real_part = mmwave[:, 0, :, :, :]  # (600, 8, 16, 8)
        imag_part = mmwave[:, 1, :, :, :]  # (600, 8, 16, 8)

        complex_data = real_part + 1j * imag_part

        return complex_data

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集详细信息。

        Returns:
            包含数据集信息的字典
        """
        # 统计会话数量
        all_sessions = self.list_sessions()

        # 按条件统计
        segment_counts = {}
        for seg in self.SEGMENT_MAP.keys():
            segment_counts[seg] = len(self.filter_sessions(segment=seg))

        gender_counts = {}
        for gen in self.GENDER_MAP.keys():
            gender_counts[gen] = len(self.filter_sessions(gender=gen))

        time_counts = {}
        for t in self.TIME_MAP.keys():
            time_counts[t] = len(self.filter_sessions(time=t))

        return {
            'dataset_name': 'PhysDrive',
            'dataset_path': str(self.dataset_path),
            'total_sessions': len(all_sessions),
            'sessions': all_sessions,
            'segment_distribution': segment_counts,
            'gender_distribution': gender_counts,
            'time_distribution': time_counts,
            'data_config': {
                'num_frames': self.NUM_FRAMES,
                'num_doppler': self.NUM_DOPPLER,
                'num_angle': self.NUM_ANGLE,
                'num_range': self.NUM_RANGE,
                'frequency': '60 GHz',
                'data_format': 'Processed RDA (Range-Doppler-Angle)'
            }
        }
