"""Nature Scientific Data mmwave-897-2026 数据集加载器。

数据来源: Parralejo et al., "Extensive Age-Balanced and Subject-Varied mmWave
Radar Dataset of Referenced Records for Vital Signs", Scientific Data 13:897 (2026).
DOI: 10.1038/s41597-026-07172-9

数据格式说明:
    - 60.25 GHz TI FMCW 雷达 (IWR6843ISK-ODS), 2TX x 4RX = 8 虚拟天线
    - radar_rFFTs.zlib: pickle(zlib) -> [complex (frames, 8, 64), float (64,)]
        * element[0]: 距离 FFT 后的复数立方体 (frames, 虚拟天线, range bins)
          （32 chirps 已在存储时平均 -> 零多普勒分量; 帧率 10 Hz）
        * element[1]: 每个 range bin 的物理距离 (米), shape (64,)
    - radar_timestamps.csv: 首帧接收时刻 + 配置周期生成的时间轴, 100 ms 间隔；
      并非每帧独立实测时刻。规则时间戳不保证内容更新或硬件同步。
    - radar_chirpConfig.json: 雷达配置
    - movesense_ecg.csv: 'Timestamp,mV', 256 Hz
    - movesense_acc.csv: 加速度计参考（其采样率需另按时间戳核验）
    - non_breathing_ts.csv: 仅 Rest 场景, 屏息区间 [start_s, end_s]

目录结构:
    <root>/P001/Lying/{Rest,Post-exercise}/...
    <root>/P001/Sitting/{Rest,Post-exercise}/...

注意:
    - 记录时长可变: 400 / 500 / 600 帧 (40 / 50 / 60 秒)
    - 该数据集只有距离 FFT 结果（无原始 ADC chirp 数据），
      故慢时间维 = 帧维 (10 Hz)，无 Doppler FFT 维。
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle
import zlib

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


class MMWaveDataLoader(BaseDataLoader):
    """mmwave-897-2026 (Nature Sci Data) 数据集加载器。"""

    # 雷达配置 (来自 radar_chirpConfig.json)
    NUM_RANGE_BINS = 64
    NUM_ANTENNAS = 8          # 2TX x 4RX 虚拟天线
    FRAME_RATE_HZ = 10.0
    ECG_RATE_HZ = 256.0

    POSTURES = ('Lying', 'Sitting')
    CONDITIONS = ('Rest', 'Post-exercise')

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    # ------------------------------------------------------------------
    # 路径构建
    # ------------------------------------------------------------------
    def _participant_dir(self, participant_id: int) -> Path:
        if not 1 <= participant_id <= 999:
            raise ValueError(f"participant_id 无效: {participant_id}")
        return self.dataset_path / f"P{participant_id:03d}"

    def _session_dir(self, participant_id: int, posture: str,
                     condition: str) -> Path:
        if posture not in self.POSTURES:
            raise ValueError(
                f"posture 必须是 {self.POSTURES}, 收到: {posture}")
        if condition not in self.CONDITIONS:
            raise ValueError(
                f"condition 必须是 {self.CONDITIONS}, 收到: {condition}")
        d = self._participant_dir(participant_id) / posture / condition
        if not d.exists():
            raise FileNotFoundError(f"会话目录不存在: {d}")
        return d

    def _load_zlib(self, session_dir: Path, name: str = 'radar_rFFTs.zlib'):
        """解压并反序列化 .zlib 文件。"""
        path = session_dir / name
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, 'rb') as f:
            return pickle.loads(zlib.decompress(f.read()))

    # ------------------------------------------------------------------
    # 雷达数据
    # ------------------------------------------------------------------
    def load_radar_data(
        self,
        participant_id: int,
        posture: str,
        condition: str,
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        """加载距离 FFT 复数立方体。

        Args:
            participant_id: 参与者编号 (1-110)
            posture: 'Lying' | 'Sitting'
            condition: 'Rest' | 'Post-exercise'
            max_frames: 仅加载前 N 帧（快速验证用），默认全部

        Returns:
            complex ndarray, shape (frames, 8, 64) = (帧, 虚拟天线, range bins)
        """
        session_dir = self._session_dir(participant_id, posture, condition)
        cube = self._load_zlib(session_dir, 'radar_rFFTs.zlib')[0]
        if max_frames is not None:
            cube = cube[:max_frames]
        return np.asarray(cube)

    def load_range_bins(self, participant_id: int, posture: str,
                        condition: str) -> np.ndarray:
        """Return the physical FFT-bin centres in metres.

        The archive also stores ``rBins`` inside ``radar_rFFTs.zlib``.  In
        the released records that vector is linearly spaced from zero to
        ``R_MAX`` (therefore its last-to-first spacing is ``R_MAX/(N-1)``),
        while the chirp configuration defines the actual FFT spacing as
        ``R_BIN`` and ``R_MAX == N * R_BIN``.  Using the serialized vector
        silently shifts range-based ROI selection by one bin over the full
        axis.  Return the configuration-derived centres here and expose the
        serialized vector through :meth:`load_published_range_bins` for
        provenance/debugging.
        """
        session_dir = self._session_dir(participant_id, posture, condition)
        published = self.load_published_range_bins(participant_id, posture,
                                                    condition)
        config = self.load_chirp_config(participant_id, posture, condition)
        n = len(published)
        try:
            step = float(config['R_BIN'])
            end = float(config['R_MAX'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Nature chirp config lacks numeric R_BIN/R_MAX') from exc
        if n < 2 or not np.isfinite(step) or not np.isfinite(end) or step <= 0:
            raise ValueError(f'Invalid Nature range configuration: n={n}, R_BIN={step}, R_MAX={end}')
        if not np.isclose(end, n * step, rtol=1e-6, atol=1e-9):
            raise ValueError(f'Expected R_MAX == n*R_BIN, got {end} vs {n * step}')
        if not np.all(np.isfinite(published)):
            raise ValueError('Serialized Nature range axis contains non-finite values')
        expected_published = np.linspace(0.0, end, n)
        if not np.allclose(published, expected_published, rtol=1e-6, atol=1e-7):
            raise ValueError('Serialized Nature range axis has an unexpected layout')
        return np.arange(n, dtype=np.float64) * step

    def load_published_range_bins(self, participant_id: int, posture: str,
                                  condition: str) -> np.ndarray:
        """Return the serialized ``rBins`` vector exactly as published.

        This is intentionally separate from :meth:`load_range_bins`, whose
        public contract is the configuration-derived physical axis.
        """
        session_dir = self._session_dir(participant_id, posture, condition)
        return np.asarray(self._load_zlib(session_dir, 'radar_rFFTs.zlib')[1],
                          dtype=np.float64)

    def load_radar_frame_support(self, participant_id: int, posture: str,
                                 condition: str) -> Dict[str, np.ndarray]:
        """Return radar-only zero/repeated/anchor masks on original frame IDs.

        ``repeated[i]`` compares the complete raw frame with frame i-1;
        ``anchor`` retains nonzero frames differing from their predecessor.
        This optional diagnostic does not repair, remove or compress frames,
        read reference labels, or certify acquisition freshness. Full-frame
        repetition exists in the official data despite its regular time grid.
        """
        from src.radar.frame_support import frame_support

        cube = self.load_radar_data(participant_id, posture, condition)
        return frame_support(cube)

    def load_radar_timestamps(self, participant_id: int, posture: str,
                              condition: str) -> np.ndarray:
        """发布的配置周期时间轴 (datetime64[ns])，不是逐帧实测时刻。"""
        session_dir = self._session_dir(participant_id, posture, condition)
        ts = pd.read_csv(session_dir / 'radar_timestamps.csv', header=None)
        return pd.to_datetime(ts.iloc[:, 0]).to_numpy()

    def load_chirp_config(self, participant_id: int, posture: str,
                          condition: str) -> Dict[str, Any]:
        session_dir = self._session_dir(participant_id, posture, condition)
        with open(session_dir / 'radar_chirpConfig.json') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # 参考数据
    # ------------------------------------------------------------------
    def load_reference_data(
        self,
        participant_id: int,
        posture: str,
        condition: str
    ) -> Dict[str, Any]:
        """加载 ECG / 加速度计 / 屏息区间等参考数据。

        Returns:
            dict:
                ecg_timestamps: (N,) datetime64[ns]
                ecg_mv:         (N,) float, ECG 幅值 (mV)
                acc_timestamps: (M,) datetime64[ns]（若无则 None）
                acc:            (M, 3) float（若无则 None）
                non_breathing:  [(start_s, end_s), ...] 屏息区间（Rest 场景）
                fs_ecg:         256.0
        """
        session_dir = self._session_dir(participant_id, posture, condition)

        ecg = pd.read_csv(session_dir / 'movesense_ecg.csv')
        ecg_ts = pd.to_datetime(ecg['Timestamp']).to_numpy()
        ecg_mv = ecg['mV'].to_numpy(dtype=float)

        acc = None
        acc_ts = None
        acc_path = session_dir / 'movesense_acc.csv'
        if acc_path.exists():
            try:
                acc_df = pd.read_csv(acc_path)
                acc_ts = pd.to_datetime(acc_df.iloc[:, 0]).to_numpy()
                acc = acc_df.iloc[:, 1:4].to_numpy(dtype=float)
            except Exception:
                acc, acc_ts = None, None

        non_breathing = None
        nb_path = session_dir / 'non_breathing_ts.csv'
        if nb_path.exists():
            # 格式: "begin,<timestamp>" / "end,<timestamp>" (相对会话开始的时间秒数)
            ts_path = session_dir / 'radar_timestamps.csv'
            ts_df = pd.read_csv(ts_path, header=None)
            session_dir_t0 = pd.to_datetime(ts_df.iloc[0, 0])
            nb_rows = []
            with open(nb_path) as f:
                for line in f:
                    tag, val = line.strip().split(',', 1)
                    try:
                        nb_rows.append((tag, float(val)))
                    except ValueError:
                        # 时间戳格式 -> 转为相对秒数
                        ts = pd.to_datetime(val)
                        nb_rows.append((tag, (ts - session_dir_t0).total_seconds()))
            pairs = []
            for i, (tag, val) in enumerate(nb_rows):
                if tag == 'begin' and i + 1 < len(nb_rows):
                    pairs.append((val, nb_rows[i + 1][1]))
            non_breathing = pairs

        return {
            'ecg_timestamps': ecg_ts,
            'ecg_mv': ecg_mv,
            'acc_timestamps': acc_ts,
            'acc': acc,
            'non_breathing': non_breathing,
            'fs_ecg': self.ECG_RATE_HZ,
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def list_sessions(self) -> list:
        """列出所有存在的 (participant, posture, condition) 会话。"""
        sessions = []
        for pdir in sorted(self.dataset_path.glob('P*')):
            if not pdir.is_dir():
                continue
            try:
                pid = int(pdir.name[1:])
            except ValueError:
                continue
            for posture in self.POSTURES:
                for condition in self.CONDITIONS:
                    if (pdir / posture / condition).exists():
                        sessions.append((pid, posture, condition))
        return sessions
