"""统一导出三种数据集脚本。

功能：
1. 调用 FTUDataLoader / BGT60TR13CDataLoader / PhysDriveDataLoader
2. 批量加载三种数据集
3. 保存到指定输出目录

示例：
python src/data/loaders/export_all_datasets.py \
  --ftu

默认运行（所有参数使用 default）：
python src/data/loaders/export_all_datasets.py

可在文件头部修改全局变量（例如 FTU_PARTICIPANT_ID / FTU_SCENARIO / FTU_DISTANCE）来筛选数据。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys

import numpy as np

# 允许直接以脚本方式运行
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.bgt60_loader import BGT60TR13CDataLoader
from src.data.loaders.ftu_loader import FTUDataLoader
from src.data.loaders.physdrive_loader import PhysDriveDataLoader

# ==================== Global Config (Manual Edit) ====================
# 数据路径与输出路径
DATASET_ROOT = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "exports"

# 导出控制
COMPRESS_OUTPUT = True
BGT_INCLUDE_LONG = False
UNIFY_TO_RDA = True
BGT_LONG_FIXED_DISTANCE = "0.6m"

# FTU 筛选（None 表示不筛选）
# 例如：
# FTU_PARTICIPANT_ID = 1
# FTU_SCENARIO = "Distance"
# FTU_DISTANCE = "40 cm"
FTU_PARTICIPANT_ID: Optional[int] = 1
FTU_SCENARIO: Optional[str] = "Distance"
FTU_DISTANCE: Optional[str] = "40 cm"

# 统一输出配置
TARGET_DOPPLER = 8
TARGET_ANGLE = 16
TARGET_RANGE = 8

# 参考论文（arXiv:2507.19172）中 mmWave 对齐到 20Hz
PHYSDRIVE_FRAME_RATE_HZ = 20.0
# ====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FTU/BGT60/PhysDrive datasets")
    parser.add_argument(
        "--ftu",
        action="store_true",
        help="仅导出 FTU（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    parser.add_argument(
        "--bgt60",
        action="store_true",
        help="仅导出 BGT60（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    parser.add_argument(
        "--physdrive",
        action="store_true",
        help="仅导出 PhysDrive（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def save_npz(
    path: Path,
    arrays: Dict[str, Any],
    compress: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_export_payload(radar: np.ndarray, ref: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "radar": radar,
        "time": ref["time"],
        "heart_rate": ref["heart_rate"],
        "respiration_rate": ref["respiration_rate"],
    }


def build_common_radar_meta(
    radar_out: np.ndarray,
    radar_source: np.ndarray,
    radar_preprocess: str,
    axes: Optional[List[str]] = None,
    range_selection_method: Optional[str] = None,
    range_center_bin: Optional[int] = None,
    range_selected_bins: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    radar_axes = axes if axes is not None else (
        ["frames", "doppler", "angle", "range"]
        if UNIFY_TO_RDA
        else ["frames", "rx", "chirps", "samples"]
    )
    out: Dict[str, Any] = {
        "radar_shape": list(radar_out.shape),
        "radar_dtype": str(radar_out.dtype),
        "radar_axes": radar_axes,
        "radar_source_shape": list(radar_source.shape),
        "radar_source_dtype": str(radar_source.dtype),
        "radar_preprocess": radar_preprocess,
    }
    if range_selection_method is not None:
        out["range_selection_method"] = range_selection_method
    if range_center_bin is not None:
        out["range_center_bin"] = int(range_center_bin)
    if range_selected_bins is not None:
        out["range_selected_bins"] = [int(x) for x in range_selected_bins.tolist()]
    return out


def build_common_reference_meta(unified_ref: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "reference_len": int(len(unified_ref["time"])),
        "reference_fields": ["time", "heart_rate", "respiration_rate"],
    }


def build_manifest_success(
    sample_tag: str,
    npz_path: Path,
    radar_out: np.ndarray,
    unified_ref: Dict[str, np.ndarray],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "sample_tag": sample_tag,
        **extra,
        "npz_path": str(npz_path),
        "radar_shape": str(tuple(radar_out.shape)),
        "reference_len": len(unified_ref["time"]),
        "status": "ok",
    }
    return row


def build_manifest_failure(
    sample_tag: str,
    error: Exception,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "sample_tag": sample_tag,
        **extra,
        "npz_path": "",
        "radar_shape": "",
        "reference_len": "",
        "status": f"failed: {error}",
    }
    return row


def _sanitize_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    if arr.size == 0:
        return arr
    arr = arr - arr.mean()
    std = arr.std()
    if std > 1e-6:
        arr = arr / std
    return arr


def _estimate_rate_series_from_signal(
    signal: np.ndarray,
    fs_hz: float,
    min_hz: float,
    max_hz: float,
    min_peak_distance_sec: float,
    peak_quantile: float = 0.7,
) -> np.ndarray:
    """从单通道生理信号估计逐时刻速率序列（bpm）。"""
    x = _sanitize_vector(signal)
    n = int(x.size)
    if n == 0:
        return np.array([], dtype=np.float32)

    time = np.arange(n, dtype=np.float32) / np.float32(fs_hz)
    if n < 3 or fs_hz <= 0:
        return np.full(n, np.nan, dtype=np.float32)

    # 局部极大值 + 阈值
    threshold = np.quantile(x, peak_quantile)
    cands = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] >= threshold))[0] + 1
    min_dist = max(1, int(round(min_peak_distance_sec * fs_hz)))

    # 非极大值抑制（按时间扫描，保留更强峰）
    peaks: List[int] = []
    for idx in cands.tolist():
        if not peaks:
            peaks.append(idx)
            continue
        if idx - peaks[-1] >= min_dist:
            peaks.append(idx)
        elif x[idx] > x[peaks[-1]]:
            peaks[-1] = idx

    # 基于峰间距估计瞬时速率
    if len(peaks) >= 2:
        peak_arr = np.asarray(peaks, dtype=np.int32)
        intervals = np.diff(peak_arr).astype(np.float32) / np.float32(fs_hz)
        valid = (intervals >= (1.0 / max_hz)) & (intervals <= (1.0 / min_hz))
        if np.any(valid):
            mid_t = (peak_arr[:-1] + peak_arr[1:]).astype(np.float32) / (2.0 * np.float32(fs_hz))
            bpm = 60.0 / intervals
            mid_t = mid_t[valid]
            bpm = bpm[valid]
            if mid_t.size >= 2:
                rate = np.interp(time, mid_t, bpm, left=bpm[0], right=bpm[-1]).astype(np.float32)
                return rate
            if mid_t.size == 1:
                return np.full(n, bpm[0], dtype=np.float32)

    # 兜底：频域主频估计为常数序列
    win = np.hanning(n).astype(np.float32)
    y = x * win
    fft_n = max(256, int(2 ** np.ceil(np.log2(max(8, n)))))
    spec = np.abs(np.fft.rfft(y, n=fft_n)) ** 2
    freqs = np.fft.rfftfreq(fft_n, d=(1.0 / fs_hz))
    band = (freqs >= min_hz) & (freqs <= max_hz)
    if np.any(band):
        band_freqs = freqs[band]
        band_spec = spec[band]
        peak_hz = float(band_freqs[int(np.argmax(band_spec))])
        return np.full(n, 60.0 * peak_hz, dtype=np.float32)
    return np.full(n, np.nan, dtype=np.float32)


def build_physdrive_reference(ref: Dict[str, Any], frame_rate_hz: float) -> Dict[str, np.ndarray]:
    ecg = np.asarray(ref["ecg"], dtype=np.float32).reshape(-1)
    resp = np.asarray(ref["respiration"], dtype=np.float32).reshape(-1)
    n = min(ecg.size, resp.size)
    ecg = ecg[:n]
    resp = resp[:n]
    time = np.arange(n, dtype=np.float32) / np.float32(frame_rate_hz)

    heart_rate = _estimate_rate_series_from_signal(
        ecg,
        fs_hz=frame_rate_hz,
        min_hz=0.6,
        max_hz=3.0,
        min_peak_distance_sec=0.30,
        peak_quantile=0.70,
    )
    respiration_rate = _estimate_rate_series_from_signal(
        resp,
        fs_hz=frame_rate_hz,
        min_hz=0.08,
        max_hz=0.70,
        min_peak_distance_sec=1.20,
        peak_quantile=0.65,
    )
    return {
        "time": time,
        "heart_rate": heart_rate.astype(np.float32),
        "respiration_rate": respiration_rate.astype(np.float32),
    }


def to_unified_reference(
    dataset_name: str,
    ref: Dict[str, Any],
    frame_rate_hz: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if dataset_name == "FTU":
        time = np.asarray(ref["timestamps"], dtype=np.float32).reshape(-1)
        heart_rate = np.asarray(ref["heart_rate"], dtype=np.float32).reshape(-1)
        n = min(time.size, heart_rate.size)
        return {
            "time": time[:n],
            "heart_rate": heart_rate[:n],
            "respiration_rate": np.full(n, np.nan, dtype=np.float32),
        }

    if dataset_name == "BGT60TR13C":
        time = np.asarray(ref["time"], dtype=np.float32).reshape(-1)
        heart_rate = np.asarray(ref["heart_rate"], dtype=np.float32).reshape(-1)
        respiration_rate = np.asarray(ref["respiration_rate"], dtype=np.float32).reshape(-1)
        n = min(time.size, heart_rate.size, respiration_rate.size)
        return {
            "time": time[:n],
            "heart_rate": heart_rate[:n],
            "respiration_rate": respiration_rate[:n],
        }

    if dataset_name == "PhysDrive":
        if frame_rate_hz is None:
            raise ValueError("PhysDrive 需要提供 frame_rate_hz 用于生成 time/HR/RR。")
        return build_physdrive_reference(ref, frame_rate_hz=frame_rate_hz)

    raise ValueError(f"未知数据集: {dataset_name}")


def _select_even_indices(length: int, target: int) -> np.ndarray:
    if target >= length:
        return np.arange(length, dtype=np.int32)
    return np.linspace(0, length - 1, num=target, dtype=np.int32)


def _crop_or_pad_last_axis(arr: np.ndarray, target: int) -> np.ndarray:
    cur = arr.shape[-1]
    if cur == target:
        return arr
    if cur > target:
        start = (cur - target) // 2
        end = start + target
        return arr[..., start:end]
    pad_shape = list(arr.shape)
    pad_shape[-1] = target - cur
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def _range_window_indices(total_bins: int, target_bins: int, center_bin: int) -> np.ndarray:
    if target_bins >= total_bins:
        return np.arange(total_bins, dtype=np.int32)
    half = target_bins // 2
    start = center_bin - half
    end = start + target_bins
    if start < 0:
        start = 0
        end = target_bins
    if end > total_bins:
        end = total_bins
        start = total_bins - target_bins
    return np.arange(start, end, dtype=np.int32)


def _build_rda_cube_from_frame(
    frame_data: np.ndarray,
    chirp_idx: np.ndarray,
    sample_idx: np.ndarray,
    target_doppler: int,
    target_angle: int,
) -> np.ndarray:
    """将单帧原始 (rx,chirps,samples) 转为 (doppler,angle,range_full)。"""
    frame = np.asarray(frame_data, dtype=np.complex64)[:, chirp_idx][:, :, sample_idx]  # (rx,c,s)
    frame = frame - frame.mean(axis=1, keepdims=True)  # slow-time clutter suppression
    range_fft = np.fft.fft(frame, axis=-1)
    range_fft = range_fft[..., : max(1, range_fft.shape[-1] // 2)]  # positive range half
    doppler_fft = np.fft.fftshift(
        np.fft.fft(range_fft, n=target_doppler, axis=1),
        axes=1,
    )  # (rx,d,r)
    angle_fft = np.fft.fftshift(
        np.fft.fft(doppler_fft, n=target_angle, axis=0),
        axes=0,
    )  # (a,d,r)
    return np.transpose(angle_fft, (1, 0, 2))  # (d,a,r)


def convert_adc_cube_to_rda(
    radar: np.ndarray,
    target_doppler: int = TARGET_DOPPLER,
    target_angle: int = TARGET_ANGLE,
    target_range: int = TARGET_RANGE,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """将 (frames, rx, chirps, samples) 转为 (frames, doppler, angle, range)。

    与旧版不同：range 维采用“全样本统一目标窗口”，而非逐帧 top-k。
    返回:
        - rda: (frames, target_doppler, target_angle, target_range)
        - selected_range_bins: 连续窗口对应的原始 range bin 索引
        - center_range_bin: 目标中心 range bin（由全样本能量估计）
    """
    if radar.ndim != 4:
        raise ValueError(f"期望 4D 雷达张量，实际 ndim={radar.ndim}")

    frames, _, chirps, samples = radar.shape
    out = np.empty((frames, target_doppler, target_angle, target_range), dtype=np.complex64)

    chirp_idx = _select_even_indices(chirps, min(chirps, 64))
    sample_idx = _select_even_indices(samples, min(samples, 256))

    # pass-1: 全样本估计目标中心 range bin（语义稳定）
    global_range_energy: Optional[np.ndarray] = None
    for fi in range(frames):
        cube_full = _build_rda_cube_from_frame(
            frame_data=radar[fi],
            chirp_idx=chirp_idx,
            sample_idx=sample_idx,
            target_doppler=target_doppler,
            target_angle=target_angle,
        )  # (d,a,rfull)
        energy = np.mean(np.abs(cube_full), axis=(0, 1))
        if global_range_energy is None:
            global_range_energy = energy.astype(np.float64)
        else:
            global_range_energy += energy

    if global_range_energy is None or global_range_energy.size == 0:
        raise ValueError("无法估计 range 能量分布。")

    center_range_bin = int(np.argmax(global_range_energy))
    selected_range_bins = _range_window_indices(
        total_bins=int(global_range_energy.size),
        target_bins=target_range,
        center_bin=center_range_bin,
    )

    # pass-2: 固定窗口提取（连续 range bins）
    for fi in range(frames):
        cube_full = _build_rda_cube_from_frame(
            frame_data=radar[fi],
            chirp_idx=chirp_idx,
            sample_idx=sample_idx,
            target_doppler=target_doppler,
            target_angle=target_angle,
        )  # (d,a,rfull)
        cube = cube_full[:, :, selected_range_bins]
        out[fi] = _crop_or_pad_last_axis(cube, target_range).astype(np.complex64)

    return out, selected_range_bins.astype(np.int32), center_range_bin


def align_rda_shape(
    radar_rda: np.ndarray,
    target_doppler: int = TARGET_DOPPLER,
    target_angle: int = TARGET_ANGLE,
    target_range: int = TARGET_RANGE,
) -> np.ndarray:
    """将现有 RDA 数据对齐到统一的 (frames, d, a, r) 尺寸。"""
    if radar_rda.ndim != 4:
        raise ValueError(f"期望 PhysDrive RDA 为 4D，实际 ndim={radar_rda.ndim}")
    x = np.asarray(radar_rda, dtype=np.complex64)
    # 通过 FFT/截断对齐 doppler 与 angle
    if x.shape[1] != target_doppler:
        x = np.fft.fftshift(np.fft.fft(x, n=target_doppler, axis=1), axes=1)
    if x.shape[2] != target_angle:
        x = np.fft.fftshift(np.fft.fft(x, n=target_angle, axis=2), axes=2)
    if x.shape[3] != target_range:
        x = _crop_or_pad_last_axis(x, target_range)
    return x.astype(np.complex64)


def export_ftu(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
    participant_id: Optional[int] = None,
    scenario: Optional[str] = None,
    distance: Optional[str] = None,
) -> Tuple[int, int]:
    loader = FTUDataLoader(str(dataset_root / "4TU.ResearchD"))
    dataset_out = output_dir / "FTU"
    manifest_rows: List[Dict[str, Any]] = []

    samples = loader.list_available_samples(
        participant_id=participant_id,
        scenario=scenario,
        limit=None
    )
    if distance is not None:
        samples = [s for s in samples if s["distance"] == distance]

    if not samples:
        print("[FTU] 没有匹配样本，跳过导出。")
        write_manifest(dataset_out / "manifest.csv", [])
        return 0, 0
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        scenario = sample["scenario"]
        distance = sample["distance"]
        repeat = sample["repeat"]
        sample_tag = (
            f"p{pid:02d}_{safe_name(scenario)}_{safe_name(distance)}_r{repeat}"
        )

        try:
            radar = loader.load_radar_data(
                participant_id=pid,
                scenario=scenario,
                distance=distance,
                repeat=repeat,
            )
            ref = loader.load_reference_data(
                participant_id=pid,
                scenario=scenario,
                distance=distance,
                repeat=repeat,
            )
            unified_ref = to_unified_reference("FTU", ref)
            selected_range_bins: Optional[np.ndarray] = None
            center_range_bin: Optional[int] = None
            if UNIFY_TO_RDA:
                radar_out, selected_range_bins, center_range_bin = convert_adc_cube_to_rda(radar)
            else:
                radar_out = radar

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                build_export_payload(radar_out, unified_ref),
                compress=compress,
            )

            meta = {
                "dataset": "FTU",
                "participant_id": pid,
                "scenario": scenario,
                "distance": distance,
                "repeat": repeat,
                **build_common_radar_meta(
                    radar_out=radar_out,
                    radar_source=radar,
                    radar_preprocess="raw_iq_to_rda_fft" if UNIFY_TO_RDA else "none",
                    range_selection_method=(
                        "global_energy_center_contiguous_window" if UNIFY_TO_RDA else "none"
                    ),
                    range_center_bin=center_range_bin,
                    range_selected_bins=selected_range_bins,
                ),
                **build_common_reference_meta(unified_ref),
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                build_manifest_success(
                    sample_tag=sample_tag,
                    npz_path=npz_path,
                    radar_out=radar_out,
                    unified_ref=unified_ref,
                    extra={
                        "participant_id": pid,
                        "scenario": scenario,
                        "distance": distance,
                        "repeat": repeat,
                    },
                )
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                build_manifest_failure(
                    sample_tag=sample_tag,
                    error=e,
                    extra={
                        "participant_id": pid,
                        "scenario": scenario,
                        "distance": distance,
                        "repeat": repeat,
                    },
                )
            )
            failure += 1

        print(f"[FTU] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def iter_bgt_samples(loader: BGT60TR13CDataLoader, include_long: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pid in loader.list_participants():
        info = loader.list_available_measurements(pid)
        for distance in info["short"]:
            rows.append(
                {
                    "participant_id": pid,
                    "distance": distance,
                    "measurement_type": "short",
                }
            )
    if include_long:
        rows.append(
            {
                "participant_id": 1,
                "distance": BGT_LONG_FIXED_DISTANCE,
                "measurement_type": "long",
            }
        )
    return rows


def export_bgt60(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
    include_long: bool,
) -> Tuple[int, int]:
    loader = BGT60TR13CDataLoader(str(dataset_root / "BGT60TR13C"))
    dataset_out = output_dir / "BGT60TR13C"
    manifest_rows: List[Dict[str, Any]] = []

    samples = iter_bgt_samples(loader, include_long=include_long)
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        distance = sample["distance"]
        measurement_type = sample["measurement_type"]
        sample_tag = f"p{pid:02d}_{safe_name(distance)}_{measurement_type}"

        try:
            radar = loader.load_radar_data(
                participant_id=pid,
                distance=distance if measurement_type == "short" else BGT_LONG_FIXED_DISTANCE,
                measurement_type=measurement_type,
                apply_dc_correction=False,
            )
            ref = loader.load_reference_data(
                participant_id=pid,
                distance=distance if measurement_type == "short" else BGT_LONG_FIXED_DISTANCE,
                measurement_type=measurement_type,
            )
            unified_ref = to_unified_reference("BGT60TR13C", ref)
            selected_range_bins: Optional[np.ndarray] = None
            center_range_bin: Optional[int] = None
            if UNIFY_TO_RDA:
                radar_out, selected_range_bins, center_range_bin = convert_adc_cube_to_rda(radar)
            else:
                radar_out = radar

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                build_export_payload(radar_out, unified_ref),
                compress=compress,
            )

            meta = {
                "dataset": "BGT60TR13C",
                "participant_id": pid,
                "distance": distance,
                "measurement_type": measurement_type,
                **build_common_radar_meta(
                    radar_out=radar_out,
                    radar_source=radar,
                    radar_preprocess="raw_adc_to_rda_fft" if UNIFY_TO_RDA else "none",
                    range_selection_method=(
                        "global_energy_center_contiguous_window" if UNIFY_TO_RDA else "none"
                    ),
                    range_center_bin=center_range_bin,
                    range_selected_bins=selected_range_bins,
                ),
                **build_common_reference_meta(unified_ref),
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                build_manifest_success(
                    sample_tag=sample_tag,
                    npz_path=npz_path,
                    radar_out=radar_out,
                    unified_ref=unified_ref,
                    extra={
                        "participant_id": pid,
                        "distance": distance,
                        "measurement_type": measurement_type,
                    },
                )
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                build_manifest_failure(
                    sample_tag=sample_tag,
                    error=e,
                    extra={
                        "participant_id": pid,
                        "distance": distance,
                        "measurement_type": measurement_type,
                    },
                )
            )
            failure += 1

        print(f"[BGT60] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def iter_phys_samples(loader: PhysDriveDataLoader) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session_id in loader.list_sessions():
        for sample_id in loader.list_samples(session_id):
            rows.append({"session_id": session_id, "sample_id": sample_id})
    return rows


def export_physdrive(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
) -> Tuple[int, int]:
    loader = PhysDriveDataLoader(str(dataset_root / "PhysDrive"))
    dataset_out = output_dir / "PhysDrive"
    manifest_rows: List[Dict[str, Any]] = []

    samples = iter_phys_samples(loader)
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        session_id = sample["session_id"]
        sample_id = sample["sample_id"]
        sample_tag = f"{session_id}_{sample_id:03d}"

        try:
            radar = loader.load_radar_data(
                session_id=session_id,
                sample_id=sample_id,
                return_complex=True,
            )
            ref = loader.load_reference_data(
                session_id=session_id,
                sample_id=sample_id,
            )
            unified_ref = to_unified_reference(
                "PhysDrive",
                ref,
                frame_rate_hz=PHYSDRIVE_FRAME_RATE_HZ,
            )
            radar_out = align_rda_shape(radar) if UNIFY_TO_RDA else radar

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                build_export_payload(radar_out, unified_ref),
                compress=compress,
            )

            meta = {
                "dataset": "PhysDrive",
                "session_id": session_id,
                "sample_id": sample_id,
                **build_common_radar_meta(
                    radar_out=radar_out,
                    radar_source=radar,
                    radar_preprocess="already_rda_aligned",
                    axes=["frames", "doppler", "angle", "range"],
                ),
                **build_common_reference_meta(unified_ref),
                "reference_source": "ecg_resp_converted",
                "reference_frame_rate_hz": PHYSDRIVE_FRAME_RATE_HZ,
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                build_manifest_success(
                    sample_tag=sample_tag,
                    npz_path=npz_path,
                    radar_out=radar_out,
                    unified_ref=unified_ref,
                    extra={
                        "session_id": session_id,
                        "sample_id": sample_id,
                    },
                )
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                build_manifest_failure(
                    sample_tag=sample_tag,
                    error=e,
                    extra={
                        "session_id": session_id,
                        "sample_id": sample_id,
                    },
                )
            )
            failure += 1

        print(f"[PhysDrive] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def main() -> None:
    args = parse_args()
    dataset_root = Path(DATASET_ROOT).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    selected_datasets = []
    if args.ftu:
        selected_datasets.append("FTU")
    if args.bgt60:
        selected_datasets.append("BGT60TR13C")
    if args.physdrive:
        selected_datasets.append("PhysDrive")
    if not selected_datasets:
        selected_datasets = ["FTU", "BGT60TR13C", "PhysDrive"]

    if "FTU" in selected_datasets:
        ok, fail = export_ftu(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
            participant_id=FTU_PARTICIPANT_ID,
            scenario=FTU_SCENARIO,
            distance=FTU_DISTANCE,
        )
        summary["FTU"] = {"success": ok, "failed": fail}

    if "BGT60TR13C" in selected_datasets:
        ok, fail = export_bgt60(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
            include_long=BGT_INCLUDE_LONG,
        )
        summary["BGT60TR13C"] = {"success": ok, "failed": fail}

    if "PhysDrive" in selected_datasets:
        ok, fail = export_physdrive(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
        )
        summary["PhysDrive"] = {"success": ok, "failed": fail}

    save_json(output_dir / "export_summary.json", summary)
    print("\n=== Export Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
