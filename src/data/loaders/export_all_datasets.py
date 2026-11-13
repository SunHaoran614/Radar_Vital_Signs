"""统一导出三种数据集脚本。

功能：
1. 调用 FTUDataLoader / BGT60TR13CDataLoader / PhysDriveDataLoader
2. 批量加载三种数据集
3. 保存到指定输出目录

示例（从仓库根目录运行）：
python -m src.data.loaders.export_all_datasets \
  --ftu

默认运行（所有参数使用 default）：
python -m src.data.loaders.export_all_datasets

可在文件头部修改全局变量（例如 FTU_PARTICIPANT_ID / FTU_SCENARIO / FTU_DISTANCE）来筛选数据。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .ftu_loader import FTUDataLoader
from src.radar.ftu_adc_config import FTUADCConfig
from src.radar.bgt_adc_config import BGTADCConfig
from src.data.ftu_uniqueness import alias_map

# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

try:
    import neurokit2 as nk  # type: ignore[import-not-found]
except ImportError:
    nk = None  # type: ignore[assignment]

# 允许直接以脚本方式运行
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================== Global Config (Manual Edit) ====================
# 数据路径与输出路径
DATASET_ROOT = Path(os.environ.get("RADAR_DATA_ROOT", "data"))
OUTPUT_DIR = Path("exports")

# 导出控制
COMPRESS_OUTPUT = True
BGT_INCLUDE_LONG = True
UNIFY_TO_RDA = True
BGT_LONG_FIXED_DISTANCE = "0.6m"

# FTU 筛选条件：默认 None 表示不过滤，导出全部可用样本
# 如需只导出部分样本，可手动设置：
# FTU_PARTICIPANT_ID = 1
# FTU_SCENARIO = "Distance"
# FTU_DISTANCE = "40 cm"
FTU_PARTICIPANT_ID: Optional[int] = None
FTU_SCENARIO: Optional[str] = None
FTU_DISTANCE: Optional[str] = None

# 统一输出配置
TARGET_DOPPLER = 8
TARGET_ANGLE = 16
TARGET_RANGE = 8

# 参考论文（arXiv:2507.19172）中 mmWave 对齐到 20Hz
PHYSDRIVE_FRAME_RATE_HZ = 20.0
PHYSDRIVE_ECG_PRIMARY_METHOD = "hamilton2002"
PHYSDRIVE_ECG_FALLBACK_METHODS = ("pantompkins1985", "elgendi2010")
PHYSDRIVE_RSP_PEAK_METHOD = "khodadad2018"
PHYSDRIVE_ECG_RELIABLE_BPM = (45.0, 130.0)
PHYSDRIVE_ECG_MIN_COVERAGE = 0.8
PHYSDRIVE_ECG_MAX_INTERPOLATION_GAP_SECONDS = 2.0
# ====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FTU/BGT60/PhysDrive datasets")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="数据集根目录路径(默认使用代码中的DATASET_ROOT配置)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录路径(默认使用代码中的OUTPUT_DIR配置)",
    )
    parser.add_argument(
        "--ftu",
        action="store_true",
        help="仅导出 FTU(若与其它数据集参数同时出现,则按出现的数据集导出)",
    )
    parser.add_argument(
        "--ftu-frontend-config",
        type=Path,
        default=None,
        help="FTU ADC→RDA 前端 JSON 配置；默认保持旧版前端。",
    )
    parser.add_argument(
        "--ftu-alias-audit",
        type=Path,
        default=None,
        help="用户提供的FTU全字节去别名清单；配置FTU ADC前端时必须显式提供。",
    )
    parser.add_argument(
        "--bgt60",
        action="store_true",
        help="仅导出 BGT60(若与其它数据集参数同时出现,则按出现的数据集导出)",
    )
    parser.add_argument(
        "--bgt-frontend-config",
        type=Path,
        default=None,
        help="BGT ADC→RDA 候选前端 JSON 配置；不指定则保持旧版数值转换。",
    )
    parser.add_argument(
        "--bgt-skip-long",
        action="store_true",
        help="BGT 版本化导出时暂不处理两小时长录制。",
    )
    parser.add_argument(
        "--physdrive",
        action="store_true",
        help="仅导出 PhysDrive(若与其它数据集参数同时出现,则按出现的数据集导出)",
    )
    args = parser.parse_args()
    includes_ftu = args.ftu or not (args.physdrive or args.bgt60)
    if includes_ftu and args.ftu_frontend_config is not None and args.ftu_alias_audit is None:
        parser.error("--ftu-frontend-config requires --ftu-alias-audit")
    return args


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

    # A failed sample and a successful sample have different diagnostic fields.
    # Keep first-seen column order while accepting either row order.
    headers = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def print_stage(dataset_name: str, message: str) -> None:
    print(f"\n[{dataset_name}] {message}", flush=True)


def format_progress(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]   0.0%"
    ratio = min(max(current / total, 0.0), 1.0)
    done = int(round(ratio * width))
    bar = "#" * done + "-" * (width - done)
    return f"[{bar}] {ratio * 100:5.1f}%"


def print_sample_progress(
    dataset_name: str,
    current: int,
    total: int,
    sample_tag: str,
    success: int,
    failure: int,
    started_at: float,
    status: str,
) -> None:
    elapsed = time.monotonic() - started_at
    print(
        (
            f"[{dataset_name}] {format_progress(current, total)} "
            f"{current}/{total} ok={success} failed={failure} "
            f"elapsed={elapsed:.1f}s status={status} sample={sample_tag}"
        ),
        flush=True,
    )


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


def _require_neurokit2():
    if nk is None:
        raise ImportError(
            "PhysDrive HR/RR extraction requires neurokit2. "
            "Install it in the active environment, for example: pip install neurokit2"
        )
    return nk


def _clean_rate(rate: np.ndarray, min_bpm: float, max_bpm: float) -> np.ndarray:
    out = np.asarray(rate, dtype=np.float32).copy()
    invalid = (~np.isfinite(out)) | (out < min_bpm) | (out > max_bpm)
    out[invalid] = np.nan
    return out


def _finite_coverage(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def _rate_from_peak_intervals(
    peaks: np.ndarray,
    sampling_rate: float,
    desired_length: int,
    min_bpm: float,
    max_bpm: float,
    max_interpolation_gap_seconds: float,
) -> np.ndarray:
    rate = np.full(desired_length, np.nan, dtype=np.float32)
    peaks = np.asarray(peaks, dtype=np.int32).reshape(-1)
    if peaks.size < 2:
        return rate

    for left, right in zip(peaks[:-1], peaks[1:]):
        if right <= left:
            continue
        bpm = 60.0 * sampling_rate / float(right - left)
        if min_bpm <= bpm <= max_bpm:
            start = max(int(left), 0)
            end = min(int(right) + 1, desired_length)
            if start < end:
                rate[start:end] = np.float32(bpm)

    finite_idx = np.flatnonzero(np.isfinite(rate))
    if finite_idx.size < 2:
        return rate

    max_gap = int(round(max_interpolation_gap_seconds * sampling_rate))
    x = np.arange(desired_length, dtype=np.float32)
    interpolated = np.interp(x, finite_idx, rate[finite_idx]).astype(np.float32)
    filled = rate.copy()
    for idx in np.flatnonzero(~np.isfinite(rate)):
        left_candidates = finite_idx[finite_idx < idx]
        right_candidates = finite_idx[finite_idx > idx]
        if left_candidates.size == 0 or right_candidates.size == 0:
            continue
        if int(right_candidates[0] - left_candidates[-1]) <= max_gap:
            filled[idx] = interpolated[idx]
    return filled


def _detect_ecg_rate_candidate(
    nk: Any,
    cleaned_ecg: np.ndarray,
    sampling_rate: float,
    desired_length: int,
    method: str,
) -> np.ndarray:
    try:
        _, info = nk.ecg_peaks(
            cleaned_ecg,
            sampling_rate=sampling_rate,
            method=method,
        )
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int32)
    except Exception:  # noqa: BLE001
        return np.full(desired_length, np.nan, dtype=np.float32)
    return _rate_from_peak_intervals(
        peaks=peaks,
        sampling_rate=sampling_rate,
        desired_length=desired_length,
        min_bpm=PHYSDRIVE_ECG_RELIABLE_BPM[0],
        max_bpm=PHYSDRIVE_ECG_RELIABLE_BPM[1],
        max_interpolation_gap_seconds=PHYSDRIVE_ECG_MAX_INTERPOLATION_GAP_SECONDS,
    )


def _neurokit_ecg_rate(
    ecg: np.ndarray,
    sampling_rate: float,
    desired_length: int,
) -> np.ndarray:
    nk = _require_neurokit2()
    try:
        cleaned = nk.ecg_clean(ecg, sampling_rate=sampling_rate, method="neurokit")
    except Exception:  # noqa: BLE001
        return np.full(desired_length, np.nan, dtype=np.float32)

    methods = (PHYSDRIVE_ECG_PRIMARY_METHOD, *PHYSDRIVE_ECG_FALLBACK_METHODS)
    candidates = [
        (
            method,
            _detect_ecg_rate_candidate(
                nk=nk,
                cleaned_ecg=cleaned,
                sampling_rate=sampling_rate,
                desired_length=desired_length,
                method=method,
            ),
        )
        for method in methods
    ]

    primary_rate = candidates[0][1]
    primary_has_values = np.isfinite(primary_rate).any()
    primary_median = np.nanmedian(primary_rate) if primary_has_values else np.nan
    if (
        _finite_coverage(primary_rate) >= PHYSDRIVE_ECG_MIN_COVERAGE
        and np.isfinite(primary_median)
    ):
        return primary_rate

    return max(
        (rate for _, rate in candidates),
        key=lambda rate: _finite_coverage(rate),
    )


def _neurokit_rsp_rate(
    resp: np.ndarray,
    sampling_rate: float,
    desired_length: int,
) -> np.ndarray:
    nk = _require_neurokit2()
    try:
        cleaned = nk.rsp_clean(resp, sampling_rate=sampling_rate)
        _, info = nk.rsp_peaks(
            cleaned,
            sampling_rate=sampling_rate,
            method=PHYSDRIVE_RSP_PEAK_METHOD,
        )
        peaks = np.asarray(info.get("RSP_Peaks", []), dtype=np.int32)
    except Exception:  # noqa: BLE001
        return np.full(desired_length, np.nan, dtype=np.float32)
    if peaks.size < 2:
        return np.full(desired_length, np.nan, dtype=np.float32)
    rate = nk.signal_rate(
        peaks,
        sampling_rate=sampling_rate,
        desired_length=desired_length,
    )
    return np.asarray(rate, dtype=np.float32).reshape(-1)


def build_physdrive_reference(ref: Dict[str, Any], frame_rate_hz: float) -> Dict[str, np.ndarray]:
    ecg = np.asarray(ref["ecg"], dtype=np.float32).reshape(-1)
    resp = np.asarray(ref["respiration"], dtype=np.float32).reshape(-1)
    if ecg.size != resp.size or ecg.size == 0:
        raise ValueError(
            f"PhysDrive ECG/RESP 长度不一致或为空: ecg={ecg.size}, resp={resp.size}"
        )
    n = ecg.size
    time = np.arange(n, dtype=np.float32) / np.float32(frame_rate_hz)

    heart_rate = _clean_rate(
        _neurokit_ecg_rate(ecg, sampling_rate=frame_rate_hz, desired_length=n),
        min_bpm=PHYSDRIVE_ECG_RELIABLE_BPM[0],
        max_bpm=PHYSDRIVE_ECG_RELIABLE_BPM[1],
    )
    respiration_rate = _clean_rate(
        _neurokit_rsp_rate(resp, sampling_rate=frame_rate_hz, desired_length=n),
        min_bpm=3.0,
        max_bpm=60.0,
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
    doppler_method: str = "legacy_truncate",
    clutter_method: str = "chirp_mean",
) -> np.ndarray:
    """将单帧原始 (rx,chirps,samples) 转为 (doppler,angle,range_full)。"""
    frame = np.asarray(frame_data, dtype=np.complex64)[:, chirp_idx][:, :, sample_idx]  # (rx,c,s)
    if clutter_method == "chirp_mean":
        frame = frame - frame.mean(axis=1, keepdims=True)
    elif clutter_method != "none":
        raise ValueError(f"Unknown clutter_method: {clutter_method}")
    range_fft = np.fft.fft(frame, axis=-1)
    range_fft = range_fft[..., : max(1, range_fft.shape[-1] // 2)]  # positive range half
    if doppler_method == "legacy_truncate":
        # Historical behavior: n < chirps silently truncates to the first n chirps.
        doppler_fft = np.fft.fftshift(
            np.fft.fft(range_fft, n=target_doppler, axis=1), axes=1
        )
    elif doppler_method == "full_center_bins":
        full = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)
        center = full.shape[1] // 2
        start = center - target_doppler // 2
        doppler_fft = full[:, start:start + target_doppler, :]
        if doppler_fft.shape[1] != target_doppler:
            raise ValueError("target_doppler exceeds selected chirp count")
    else:
        raise ValueError(f"Unknown doppler_method: {doppler_method}")
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
    doppler_method: str = "legacy_truncate",
    clutter_method: str = "chirp_mean",
    chirp_selection_method: str = "legacy_64",
    range_selection_method: str = "global_energy",
    range_search_bins: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """将 (frames, rx, chirps, samples) 转为 (frames, doppler, angle, range)。

    range 维采用全样本统一目标窗口，而非逐帧 top-k。默认参数保持
    旧版导出行为；显式的 all/full_center_bins/none/motion_zero_doppler
    组合用于尚未晋级正式导出的物理候选实验。
    返回:
        - rda: (frames, target_doppler, target_angle, target_range)
        - selected_range_bins: 连续窗口对应的原始 range bin 索引
        - center_range_bin: 目标中心 range bin（由全样本能量估计）
    """
    if radar.ndim != 4:
        raise ValueError(f"期望 4D 雷达张量，实际 ndim={radar.ndim}")

    frames, _, chirps, samples = radar.shape
    out = np.empty((frames, target_doppler, target_angle, target_range), dtype=np.complex64)

    if chirp_selection_method == "legacy_64":
        chirp_idx = _select_even_indices(chirps, min(chirps, 64))
    elif chirp_selection_method == "all":
        chirp_idx = np.arange(chirps, dtype=np.int32)
    else:
        raise ValueError(f"Unknown chirp_selection_method: {chirp_selection_method}")
    if range_selection_method not in {"global_energy", "motion_zero_doppler"}:
        raise ValueError(f"Unknown range_selection_method: {range_selection_method}")
    if range_selection_method == "motion_zero_doppler" and (
        doppler_method != "full_center_bins" or clutter_method != "none"
    ):
        raise ValueError("motion_zero_doppler requires full_center_bins and no chirp-mean clutter removal")
    sample_idx = _select_even_indices(samples, min(samples, 256))

    # pass-1: 全样本估计目标中心 range bin（语义稳定）
    global_range_energy: Optional[np.ndarray] = None
    zero_sum: Optional[np.ndarray] = None
    zero_power: Optional[np.ndarray] = None
    for fi in range(frames):
        cube_full = _build_rda_cube_from_frame(
            frame_data=radar[fi],
            chirp_idx=chirp_idx,
            sample_idx=sample_idx,
            target_doppler=target_doppler,
            target_angle=target_angle,
            doppler_method=doppler_method,
            clutter_method=clutter_method,
        )  # (d,a,rfull)
        if range_selection_method == "global_energy":
            energy = np.mean(np.abs(cube_full), axis=(0, 1))
            if global_range_energy is None:
                global_range_energy = energy.astype(np.float64)
            else:
                global_range_energy += energy
        else:
            # Preserve complex slow-time motion in the zero-Doppler slice.
            zero = np.asarray(cube_full[target_doppler // 2], dtype=np.complex128)
            if zero_sum is None:
                zero_sum = np.zeros_like(zero, dtype=np.complex128)
                zero_power = np.zeros_like(zero.real, dtype=np.float64)
            zero_sum += zero
            zero_power += np.abs(zero) ** 2

    if range_selection_method == "motion_zero_doppler" and zero_sum is not None:
        variance = zero_power / frames - np.abs(zero_sum / frames) ** 2
        global_range_energy = np.mean(np.maximum(variance, 0), axis=0)
    if global_range_energy is None or global_range_energy.size == 0:
        raise ValueError("无法估计 range 能量分布。")

    if range_search_bins is None and range_selection_method == "motion_zero_doppler":
        range_search_bins = (8, min(46, len(global_range_energy)))
    if range_search_bins is not None:
        lo, hi = map(int, range_search_bins)
        if not (0 <= lo < hi <= len(global_range_energy)):
            raise ValueError("Invalid range_search_bins")
        center_range_bin = lo + int(np.argmax(global_range_energy[lo:hi]))
    else:
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
            doppler_method=doppler_method,
            clutter_method=clutter_method,
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
    """Align an *already spectral* RDA cube without another Doppler/angle FFT."""
    if radar_rda.ndim != 4:
        raise ValueError(f"期望 PhysDrive RDA 为 4D，实际 ndim={radar_rda.ndim}")
    x = np.asarray(radar_rda, dtype=np.complex64)
    # Doppler and angle are frequency-bin axes here. Transforming them again
    # changes their meaning, and np.fft.fft(..., n=smaller) also drops bins.
    for axis, target in ((1, target_doppler), (2, target_angle)):
        source = x.shape[axis]
        if source == target:
            continue
        if source > target:
            lo = (source - target) // 2
            selection = [slice(None)] * x.ndim
            selection[axis] = slice(lo, lo + target)
            x = x[tuple(selection)]
        else:
            before = (target - source) // 2
            padding = [(0, 0)] * x.ndim
            padding[axis] = (before, target - source - before)
            x = np.pad(x, padding)
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
    frontend_config: Optional[FTUADCConfig] = None,
    alias_audit: Optional[Path] = None,
    participant_ids: Optional[set[int]] = None,
    resume: bool = False,
) -> Tuple[int, int]:
    started_at = time.monotonic()
    print_stage("FTU", "Stage 1/4: initialize loader")
    ftu_roots = (dataset_root, dataset_root / "4TU", dataset_root / "4TU.ResearchD")
    ftu_root = next((path for path in ftu_roots
                     if all((path / part).is_dir() for part in
                            ("Radar data", "HR_Ref_Values", "BR_Ref_Values"))), None)
    if ftu_root is None:
        raise FileNotFoundError(f"FTU dataset not found under {dataset_root}")
    loader = FTUDataLoader(str(ftu_root))
    frontend_config = frontend_config or FTUADCConfig()
    dataset_out = output_dir / "FTU"
    manifest_rows: List[Dict[str, Any]] = []

    print_stage("FTU", "Stage 2/4: discover samples")
    samples = loader.list_available_samples(
        participant_id=participant_id,
        scenario=scenario,
        limit=None
    )
    if participant_ids is not None:
        if participant_id is not None:
            raise ValueError("participant_id and participant_ids cannot be combined")
        samples = [sample for sample in samples if sample["participant_id"] in participant_ids]
    if distance is not None:
        samples = [s for s in samples if s["distance"] == distance]
    alias_audit_sha256 = None
    if alias_audit is not None:
        aliases = alias_map(alias_audit)
        alias_audit_sha256 = hashlib.sha256(alias_audit.read_bytes()).hexdigest()
        samples = [sample for sample in samples if (
            tag := f"p{sample['participant_id']:02d}_{safe_name(sample['scenario'])}_"
                   f"{safe_name(sample['distance'])}_r{sample['repeat']}")
            == aliases.get(tag, tag)]
        if participant_ids is None and participant_id is None and scenario is None and distance is None and len(samples) != 376:
            raise ValueError(f"Expected 376 unique FTU recordings, found {len(samples)}")

    if not samples:
        print("[FTU] 没有匹配样本，跳过导出。")
        write_manifest(dataset_out / "manifest.csv", [])
        return 0, 0
    print(f"[FTU] discovered {len(samples)} sample(s)", flush=True)
    success = 0
    failure = 0

    print_stage("FTU", "Stage 3/4: export samples")
    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        scenario = sample["scenario"]
        distance = sample["distance"]
        repeat = sample["repeat"]
        sample_tag = (
            f"p{pid:02d}_{safe_name(scenario)}_{safe_name(distance)}_r{repeat}"
        )

        npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
        meta_path = dataset_out / "meta" / f"{sample_tag}.json"
        if resume and npz_path.is_file() and meta_path.is_file():
            try:
                with meta_path.open(encoding="utf-8") as handle:
                    saved_meta = json.load(handle)
                if saved_meta.get("ftu_adc_frontend_fingerprint") != frontend_config.fingerprint:
                    raise ValueError("saved ADC frontend fingerprint mismatch")
                if saved_meta.get("ftu_alias_audit_sha256") != alias_audit_sha256:
                    raise ValueError("saved alias audit hash mismatch")
                with np.load(npz_path, allow_pickle=False) as saved:
                    saved_radar = saved["radar"]
                    saved_time = saved["time"]
                    saved_hr = saved["heart_rate"]
                    if (list(saved_radar.shape) != saved_meta["radar_shape"]
                            or len(saved_time) != saved_meta["reference_len"]
                            or len(saved_hr) != len(saved_time)
                            or not np.all(np.isfinite(saved_radar))
                            or not np.all(np.isfinite(saved_time))
                            or not np.all(np.isfinite(saved_hr))):
                        raise ValueError("saved numeric arrays are invalid")
                manifest_rows.append({
                    "sample_tag": sample_tag,
                    "participant_id": pid,
                    "scenario": scenario,
                    "distance": distance,
                    "repeat": repeat,
                    "ftu_adc_frontend_fingerprint": frontend_config.fingerprint,
                    "ftu_alias_audit_sha256": alias_audit_sha256 or "none",
                    "npz_path": str(npz_path),
                    "radar_shape": str(tuple(saved_meta["radar_shape"])),
                    "reference_len": saved_meta["reference_len"],
                    "status": "ok",
                })
                success += 1
                print_sample_progress(
                    dataset_name="FTU", current=i, total=len(samples),
                    sample_tag=sample_tag, success=success, failure=failure,
                    started_at=started_at, status="resume-verified",
                )
                continue
            except Exception as exc:
                print(f"[FTU] resume check failed for {sample_tag}; re-exporting: {exc}", flush=True)

        try:
            if UNIFY_TO_RDA:
                radar = loader.open_radar_frame_view(
                    participant_id=pid, scenario=scenario,
                    distance=distance, repeat=repeat,
                )
            else:
                radar = loader.load_radar_data(
                    participant_id=pid, scenario=scenario,
                    distance=distance, repeat=repeat,
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
                radar_out, selected_range_bins, center_range_bin = frontend_config.convert(radar)
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
                    range_selection_method=(frontend_config.range_selection_method
                                            if UNIFY_TO_RDA else "none"),
                    range_center_bin=center_range_bin,
                    range_selected_bins=selected_range_bins,
                ),
                **build_common_reference_meta(unified_ref),
            }
            if UNIFY_TO_RDA:
                meta["ftu_adc_frontend_config"] = frontend_config.to_dict()
                meta["ftu_adc_frontend_fingerprint"] = frontend_config.fingerprint
            if alias_audit_sha256 is not None:
                meta["ftu_alias_audit_sha256"] = alias_audit_sha256
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
                        "ftu_adc_frontend_fingerprint": (
                            frontend_config.fingerprint if UNIFY_TO_RDA else "none"
                        ),
                        "ftu_alias_audit_sha256": alias_audit_sha256 or "none",
                    },
                )
            )
            success += 1
            status = "ok"
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
                        "ftu_adc_frontend_fingerprint": (
                            frontend_config.fingerprint if UNIFY_TO_RDA else "none"
                        ),
                        "ftu_alias_audit_sha256": alias_audit_sha256 or "none",
                    },
                )
            )
            failure += 1
            status = f"failed: {e}"

        print_sample_progress(
            dataset_name="FTU",
            current=i,
            total=len(samples),
            sample_tag=sample_tag,
            success=success,
            failure=failure,
            started_at=started_at,
            status=status,
        )

    print_stage("FTU", "Stage 4/4: write manifest")
    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    print(
        f"[FTU] finished: ok={success}, failed={failure}, elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
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
    resume: bool = False,
    frontend_config: FTUADCConfig | None = None,
) -> Tuple[int, int]:
    started_at = time.monotonic()
    print_stage("BGT60", "Stage 1/4: initialize loader")
    from .bgt60_loader import BGT60TR13CDataLoader

    loader = BGT60TR13CDataLoader(str(dataset_root / "BGT60TR13C"))
    frontend_config = frontend_config or FTUADCConfig()
    dataset_out = output_dir / "BGT60TR13C"
    manifest_rows: List[Dict[str, Any]] = []
    existing_rows: Dict[str, Dict[str, Any]] = {}
    if resume and (dataset_out / "manifest.csv").exists():
        with (dataset_out / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            existing_rows = {row["sample_tag"]: row for row in csv.DictReader(handle)}

    print_stage("BGT60", "Stage 2/4: discover samples")
    samples = iter_bgt_samples(loader, include_long=include_long)
    print(f"[BGT60] discovered {len(samples)} sample(s)", flush=True)
    success = 0
    failure = 0

    print_stage("BGT60", "Stage 3/4: export samples")
    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        distance = sample["distance"]
        measurement_type = sample["measurement_type"]
        sample_tag = f"p{pid:02d}_{safe_name(distance)}_{measurement_type}"

        previous = existing_rows.get(sample_tag)
        if previous is not None and previous.get("status") == "ok":
            previous_path = Path(previous["npz_path"])
            if previous_path.is_file() and (dataset_out / "meta" / f"{sample_tag}.json").is_file():
                previous_meta = json.loads((dataset_out / "meta" / f"{sample_tag}.json").read_text())
                previous_fingerprint = previous_meta.get("bgt_adc_frontend_fingerprint")
                legacy_unversioned = (previous_fingerprint is None and
                                      frontend_config == FTUADCConfig() and
                                      previous_meta.get("range_selection_method") ==
                                      "global_energy_center_contiguous_window")
                if previous_fingerprint != frontend_config.fingerprint and not legacy_unversioned:
                    raise ValueError(f"Cannot resume mixed BGT frontends: {sample_tag}")
                manifest_rows.append(previous)
                success += 1
                print_sample_progress(
                    "BGT60", i, len(samples), sample_tag, success, failure,
                    started_at, "resumed",
                )
                continue

        try:
            radar = loader.load_radar_data(
                participant_id=pid,
                distance=distance if measurement_type == "short" else BGT_LONG_FIXED_DISTANCE,
                measurement_type=measurement_type,
                apply_dc_correction=False,
                memory_map=(measurement_type == "long"),
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
                radar_out, selected_range_bins, center_range_bin = frontend_config.convert(radar)
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
                        frontend_config.range_selection_method if UNIFY_TO_RDA else "none"
                    ),
                    range_center_bin=center_range_bin,
                    range_selected_bins=selected_range_bins,
                ),
                **build_common_reference_meta(unified_ref),
                "bgt_adc_frontend_config": frontend_config.to_dict(),
                "bgt_adc_frontend_fingerprint": frontend_config.fingerprint,
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
                        "bgt_adc_frontend_fingerprint": frontend_config.fingerprint,
                    },
                )
            )
            success += 1
            status = "ok"
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
            status = f"failed: {e}"

        print_sample_progress(
            dataset_name="BGT60",
            current=i,
            total=len(samples),
            sample_tag=sample_tag,
            success=success,
            failure=failure,
            started_at=started_at,
            status=status,
        )

    print_stage("BGT60", "Stage 4/4: write manifest")
    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    print(
        f"[BGT60] finished: ok={success}, failed={failure}, elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
    return success, failure


def iter_phys_samples(loader: PhysDriveDataLoader) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session_id in loader.list_sessions():
        session_dir = loader.mmwave_path / session_id
        candidates = []
        for directory in session_dir.iterdir():
            prefix = f"{session_id}_"
            if directory.is_dir() and directory.name.startswith(prefix):
                suffix = directory.name[len(prefix):]
                if suffix.isdigit():
                    candidates.append((int(suffix), directory.name))
        id_counts: Dict[int, int] = {}
        for sample_id, _ in candidates:
            id_counts[sample_id] = id_counts.get(sample_id, 0) + 1
        for sample_id, sample_dir_name in sorted(candidates):
            sample_tag = (
                sample_dir_name if id_counts[sample_id] > 1
                else f"{session_id}_{sample_id:03d}"
            )
            rows.append({
                "session_id": session_id,
                "sample_id": sample_id,
                "sample_dir_name": sample_dir_name,
                "sample_tag": sample_tag,
            })
    return rows


def export_physdrive(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
) -> Tuple[int, int]:
    started_at = time.monotonic()
    print_stage("PhysDrive", "Stage 1/4: initialize loader")
    from .physdrive_loader import PhysDriveDataLoader

    loader = PhysDriveDataLoader(str(dataset_root / "PhysDrive"))
    dataset_out = output_dir / "PhysDrive"
    manifest_rows: List[Dict[str, Any]] = []

    print_stage("PhysDrive", "Stage 2/4: discover samples")
    samples = iter_phys_samples(loader)
    print(f"[PhysDrive] discovered {len(samples)} sample(s)", flush=True)
    success = 0
    failure = 0

    print_stage("PhysDrive", "Stage 3/4: export samples")
    for i, sample in enumerate(samples, start=1):
        session_id = sample["session_id"]
        sample_id = sample["sample_id"]
        sample_dir_name = sample["sample_dir_name"]
        sample_tag = sample["sample_tag"]

        try:
            radar = loader.load_radar_data(
                session_id=session_id,
                sample_id=sample_id,
                return_complex=True,
                normalize_frames=False,
                sample_dir_name=sample_dir_name,
            )
            ref = loader.load_reference_data(
                session_id=session_id,
                sample_id=sample_id,
                sample_dir_name=sample_dir_name,
            )
            unified_ref = to_unified_reference(
                "PhysDrive",
                ref,
                frame_rate_hz=PHYSDRIVE_FRAME_RATE_HZ,
            )
            radar_out = align_rda_shape(radar) if UNIFY_TO_RDA else radar
            if radar_out.shape[0] != unified_ref["time"].size:
                raise ValueError(
                    f"PhysDrive 雷达与 ECG/RESP 帧数不一致: "
                    f"radar={radar_out.shape[0]}, reference={unified_ref['time'].size}"
                )

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
                "source_sample_dir": sample_dir_name,
                **build_common_radar_meta(
                    radar_out=radar_out,
                    radar_source=radar,
                    radar_preprocess="already_rda_aligned",
                    axes=["frames", "doppler", "angle", "range"],
                ),
                **build_common_reference_meta(unified_ref),
                "reference_source": "ecg_resp_neurokit2_derived",
                "reference_frame_rate_hz": PHYSDRIVE_FRAME_RATE_HZ,
                "reference_processing": {
                    "library": "NeuroKit2",
                    "heart_rate_method": "adaptive_rr_interval_qc",
                    "heart_rate_primary_method": PHYSDRIVE_ECG_PRIMARY_METHOD,
                    "heart_rate_fallback_methods": list(PHYSDRIVE_ECG_FALLBACK_METHODS),
                    "respiration_rate_method": PHYSDRIVE_RSP_PEAK_METHOD,
                    "heart_rate_reliable_bpm": list(PHYSDRIVE_ECG_RELIABLE_BPM),
                    "heart_rate_min_coverage": PHYSDRIVE_ECG_MIN_COVERAGE,
                    "heart_rate_max_interpolation_gap_seconds": PHYSDRIVE_ECG_MAX_INTERPOLATION_GAP_SECONDS,
                    "respiration_rate_valid_bpm": [3.0, 60.0],
                },
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
            status = "ok"
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
            status = f"failed: {e}"

        print_sample_progress(
            dataset_name="PhysDrive",
            current=i,
            total=len(samples),
            sample_tag=sample_tag,
            success=success,
            failure=failure,
            started_at=started_at,
            status=status,
        )

    print_stage("PhysDrive", "Stage 4/4: write manifest")
    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    print(
        f"[PhysDrive] finished: ok={success}, failed={failure}, elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )
    return success, failure


def main() -> None:
    args = parse_args()
    ftu_frontend = FTUADCConfig()
    if args.ftu_frontend_config is not None:
        with args.ftu_frontend_config.open(encoding="utf-8") as handle:
            ftu_frontend = FTUADCConfig.from_dict(json.load(handle))
    bgt_frontend: FTUADCConfig = FTUADCConfig()
    if args.bgt_frontend_config is not None:
        with args.bgt_frontend_config.open(encoding="utf-8") as handle:
            bgt_frontend = BGTADCConfig.from_dict(json.load(handle))
    # 优先使用CLI参数,否则使用默认配置
    dataset_root = (Path(args.dataset_root).resolve() if args.dataset_root 
                    else Path(DATASET_ROOT).resolve())
    output_dir = (Path(args.output_dir).resolve() if args.output_dir 
                  else Path(OUTPUT_DIR).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    export_order = ["PhysDrive", "FTU", "BGT60TR13C"]
    selected_datasets = []
    if args.physdrive:
        selected_datasets.append("PhysDrive")
    if args.ftu:
        selected_datasets.append("FTU")
    if args.bgt60:
        selected_datasets.append("BGT60TR13C")
    if not selected_datasets:
        selected_datasets = export_order
    if args.ftu_frontend_config is not None:
        ftu_output = output_dir / "FTU"
        if "FTU" in selected_datasets and ftu_output.exists() and any(ftu_output.iterdir()):
            raise FileExistsError(
                f"FTU frontend configuration requires a fresh output directory: {ftu_output}"
            )
    if args.bgt_frontend_config is not None and "BGT60TR13C" in selected_datasets:
        bgt_output = output_dir / "BGT60TR13C"
        if bgt_output.exists() and any(bgt_output.iterdir()):
            raise FileExistsError(
                f"BGT frontend export requires a fresh output directory: {bgt_output}"
            )

    print_stage("Export", "Stage 0: configuration")
    print(f"[Export] dataset_root: {dataset_root}", flush=True)
    print(f"[Export] output_dir: {output_dir}", flush=True)
    print(f"[Export] selected_datasets: {', '.join(selected_datasets)}", flush=True)
    print(f"[Export] compress_output: {COMPRESS_OUTPUT}", flush=True)
    print(f"[Export] unify_to_rda: {UNIFY_TO_RDA}", flush=True)
    if "FTU" in selected_datasets:
        print(f"[Export] ftu_frontend_fingerprint: {ftu_frontend.fingerprint}", flush=True)
    if "BGT60TR13C" in selected_datasets:
        print(f"[Export] bgt_frontend_fingerprint: {bgt_frontend.fingerprint}", flush=True)

    for dataset_name in selected_datasets:
        if dataset_name == "PhysDrive":
            ok, fail = export_physdrive(
                dataset_root=dataset_root,
                output_dir=output_dir,
                compress=COMPRESS_OUTPUT,
            )
        elif dataset_name == "FTU":
            ok, fail = export_ftu(
                dataset_root=dataset_root,
                output_dir=output_dir,
                compress=COMPRESS_OUTPUT,
                participant_id=FTU_PARTICIPANT_ID,
                scenario=FTU_SCENARIO,
                distance=FTU_DISTANCE,
                frontend_config=ftu_frontend,
                alias_audit=args.ftu_alias_audit,
            )
        elif dataset_name == "BGT60TR13C":
            ok, fail = export_bgt60(
                dataset_root=dataset_root,
                output_dir=output_dir,
                compress=COMPRESS_OUTPUT,
                include_long=BGT_INCLUDE_LONG and not args.bgt_skip_long,
                frontend_config=bgt_frontend,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        summary[dataset_name] = {"success": ok, "failed": fail}

    save_json(output_dir / "export_summary.json", summary)
    print("\n=== Export Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
