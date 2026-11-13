"""Evaluate FFT and STFT signal-processing baselines on training_exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.common.metrics import aggregate, participant_id
from src.training.common.spectrum import band_mask, peak_from_spectrum
from src.training.datasets import RadarWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FFT/STFT HR baselines without training.")
    parser.add_argument("--method", choices=["fft", "stft"], required=True)
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        default=None,
        choices=["FTU", "BGT60TR13C", "PhysDrive"],
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sampling-rate-hz", type=float, default=20.0)
    parser.add_argument("--hr-min-bpm", type=float, default=40.0)
    parser.add_argument("--hr-max-bpm", type=float, default=180.0)
    parser.add_argument("--stft-window", type=int, default=64)
    parser.add_argument("--stft-hop", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="Export per-window predictions with original record/window identities.",
    )
    parser.add_argument(
        "--fft-pad",
        type=int,
        default=1,
        help="Zero-padding factor for the FFT. 1 keeps the historical behaviour; "
        "4-8 gives sub-bin resolution (default bin is 4.7 bpm @20Hz / 7.0 bpm @30Hz "
        "with 256-frame windows).",
    )
    parser.add_argument(
        "--peak-refine",
        action="store_true",
        help="Parabolic interpolation around the spectral peak (sub-bin HR estimate). "
        "Has no effect when the export ships a fixed-grid x_freq spectrum.",
    )
    return parser.parse_args()


def load_window(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].astype(np.float32, copy=False) for key in data.files if key != "meta_json"}


def predict_fft(
    arrays: Dict[str, np.ndarray],
    sampling_rate_hz: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    pad: int = 1,
    refine: bool = False,
) -> float:
    if "x_freq" in arrays:
        # The export ships a spectrum computed on a fixed grid; padding cannot be
        # undone afterwards, so only peak refinement applies here.
        x_freq = np.asarray(arrays["x_freq"], dtype=np.float32)
        spectrum = np.mean(x_freq, axis=0) if x_freq.ndim == 2 else np.ravel(x_freq)
        if "freq_hz" in arrays and arrays["freq_hz"].size == spectrum.size:
            freq_hz = np.asarray(arrays["freq_hz"], dtype=np.float32)
        else:
            freq_hz = np.fft.rfftfreq((spectrum.size - 1) * 2, d=1.0 / sampling_rate_hz).astype(np.float32)
        return peak_from_spectrum(spectrum, freq_hz, hr_min_bpm, hr_max_bpm, refine=refine)

    x_time = np.asarray(arrays["x_time"], dtype=np.float32)
    signal = np.mean(x_time, axis=0) if x_time.ndim == 2 else np.ravel(x_time)
    window = np.hanning(signal.size).astype(np.float32)
    nfft = max(int(signal.size) * max(1, int(pad)), int(signal.size))
    spectrum = np.log1p(np.abs(np.fft.rfft(signal * window, n=nfft)))
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate_hz).astype(np.float32)
    return peak_from_spectrum(spectrum, freq_hz, hr_min_bpm, hr_max_bpm, refine=refine)


def predict_stft(
    arrays: Dict[str, np.ndarray],
    sampling_rate_hz: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    window_size: int,
    hop: int,
    pad: int = 1,
    refine: bool = False,
) -> float:
    x_time = np.asarray(arrays["x_time"], dtype=np.float32)
    signals = x_time if x_time.ndim == 2 else x_time[None, :]
    n = signals.shape[-1]
    if n < window_size:
        return predict_fft(
            arrays, sampling_rate_hz, hr_min_bpm, hr_max_bpm, pad=pad, refine=refine
        )

    nfft = max(window_size * max(1, int(pad)), window_size)
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate_hz).astype(np.float32)
    window = np.hanning(window_size).astype(np.float32)
    track: list[float] = []
    for start in range(0, n - window_size + 1, max(1, hop)):
        segment = signals[:, start : start + window_size] * window[None, :]
        power = np.mean(np.log1p(np.abs(np.fft.rfft(segment, axis=-1, n=nfft))), axis=0)
        pred = peak_from_spectrum(power, freq_hz, hr_min_bpm, hr_max_bpm, refine=refine)
        if np.isfinite(pred):
            track.append(pred)
    return float(np.median(track)) if track else math.nan


def main() -> None:
    args = parse_args()
    target_datasets = set(args.target_datasets) if args.target_datasets else None
    dataset = RadarWindowDataset(args.export_dir, args.split, datasets=target_datasets)
    rows: list[Dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        arrays = load_window(item["window_path"])
        if args.method == "fft":
            pred = predict_fft(
                arrays,
                args.sampling_rate_hz,
                args.hr_min_bpm,
                args.hr_max_bpm,
                pad=args.fft_pad,
                refine=args.peak_refine,
            )
        else:
            pred = predict_stft(
                arrays,
                args.sampling_rate_hz,
                args.hr_min_bpm,
                args.hr_max_bpm,
                args.stft_window,
                args.stft_hop,
                pad=args.fft_pad,
                refine=args.peak_refine,
            )
        label = float(item["y_bpm"])
        if not np.isfinite(pred):
            pred = float(dataset.label_stats.mean)
        dataset_name = str(item["dataset"])
        group_key = str(item["group_key"])
        sample_tag = str(item["sample_tag"])
        rows.append(
            {
                "dataset": dataset_name,
                "group_key": group_key,
                "sample_tag": sample_tag,
                "participant_id": participant_id(dataset_name, group_key, sample_tag),
                "window_start": item.get("window_start", -1),
                "label_bpm": label,
                "pred_bpm": float(pred),
                "abs_error_bpm": abs(float(pred) - label),
            }
        )
        if (idx + 1) % 1000 == 0:
            print(f"processed {idx + 1}/{len(dataset)} windows", flush=True)

    result = {
        "method": args.method,
        "target": {
            "export_dir": str(args.export_dir),
            "split": args.split,
            "datasets": sorted(target_datasets) if target_datasets else "all",
            "size": len(dataset),
        },
        "parameters": {
            "sampling_rate_hz": args.sampling_rate_hz,
            "hr_min_bpm": args.hr_min_bpm,
            "hr_max_bpm": args.hr_max_bpm,
            "stft_window": args.stft_window,
            "stft_hop": args.stft_hop,
            "fft_pad": args.fft_pad,
            "peak_refine": bool(args.peak_refine),
            "bin_resolution_bpm": 60.0
            * args.sampling_rate_hz
            / float(max(1, args.fft_pad) * (args.stft_window if args.method == "stft" else 256)),
        },
        "overall": aggregate(rows)["overall"],
        "by_dataset": aggregate(rows, "dataset"),
        "by_participant": aggregate(rows, "participant_id"),
        "by_group": aggregate(rows, "group_key"),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output_json"] = str(args.output_json)
    if args.dump_predictions is not None:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset",
                    "group_key",
                    "participant_id",
                    "sample_tag",
                    "window_start",
                    "label_bpm",
                    "pred_bpm",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        result["dump_predictions"] = str(args.dump_predictions)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
