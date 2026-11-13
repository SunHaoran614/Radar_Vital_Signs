"""Build fixed-window training data from unified exported radar datasets.

Input:
    exports/{DATASET}/manifest.csv
    exports/{DATASET}/samples/*.npz

Output:
    training_exports/windows/{DATASET}/{train,val,test}/*.npz
    training_exports/manifest.csv
    training_exports/build_config.json
    training_exports/summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.edacm import (
    DEFAULT_EDACM_CANDIDATE_MULTIPLIER,
    DEFAULT_EDACM_TOP_BINS,
    DEFAULT_HR_BAND_HZ,
    DEFAULT_RDA_STABILITY_SEGMENTS,
)
from src.features.hr_adavmd import (
    DECOMPOSITION_ALGORITHM_VERSION,
    DEFAULT_RESP_BAND_HZ,
    DEFAULT_SAMPLING_RATE_HZ,
    DEFAULT_VMD_ALPHA,
    DEFAULT_VMD_K,
    DEFAULT_VMD_MAX_ITER,
    DEFAULT_VMD_TOL,
)
from src.features.representations import (
    DOMAIN_ADAPTATION_METHOD,
    INNOVATION_MODULES,
    METHOD_PIPELINE,
    DEFAULT_RDA_REPRESENTATION,
    radar_to_feature_bundle,
)


DEFAULT_EXPORTS_DIR = ROOT / "exports"
DEFAULT_OUTPUT_DIR = ROOT / "training_exports"
DEFAULT_DATASETS = ("FTU", "BGT60TR13C", "PhysDrive")
DEFAULT_WINDOW_SIZE = 256
DEFAULT_STRIDE = 128
DEFAULT_MIN_LABEL_COVERAGE = 0.8
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_BGT_LONG_SPLIT = "test"
DEFAULT_REPRESENTATION = "proposed"
DEFAULT_SPLIT_MODE = "balanced_grouped"
SOURCE_SAMPLING_RATE_HZ = {"FTU": 20.0, "BGT60TR13C": 30.0, "PhysDrive": 20.0}
SOURCE_HR_BAND_HZ = {
    "FTU": (0.75, 3.5),  # Elevated Polar reference includes >150 BPM.
    "BGT60TR13C": DEFAULT_HR_BAND_HZ,
    "PhysDrive": DEFAULT_HR_BAND_HZ,
}
SOURCE_VMD_INIT_PROFILE = {dataset: "legacy" for dataset in SOURCE_HR_BAND_HZ}
BALANCED_EXHAUSTIVE_MAX_GROUPS = 14
FTU_SPECIAL_PARTICIPANTS = ("2", "5", "6")
FTU_ELEVATED_HR_PARTICIPANTS = ("2", "3", "4", "6")
DEFAULT_PARTICIPANT_SPLITS = {
    "FTU": {
        # Frozen FTU source protocol used by training_exports_protocol_v2 and
        # all raw-ADC train/val diagnostics. Test participants remain sealed.
        "train": ("1", "2", "4", "7", "8", "9"),
        "val": ("5", "6"),
        "test": ("3", "10"),
    },
    "BGT60TR13C": {
        "train": ("1", "2", "4", "6", "7", "8"),
        "val": ("5",),
        "test": ("3",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-window training dataset from unified exports."
    )
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to include.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--representation",
        choices=[
            "proposed",
            "target_edacm_vmd",
            "target_edacm",
            "log_magnitude",
            "magnitude",
            "real_imag",
        ],
        default=DEFAULT_REPRESENTATION,
        help="How to convert complex radar tensors for model input.",
    )
    parser.add_argument(
        "--target",
        default="heart_rate",
        help="Training target. This builder currently supports heart_rate only.",
    )
    parser.add_argument(
        "--label-reduction",
        choices=["mean", "median", "center"],
        default="mean",
        help="How to convert frame-level labels into one window label.",
    )
    parser.add_argument(
        "--min-label-coverage",
        type=float,
        default=DEFAULT_MIN_LABEL_COVERAGE,
        help="Minimum non-NaN label fraction inside a window for required targets.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "window_zscore"],
        default="window_zscore",
        help="Feature normalization applied after representation conversion.",
    )
    parser.add_argument(
        "--max-windows-per-sample",
        type=int,
        default=None,
        help="Optional cap for quick experiments/debugging.",
    )
    parser.add_argument(
        "--drop-rda",
        action="store_true",
        help=(
            "Skip construction and storage of per-window x_rda. No current training "
            "entrypoint reads it; this saves memory and I/O per window."
        ),
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Mark built windows as ineligible for source-model training.",
    )
    parser.add_argument(
        "--only-splits", nargs="+", choices=("train", "val", "test"),
        default=None,
        help="Build windows only for these already assigned splits; other sample arrays stay unopened.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--split-mode",
        choices=["stable_grouped", "balanced_grouped"],
        default=DEFAULT_SPLIT_MODE,
        help=(
            "stable_grouped keeps the old seed-hash group split; "
            "balanced_grouped keeps groups intact and balances HR label distributions."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def print_stage(message: str) -> None:
    print(f"\n[builder] {message}", flush=True)


def print_progress(current: int, total: int, message: str) -> None:
    if total <= 0:
        bar = "------------------------"
        pct = 0.0
    else:
        ratio = min(max(current / total, 0.0), 1.0)
        done = int(round(ratio * 24))
        bar = "#" * done + "-" * (24 - done)
        pct = ratio * 100.0
    print(f"[builder] [{bar}] {pct:5.1f}% {current}/{total} {message}", flush=True)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def required_targets(target: str) -> Tuple[str, ...]:
    if target != "heart_rate":
        raise ValueError("This training dataset builder currently supports HR only: --target heart_rate")
    return ("heart_rate",)


def validate_split_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split-ratios must contain exactly 3 values")
    vals = tuple(float(x) for x in ratios)
    if any(x < 0 for x in vals):
        raise ValueError("split-ratios cannot contain negative values")
    total = sum(vals)
    if total <= 0:
        raise ValueError("split-ratios must sum to a positive value")
    return tuple(x / total for x in vals)  # type: ignore[return-value]


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_group_keys(
    group_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    forced_test_groups: Optional[set[str]] = None,
) -> Dict[str, str]:
    forced_test_groups = forced_test_groups or set()
    unique_groups = sorted(set(group_keys))
    flexible_groups = [g for g in unique_groups if g not in forced_test_groups]
    flexible_groups.sort(key=lambda g: stable_unit_interval(g, seed))

    n = len(flexible_groups)
    assignment: Dict[str, str] = {g: "test" for g in forced_test_groups if g in unique_groups}
    if n == 0:
        return assignment
    if n == 1:
        assignment[flexible_groups[0]] = "train"
        return assignment
    if n == 2:
        assignment[flexible_groups[0]] = "train"
        assignment[flexible_groups[1]] = "test"
        return assignment

    train_ratio, val_ratio, test_ratio = ratios
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test

    for group in flexible_groups[:n_train]:
        assignment[group] = "train"
    for group in flexible_groups[n_train : n_train + n_val]:
        assignment[group] = "val"
    for group in flexible_groups[n_train + n_val :]:
        assignment[group] = "test"
    return assignment


def split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    _, val_ratio, test_ratio = ratios
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def participant_id_from_group(group_key: str) -> Optional[str]:
    parts = group_key.split("/")
    if len(parts) >= 3 and parts[-2] == "participant":
        return parts[-1]
    return None


def default_participant_split(
    dataset: str,
    group_keys: Sequence[str],
    forced_test_groups: Optional[set[str]] = None,
) -> Optional[Dict[str, str]]:
    if forced_test_groups:
        return None
    spec = DEFAULT_PARTICIPANT_SPLITS.get(dataset)
    if spec is None:
        return None
    unique_groups = sorted(set(group_keys))
    group_by_participant = {
        participant_id_from_group(group): group
        for group in unique_groups
        if participant_id_from_group(group) is not None
    }
    requested = {
        participant_id
        for participant_ids in spec.values()
        for participant_id in participant_ids
    }
    available = set(group_by_participant)
    if dataset == "FTU":
        # Diagnostic subsets still keep each observed participant in its frozen
        # split; never silently reassign a sealed test subject by hash.
        if not available or not available.issubset(requested) or len(group_by_participant) != len(unique_groups):
            return None
    elif requested != available:
        return None
    return {
        group_by_participant[participant_id]: split
        for split, participant_ids in spec.items()
        for participant_id in participant_ids
        if participant_id in group_by_participant
    }


def stratified_group_split_by_label_range(
    group_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    group_label_means: Dict[str, float],
    forced_test_groups: Optional[set[str]] = None,
) -> Optional[Dict[str, str]]:
    if forced_test_groups:
        return None
    unique_groups = sorted(set(group_keys))
    if any(group not in group_label_means for group in unique_groups):
        return None

    n_train, n_val, n_test = split_counts(len(unique_groups), ratios)
    n_eval = n_val + n_test
    if n_eval <= 0 or n_train <= 0 or len(unique_groups) < n_eval * 2:
        return None

    ordered = sorted(
        unique_groups,
        key=lambda group: (group_label_means[group], stable_unit_interval(group, seed)),
    )
    base_size = len(ordered) // n_eval
    remainder = len(ordered) % n_eval
    strata: List[List[str]] = []
    start = 0
    for idx in range(n_eval):
        size = base_size + int(idx < remainder)
        strata.append(ordered[start : start + size])
        start += size

    assignment: Dict[str, str] = {}
    val_count = 0
    test_count = 0
    for idx, stratum in enumerate(strata):
        if not stratum:
            continue
        candidate_order = sorted(
            stratum,
            key=lambda group: (
                abs(group_label_means[group] - float(np.mean([group_label_means[g] for g in stratum]))),
                stable_unit_interval(group, seed),
            ),
        )
        selected = candidate_order[0]
        if val_count >= n_val:
            split = "test"
        elif test_count >= n_test:
            split = "val"
        elif val_count <= test_count:
            split = "val"
        else:
            split = "test"
        assignment[selected] = split
        val_count += int(split == "val")
        test_count += int(split == "test")

    for group in unique_groups:
        assignment.setdefault(group, "train")

    if (
        sum(1 for split in assignment.values() if split == "train") != n_train
        or sum(1 for split in assignment.values() if split == "val") != n_val
        or sum(1 for split in assignment.values() if split == "test") != n_test
    ):
        return None
    return assignment


def group_mean_label(sample: Dict[str, Any]) -> Optional[float]:
    try:
        data = np.load(Path(sample["npz_path"]), allow_pickle=False)
        labels = np.asarray(data["heart_rate"], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    valid = labels[np.isfinite(labels)]
    if valid.size == 0:
        return None
    return float(np.mean(valid))


def compute_group_label_means(
    samples: Sequence[Dict[str, Any]],
    show_progress: bool = False,
) -> Dict[str, float]:
    values_by_group: Dict[str, List[float]] = {}
    for idx, sample in enumerate(samples, start=1):
        mean_label = group_mean_label(sample)
        if mean_label is None:
            if show_progress and (idx == 1 or idx == len(samples) or idx % 100 == 0):
                print_progress(idx, len(samples), f"label mean scan: skip {sample['dataset']}/{sample['sample_tag']}")
            continue
        values_by_group.setdefault(get_group_key(sample), []).append(mean_label)
        if show_progress and (idx == 1 or idx == len(samples) or idx % 100 == 0):
            print_progress(idx, len(samples), f"label mean scan: {sample['dataset']}/{sample['sample_tag']}")
    return {
        group: float(np.mean(values))
        for group, values in values_by_group.items()
        if values
    }


def ftu_extreme_balance_penalty(split_groups: Dict[str, Sequence[str]]) -> float:
    """Softly distribute known FTU special cases across train/val/test.

    The 4TU/FTU paper marks participant 2 as an experienced meditator and
    participants 5/6 as asthma cases. We avoid placing all such cases in one
    split, while still letting HR distribution balance dominate.
    """
    special = set(FTU_SPECIAL_PARTICIPANTS)
    counts: Dict[str, int] = {}
    for split, groups in split_groups.items():
        counts[split] = sum(
            1
            for group in groups
            if participant_id_from_group(group) in special
        )
    missing_eval = int(counts.get("val", 0) == 0) + int(counts.get("test", 0) == 0)
    missing_train = int(counts.get("train", 0) == 0)
    concentration = max(counts.values(), default=0) - min(counts.values(), default=0)
    return 2.0 * missing_eval + 1.0 * missing_train + 0.25 * concentration


def greedy_balanced_split_group_keys(
    group_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    group_label_means: Dict[str, float],
    forced_test_groups: Optional[set[str]] = None,
) -> Dict[str, str]:
    forced_test_groups = forced_test_groups or set()
    unique_groups = sorted(set(group_keys))
    assignment: Dict[str, str] = {g: "test" for g in forced_test_groups if g in unique_groups}
    flexible_groups = [g for g in unique_groups if g not in forced_test_groups]
    n_train, n_val, n_test_total = split_counts(len(unique_groups), ratios)
    targets = {
        "train": n_train,
        "val": n_val,
        "test": n_test_total,
    }
    targets["test"] = max(0, targets["test"] - len(assignment))
    if targets["val"] + targets["test"] >= len(flexible_groups) and len(flexible_groups) >= 3:
        targets["val"] = 1
        targets["test"] = 1
    targets["train"] = len(flexible_groups) - targets["val"] - targets["test"]
    if targets["train"] <= 0:
        return split_group_keys(group_keys, ratios, seed, forced_test_groups)

    global_mean = float(np.mean([group_label_means[g] for g in flexible_groups]))
    split_groups: Dict[str, List[str]] = {"train": [], "val": [], "test": sorted(assignment)}
    split_sums = {
        "train": 0.0,
        "val": 0.0,
        "test": sum(group_label_means[g] for g in assignment if g in group_label_means),
    }
    order = sorted(
        flexible_groups,
        key=lambda g: (-abs(group_label_means[g] - global_mean), stable_unit_interval(g, seed)),
    )
    for group in order:
        best_split: Optional[str] = None
        best_score: Optional[float] = None
        for split in ("train", "val", "test"):
            if len(split_groups[split]) >= targets[split]:
                continue
            next_count = len(split_groups[split]) + 1
            next_mean = (split_sums[split] + group_label_means[group]) / next_count
            fill_ratio = next_count / max(1, targets[split])
            score = abs(next_mean - global_mean) - 0.01 * fill_ratio
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        if best_split is None:
            best_split = "train"
        split_groups[best_split].append(group)
        split_sums[best_split] += group_label_means[group]

    return {
        group: split
        for split, groups in split_groups.items()
        for group in groups
    }


def balanced_split_group_keys(
    dataset: str,
    group_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    group_label_means: Dict[str, float],
    forced_test_groups: Optional[set[str]] = None,
) -> Dict[str, str]:
    forced_test_groups = forced_test_groups or set()
    unique_groups = sorted(set(group_keys))
    default_assignment = default_participant_split(dataset, group_keys, forced_test_groups)
    if default_assignment is not None:
        return default_assignment
    if dataset == "PhysDrive":
        stratified_assignment = stratified_group_split_by_label_range(
            group_keys=group_keys,
            ratios=ratios,
            seed=seed,
            group_label_means=group_label_means,
            forced_test_groups=forced_test_groups,
        )
        if stratified_assignment is not None:
            return stratified_assignment

    assignment: Dict[str, str] = {g: "test" for g in forced_test_groups if g in unique_groups}
    flexible_groups = [g for g in unique_groups if g not in forced_test_groups]
    n_train, n_val, n_test_total = split_counts(len(unique_groups), ratios)
    n_test_flexible = max(0, n_test_total - len(assignment))
    if n_test_flexible + n_val >= len(flexible_groups) and len(flexible_groups) >= 3:
        n_val = 1
        n_test_flexible = 1
    n_train_flexible = len(flexible_groups) - n_val - n_test_flexible
    if n_train_flexible <= 0:
        return split_group_keys(group_keys, ratios, seed, forced_test_groups)

    missing = [g for g in flexible_groups if g not in group_label_means]
    if missing:
        return split_group_keys(group_keys, ratios, seed, forced_test_groups)

    if len(flexible_groups) > BALANCED_EXHAUSTIVE_MAX_GROUPS:
        return greedy_balanced_split_group_keys(
            group_keys=group_keys,
            ratios=ratios,
            seed=seed,
            group_label_means=group_label_means,
            forced_test_groups=forced_test_groups,
        )

    best: Optional[Tuple[float, Dict[str, str]]] = None
    flexible_set = set(flexible_groups)
    forced_test = sorted(assignment)
    for val_groups_tuple in itertools.combinations(flexible_groups, n_val):
        val_groups = set(val_groups_tuple)
        remaining_after_val = sorted(flexible_set - val_groups)
        for test_groups_tuple in itertools.combinations(remaining_after_val, n_test_flexible):
            test_groups = set(test_groups_tuple)
            train_groups = sorted(flexible_set - val_groups - test_groups)
            if len(train_groups) != n_train_flexible:
                continue

            split_groups = {
                "train": train_groups,
                "val": sorted(val_groups),
                "test": sorted(test_groups | set(forced_test)),
            }
            split_means = {
                split: float(np.mean([group_label_means[g] for g in groups]))
                for split, groups in split_groups.items()
                if groups
            }
            train_mean = split_means.get("train")
            val_mean = split_means.get("val")
            test_mean = split_means.get("test")
            if train_mean is None or val_mean is None or test_mean is None:
                continue
            score = (
                abs(val_mean - train_mean)
                + abs(test_mean - train_mean)
                + 0.5 * abs(val_mean - test_mean)
            )
            score += 1e-6 * sum(stable_unit_interval(g, seed) for groups in split_groups.values() for g in groups)
            if dataset == "FTU":
                score += ftu_extreme_balance_penalty(split_groups)

            candidate_assignment = {
                **{g: "train" for g in train_groups},
                **{g: "val" for g in val_groups},
                **{g: "test" for g in test_groups},
                **assignment,
            }
            candidate = (score, candidate_assignment)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        return split_group_keys(group_keys, ratios, seed, forced_test_groups)
    return best[1]


def build_domain_split_map(
    samples: Sequence[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    seed: int,
    bgt_long_participants: set[str],
    split_mode: str,
    group_label_means: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    groups_by_dataset: Dict[str, List[str]] = {}
    for sample in samples:
        groups_by_dataset.setdefault(sample["dataset"], []).append(get_group_key(sample))
    group_label_means = group_label_means or {}

    split_map: Dict[str, str] = {}
    forced_test_groups = {
        f"BGT60TR13C/participant/{participant_id}"
        for participant_id in bgt_long_participants
    }
    for dataset, group_keys in groups_by_dataset.items():
        dataset_forced_test = forced_test_groups if dataset == "BGT60TR13C" else set()
        if dataset == "FTU":
            frozen = default_participant_split(dataset, group_keys)
            if frozen is None:
                raise ValueError("FTU participants do not match frozen source protocol; refusing an implicit resplit")
            split_map.update(frozen)
            continue
        if split_mode == "balanced_grouped":
            split_map.update(
                balanced_split_group_keys(
                    dataset=dataset,
                    group_keys=group_keys,
                    ratios=ratios,
                    seed=seed,
                    group_label_means=group_label_means,
                    forced_test_groups=dataset_forced_test,
                )
            )
        else:
            split_map.update(
                split_group_keys(
                    group_keys=group_keys,
                    ratios=ratios,
                    seed=seed,
                    forced_test_groups=dataset_forced_test,
                )
            )
    return split_map


def get_group_key(sample: Dict[str, Any]) -> str:
    dataset = sample["dataset"]
    row = sample.get("source_row", {})
    sample_tag = sample["sample_tag"]
    if dataset in {"FTU", "BGT60TR13C"}:
        participant_id = row.get("participant_id")
        if participant_id not in (None, ""):
            return f"{dataset}/participant/{participant_id}"
    if dataset == "PhysDrive":
        session_id = row.get("session_id")
        if session_id:
            return f"{dataset}/session/{session_id}"
    return f"{dataset}/sample/{sample_tag}"


def is_bgt_long_sample(sample: Dict[str, Any]) -> bool:
    return (
        sample.get("dataset") == "BGT60TR13C"
        and sample.get("source_row", {}).get("measurement_type") == "long"
    )


def collect_bgt_long_participants(samples: Sequence[Dict[str, Any]]) -> set[str]:
    participants: set[str] = set()
    for sample in samples:
        if not is_bgt_long_sample(sample):
            continue
        participant_id = sample.get("source_row", {}).get("participant_id")
        if participant_id not in (None, ""):
            participants.add(str(participant_id))
    return participants


def resolve_npz_path(row: Dict[str, str], exports_dir: Path, dataset: str) -> Optional[Path]:
    raw = row.get("npz_path", "")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    candidate = exports_dir / dataset / "samples" / path.name
    if candidate.exists():
        return candidate
    return path


def collect_exported_samples(exports_dir: Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    seen_npz: set[Tuple[str, str]] = set()
    for dataset in datasets:
        manifest_path = exports_dir / dataset / "manifest.csv"
        for row in read_manifest(manifest_path):
            if row.get("status") != "ok":
                continue
            npz_path = resolve_npz_path(row, exports_dir, dataset)
            if npz_path is None:
                continue
            sample_id = (dataset, str(npz_path.resolve()))
            if sample_id in seen_npz:
                continue
            seen_npz.add(sample_id)
            sample_tag = row.get("sample_tag") or npz_path.stem
            samples.append(
                {
                    "dataset": dataset,
                    "sample_tag": sample_tag,
                    "npz_path": npz_path,
                    "source_row": row,
                }
            )
    return samples


# Feature extraction is implemented in src.features.representations.


def normalize_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=False)
    if mode != "window_zscore":
        raise ValueError(f"Unsupported normalize mode: {mode}")
    y = x.astype(np.float32, copy=True)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if np.isfinite(std) and std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    y[~np.isfinite(y)] = 0.0
    return y


def build_frame_times(time: np.ndarray, n_frames: int) -> np.ndarray:
    ref_time = np.asarray(time, dtype=np.float32).reshape(-1)
    if n_frames <= 0:
        return np.array([], dtype=np.float32)
    if ref_time.size == n_frames:
        return ref_time
    if ref_time.size >= 2:
        start = float(np.nanmin(ref_time))
        end = float(np.nanmax(ref_time))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return np.linspace(start, end, n_frames, dtype=np.float32)
    return np.arange(n_frames, dtype=np.float32)


def dataset_frame_times(dataset: str, ref_time: np.ndarray, n_frames: int) -> np.ndarray:
    """Assign native radar times; reference endpoints do not set radar duration."""
    if dataset == "PhysDrive":
        return build_frame_times(ref_time, n_frames)
    return np.arange(n_frames, dtype=np.float32) / SOURCE_SAMPLING_RATE_HZ[dataset]


def native_window_geometry(dataset: str, window_size: int, stride: int) -> Tuple[int, int]:
    ratio = SOURCE_SAMPLING_RATE_HZ[dataset] / DEFAULT_SAMPLING_RATE_HZ
    native_size = window_size * ratio
    native_stride = stride * ratio
    if not native_size.is_integer() or not native_stride.is_integer():
        raise ValueError(f"{dataset}: window-size and stride must map to whole native frames")
    return int(native_size), int(native_stride)


def radar_window_at_target_rate(radar_window: np.ndarray, dataset: str, target_size: int) -> np.ndarray:
    if dataset != "BGT60TR13C":
        return radar_window
    result = resample_poly(radar_window, up=2, down=3, axis=0)
    if result.shape[0] != target_size:
        raise ValueError(f"BGT resampling produced {result.shape[0]} frames, expected {target_size}")
    return result.astype(np.complex64, copy=False)


def label_window_at_target_rate(values: np.ndarray, dataset: str, target_size: int) -> np.ndarray:
    if dataset != "BGT60TR13C":
        return np.asarray(values, dtype=np.float32)
    target_positions = np.arange(target_size, dtype=np.float32) * np.float32(1.5)
    return np.interp(
        target_positions,
        np.arange(len(values), dtype=np.float32),
        np.asarray(values, dtype=np.float32),
    ).astype(np.float32)


def interpolate_label_to_frames(
    label: np.ndarray,
    ref_time: np.ndarray,
    frame_time: np.ndarray,
) -> np.ndarray:
    y = np.asarray(label, dtype=np.float32).reshape(-1)
    t = np.asarray(ref_time, dtype=np.float32).reshape(-1)
    n = min(y.size, t.size)
    if n == 0:
        return np.full(frame_time.shape, np.nan, dtype=np.float32)
    y = y[:n]
    t = t[:n]
    # 已逐帧对齐的标签不能重插值，否则 NeuroKit 标记的 NaN 缺口会被填平。
    if n == frame_time.size and np.array_equal(t, frame_time):
        return y.copy()
    valid = np.isfinite(y) & np.isfinite(t)
    if int(valid.sum()) == 0:
        return np.full(frame_time.shape, np.nan, dtype=np.float32)
    if int(valid.sum()) == 1:
        return np.full(frame_time.shape, float(y[valid][0]), dtype=np.float32)
    order = np.argsort(t[valid])
    valid_t = t[valid][order]
    valid_y = y[valid][order]
    unique_t, unique_idx = np.unique(valid_t, return_index=True)
    unique_y = valid_y[unique_idx]
    if unique_t.size == 1:
        return np.full(frame_time.shape, float(unique_y[0]), dtype=np.float32)
    out = np.interp(frame_time, unique_t, unique_y, left=np.nan, right=np.nan)
    return out.astype(np.float32)


def reduce_label(values: np.ndarray, mode: str) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(valid))
    if mode == "median":
        return float(np.median(valid))
    if mode == "center":
        center = len(values) // 2
        if np.isfinite(values[center]):
            return float(values[center])
        return float(valid[len(valid) // 2])
    raise ValueError(f"Unsupported label reduction: {mode}")


def label_coverage(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def iter_window_starts(n_frames: int, window_size: int, stride: int) -> List[int]:
    if n_frames < window_size:
        return []
    return list(range(0, n_frames - window_size + 1, stride))


def save_window(
    path: Path,
    feature_bundle: Dict[str, Any],
    labels: Dict[str, float],
    label_series: Dict[str, np.ndarray],
    meta: Dict[str, Any],
) -> None:
    arrays: Dict[str, Any] = {
        "label_heart_rate": np.float32(labels.get("heart_rate", np.nan)),
        "label_series_heart_rate": label_series["heart_rate"].astype(np.float32),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
    }
    for key in ("x", "x_time", "x_freq", "x_rda", "freq_hz"):
        if key in feature_bundle:
            arrays[key] = np.asarray(feature_bundle[key], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to continue."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "windows").mkdir(parents=True, exist_ok=True)


def build_training_dataset(args: argparse.Namespace) -> None:
    exports_dir = Path(args.exports_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_ratios = validate_split_ratios(args.split_ratios)
    required = required_targets(args.target)

    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not 0.0 <= args.min_label_coverage <= 1.0:
        raise ValueError("--min-label-coverage must be between 0 and 1")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print_stage("Stage 1/4: collect exported samples")
    samples = collect_exported_samples(exports_dir, args.datasets)
    ftu_frontend_fingerprints = {
        sample["source_row"].get("ftu_adc_frontend_fingerprint", "").strip()
        for sample in samples if sample["dataset"] == "FTU"
    }
    if len(ftu_frontend_fingerprints) > 1:
        raise ValueError(
            "FTU source manifest mixes ADC frontends; build each frontend into a separate export"
        )
    ftu_frontend_fingerprint = next(iter(ftu_frontend_fingerprints), "") or "unversioned"
    ftu_alias_audit_hashes = {
        sample["source_row"].get("ftu_alias_audit_sha256", "").strip()
        for sample in samples if sample["dataset"] == "FTU"
    }
    if len(ftu_alias_audit_hashes) > 1:
        raise ValueError("FTU source manifest mixes physical-alias audit versions")
    ftu_alias_audit_sha256 = next(iter(ftu_alias_audit_hashes), "") or "unversioned"
    bgt_long_participants = collect_bgt_long_participants(samples)
    print(f"[builder] collected {len(samples)} exported sample(s)", flush=True)
    group_label_means: Dict[str, float] = {}
    if args.split_mode == "balanced_grouped":
        print_stage("Stage 1/4: scan non-FTU group label means for balanced split")
        # FTU assignment is frozen. Reading its val/test labels to compute a
        # split score would be unnecessary and would weaken the sealed protocol.
        group_label_means = compute_group_label_means(
            [sample for sample in samples if sample["dataset"] != "FTU"],
            show_progress=True,
        )
    domain_split_map = build_domain_split_map(
        samples=samples,
        ratios=split_ratios,
        seed=args.seed,
        bgt_long_participants=bgt_long_participants,
        split_mode=args.split_mode,
        group_label_means=group_label_means,
    )
    if bgt_long_participants:
        print(
            (
                f"[builder] BGT60 long participant(s) forced to {DEFAULT_BGT_LONG_SPLIT}: "
                f"{', '.join(sorted(bgt_long_participants))}"
            ),
            flush=True,
        )

    manifest_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "samples_seen": len(samples),
        "samples_used": 0,
        "windows_written": 0,
        "windows_skipped_label": 0,
        "windows_skipped_short": 0,
        "by_dataset": {},
        "by_split": {"train": 0, "val": 0, "test": 0},
        "groups_by_dataset_split": {
            dataset: {"train": [], "val": [], "test": []}
            for dataset in args.datasets
        },
        "group_label_means": {},
        "split_label_means": {
            dataset: {"train": None, "val": None, "test": None}
            for dataset in args.datasets
        },
    }

    print_stage("Stage 2/4: build fixed windows")
    for sample_idx, sample in enumerate(samples, start=1):
        dataset = sample["dataset"]
        sample_tag = sample["sample_tag"]
        sample_key = f"{dataset}/{sample_tag}"
        group_key = get_group_key(sample)
        split = domain_split_map[group_key]
        if args.only_splits is not None and split not in args.only_splits:
            continue
        if group_key not in summary["groups_by_dataset_split"][dataset][split]:
            summary["groups_by_dataset_split"][dataset][split].append(group_key)
        npz_path = Path(sample["npz_path"])

        try:
            data = np.load(npz_path, allow_pickle=False)
            radar = data["radar"]
            ref_time = data["time"]
            heart_rate = data["heart_rate"]
        except Exception as exc:  # noqa: BLE001
            print_progress(sample_idx, len(samples), f"skip {sample_key}: {exc}")
            continue
        if dataset == "PhysDrive" and (
            radar.shape[0] != len(ref_time)
            or radar.shape[0] != len(heart_rate)
        ):
            raise ValueError(f"{sample_key}: 逐帧雷达、时间与心率长度不一致；请重新导出")
        frame_time = dataset_frame_times(dataset, ref_time, int(radar.shape[0]))
        frame_labels = {
            "heart_rate": interpolate_label_to_frames(heart_rate, ref_time, frame_time),
        }

        native_window_size, native_stride = native_window_geometry(
            dataset, args.window_size, args.stride
        )
        starts = iter_window_starts(int(radar.shape[0]), native_window_size, native_stride)
        if not starts:
            summary["windows_skipped_short"] += 1
            print_progress(sample_idx, len(samples), f"short sample {sample_key}")
            continue
        if args.max_windows_per_sample is not None:
            starts = starts[: args.max_windows_per_sample]

        sample_windows = 0
        dataset_summary = summary["by_dataset"].setdefault(
            dataset,
            {"samples_used": 0, "windows_written": 0, "windows_skipped_label": 0},
        )

        for window_idx, start in enumerate(starts):
            end = start + native_window_size
            labels: Dict[str, float] = {}
            coverage: Dict[str, float] = {}
            keep = True
            for target_name in ("heart_rate",):
                window_values = frame_labels[target_name][start:end]
                coverage[target_name] = label_coverage(window_values)
                labels[target_name] = reduce_label(window_values, args.label_reduction)
            for target_name in required:
                if coverage[target_name] < args.min_label_coverage:
                    keep = False
                if not np.isfinite(labels[target_name]):
                    keep = False
            if not keep:
                summary["windows_skipped_label"] += 1
                dataset_summary["windows_skipped_label"] += 1
                continue

            radar_window = radar_window_at_target_rate(
                radar[start:end], dataset, args.window_size
            )
            feature_bundle = radar_to_feature_bundle(
                radar_window, args.representation,
                hr_band_hz=SOURCE_HR_BAND_HZ[dataset],
                vmd_init_profile=SOURCE_VMD_INIT_PROFILE[dataset],
                include_rda=not args.drop_rda,
            )
            feature_bundle["x"] = normalize_features(feature_bundle["x"], args.normalize)
            if "x_time" in feature_bundle:
                feature_bundle["x_time"] = normalize_features(feature_bundle["x_time"], args.normalize)
            if "x_freq" in feature_bundle:
                feature_bundle["x_freq"] = normalize_features(feature_bundle["x_freq"], args.normalize)
            if "x_rda" in feature_bundle:
                if args.drop_rda:
                    feature_bundle.pop("x_rda", None)
                else:
                    feature_bundle["x_rda"] = normalize_features(
                        feature_bundle["x_rda"], args.normalize
                    )
            x = np.asarray(feature_bundle["x"], dtype=np.float32)
            x_freq = feature_bundle.get("x_freq")
            x_rda = feature_bundle.get("x_rda")
            feature_meta = feature_bundle.get("meta", {})
            out_name = f"{safe_name(dataset)}__{safe_name(sample_tag)}__w{window_idx:05d}.npz"
            out_path = output_dir / "windows" / dataset / split / out_name
            meta = {
                "dataset": dataset,
                "sample_tag": sample_tag,
                "group_key": group_key,
                "source_npz_path": str(npz_path),
                "ftu_adc_frontend_fingerprint": (
                    ftu_frontend_fingerprint if dataset == "FTU" else None
                ),
                "ftu_alias_audit_sha256": (
                    ftu_alias_audit_sha256 if dataset == "FTU" else None
                ),
                "domain_split": split,
                "window_start": int(start),
                "window_end": int(end),
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "source_window_size": native_window_size,
                "source_stride": native_stride,
                "source_sampling_rate_hz": SOURCE_SAMPLING_RATE_HZ[dataset],
                "sampling_rate_hz": DEFAULT_SAMPLING_RATE_HZ,
                "window_start_time_seconds": float(frame_time[start]),
                "resampling": "polyphase_2_3" if dataset == "BGT60TR13C" else "none",
                "representation": args.representation,
                "normalize": args.normalize,
                "label_reduction": args.label_reduction,
                "label_coverage": coverage,
                "feature_meta": feature_meta,
            }
            save_window(
                path=out_path,
                feature_bundle=feature_bundle,
                labels=labels,
                label_series={
                    "heart_rate": label_window_at_target_rate(
                        frame_labels["heart_rate"][start:end], dataset, args.window_size
                    ),
                },
                meta=meta,
            )
            manifest_rows.append(
                {
                    "window_path": str(out_path),
                    "dataset": dataset,
                    "sample_tag": sample_tag,
                    "group_key": group_key,
                    "domain_split": split,
                    "split": split,
                    "window_start": int(start),
                    "window_end": int(end),
                    "x_shape": str(tuple(x.shape)),
                    "x_freq_shape": str(tuple(np.asarray(x_freq).shape)) if x_freq is not None else "",
                    "x_rda_shape": str(tuple(np.asarray(x_rda).shape)) if x_rda is not None else "",
                    "label_heart_rate": labels["heart_rate"],
                    "coverage_heart_rate": coverage["heart_rate"],
                }
            )
            sample_windows += 1
            summary["windows_written"] += 1
            summary["by_split"][split] += 1
            dataset_summary["windows_written"] += 1

        if sample_windows > 0:
            summary["samples_used"] += 1
            dataset_summary["samples_used"] += 1
        print_progress(
            sample_idx,
            len(samples),
            f"{sample_key} -> {sample_windows} window(s)",
        )

    print_stage("Stage 3/4: write manifest and config")
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    write_json(
        output_dir / "build_config.json",
        {
            "exports_dir": str(exports_dir),
            "output_dir": str(output_dir),
            "datasets": list(args.datasets),
            "diagnostic_only": bool(getattr(args, "diagnostic_only", False)),
            "only_splits": list(args.only_splits) if args.only_splits is not None else None,
            "source_gate_pass": False,
            "drop_rda": bool(args.drop_rda),
            "ftu_adc_frontend_fingerprint": ftu_frontend_fingerprint,
            "ftu_alias_audit_sha256": ftu_alias_audit_sha256,
            "window_size": args.window_size,
            "stride": args.stride,
            "representation": args.representation,
            "target": args.target,
            "required_targets": list(required),
            "label_reduction": args.label_reduction,
            "min_label_coverage": args.min_label_coverage,
            "normalize": args.normalize,
            "edacm_top_bins": DEFAULT_EDACM_TOP_BINS,
            "edacm_candidate_multiplier": DEFAULT_EDACM_CANDIDATE_MULTIPLIER,
            "target_selection_method": "rda_vital_sign_stability_score",
            "target_selection_score_weights": {
                "energy": 0.35,
                "phase_stability": 0.25,
                "hr_band_peak": 0.25,
                "spatial_consistency": 0.15,
            },
            "hr_band_hz_by_dataset": {
                dataset: list(SOURCE_HR_BAND_HZ[dataset]) for dataset in args.datasets
            },
            "vmd_init_profile_by_dataset": {
                dataset: SOURCE_VMD_INIT_PROFILE[dataset] for dataset in args.datasets
            },
            "rda_stability_segments": DEFAULT_RDA_STABILITY_SEGMENTS,
            "vmd_k": DEFAULT_VMD_K,
            "decomposition_algorithm_version": DECOMPOSITION_ALGORITHM_VERSION,
            "vmd_alpha": DEFAULT_VMD_ALPHA,
            "vmd_max_iter": DEFAULT_VMD_MAX_ITER,
            "vmd_tol": DEFAULT_VMD_TOL,
            "resp_band_hz": list(DEFAULT_RESP_BAND_HZ),
            "sampling_rate_hz": DEFAULT_SAMPLING_RATE_HZ,
            "source_sampling_rate_hz_by_dataset": SOURCE_SAMPLING_RATE_HZ,
            "bgt60_resampling": "window_polyphase_2_3_complex_rda",
            "x_rda_representation": DEFAULT_RDA_REPRESENTATION,
            "output_arrays": (["x", "x_time", "x_freq", "freq_hz"]
                              if args.drop_rda else ["x", "x_time", "x_freq", "x_rda", "freq_hz"]),
            "method_pipeline": list(METHOD_PIPELINE),
            "innovation_modules": list(INNOVATION_MODULES),
            "domain_adaptation_method": DOMAIN_ADAPTATION_METHOD,
            "split_mode": args.split_mode,
            "fixed_participant_splits": {
                dataset: {split: list(participants)
                          for split, participants in DEFAULT_PARTICIPANT_SPLITS[dataset].items()}
                for dataset in args.datasets if dataset in DEFAULT_PARTICIPANT_SPLITS
            },
            "grouping": {
                "FTU": "participant_id",
                "BGT60TR13C": "participant_id",
                "PhysDrive": "session_id",
            },
            "ftu_special_case_policy": {
                "mode": "fixed_ftu_participant_protocol_v2",
                "special_participants": list(FTU_SPECIAL_PARTICIPANTS),
                "special_participants_note": {
                    "2": "experienced meditator",
                    "5": "asthma",
                    "6": "asthma",
                },
                "elevated_hr_participants": list(FTU_ELEVATED_HR_PARTICIPANTS),
                "rule": "Use the frozen FTU participant assignment above; keep all records of a participant in the same split.",
            },
            "bgt_long_split": DEFAULT_BGT_LONG_SPLIT,
            "bgt_long_participants": sorted(bgt_long_participants),
            "split_ratios": list(split_ratios),
            "seed": args.seed,
            "max_windows_per_sample": args.max_windows_per_sample,
        },
    )

    print_stage("Stage 4/4: write summary")
    for dataset, splits in summary["groups_by_dataset_split"].items():
        for split in splits:
            summary["groups_by_dataset_split"][dataset][split] = sorted(splits[split])
            groups = summary["groups_by_dataset_split"][dataset][split]
            labels = [group_label_means[g] for g in groups if g in group_label_means]
            if labels:
                summary["split_label_means"][dataset][split] = float(np.mean(labels))
            for group in groups:
                if group in group_label_means:
                    summary["group_label_means"][group] = group_label_means[group]
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[builder] output_dir: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    build_training_dataset(args)


if __name__ == "__main__":
    main()
