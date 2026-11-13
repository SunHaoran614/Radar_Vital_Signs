"""Pseudo-label adaptation with temporal confidence weighting for HR regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import MODEL_CHOICES, count_parameters, create_model
from src.training.common.checkpoints import load_run_config, resolve_checkpoint
from src.training.common.protocol import require_single_source_dataset, require_transfer_feature_contract
from src.training.datasets import LabelStats, RadarUnlabeledWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label temporal adaptation on unlabeled target-domain data.")
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--expected-source-dataset", choices=["FTU", "BGT60TR13C", "PhysDrive", "Nature"],
                        default="FTU", help="Assert the checkpoint used only this labeled source (default: FTU).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <source-model-dir>/best.pt.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument("--target-datasets", nargs="+", required=True,
                        choices=["FTU", "BGT60TR13C", "PhysDrive", "Nature"])
    parser.add_argument("--adapt-split", choices=["train"], default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--confidence-temperature", type=float, default=3.0)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float32, copy=True)
    radius = max(1, int(window) // 2)
    out = np.zeros_like(values, dtype=np.float32)
    for idx in range(values.size):
        lo = max(0, idx - radius)
        hi = min(values.size, idx + radius + 1)
        out[idx] = float(np.mean(values[lo:hi]))
    return out


def generate_temporal_pseudo_labels(
    model: nn.Module,
    dataset: RadarUnlabeledWindowDataset,
    loader: DataLoader,
    stats: LabelStats,
    device: torch.device,
    temporal_window: int,
    confidence_temperature: float,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if temporal_window < 1 or temporal_window % 2 != 1:
        raise ValueError("temporal_window must be a positive odd number")
    model.eval()
    pred_by_index: Dict[int, float] = {}
    with torch.inference_mode():
        for batch in loader:
            x_time = batch["x_time"].to(device, non_blocking=True)
            x_freq = batch["x_freq"].to(device, non_blocking=True)
            row_indices = batch["row_index"].detach().cpu().numpy().astype(int)
            pred_norm = model(x_time, x_freq).detach().cpu()
            pred_bpm = stats.denormalize(pred_norm).numpy().astype(np.float32)
            for row_index, pred in zip(row_indices, pred_bpm):
                if int(row_index) in pred_by_index:
                    raise ValueError(f"Duplicate adaptation row index: {row_index}")
                pred_by_index[int(row_index)] = float(pred)

    if set(pred_by_index) != set(range(len(dataset))):
        raise ValueError("Pseudo-label inference did not cover every target window exactly once")

    pseudo_bpm = np.zeros(len(dataset), dtype=np.float32)
    weights = np.ones(len(dataset), dtype=np.float32)
    groups: Dict[tuple[str, str, str], list[int]] = {}
    for idx, row in enumerate(dataset.rows):
        key = (row.get("dataset", ""), row.get("group_key", ""), row.get("sample_tag", ""))
        if not all(key):
            raise ValueError(f"Target window lacks record identity: {idx}")
        groups.setdefault(key, []).append(idx)

    record_summary: list[Dict[str, Any]] = []
    for (dataset_name, group_key, sample_tag), indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda idx: int(dataset.rows[idx].get("window_start") or 0))
        starts = [int(dataset.rows[idx]["window_start"]) for idx in indices]
        if len(starts) != len(set(starts)):
            raise ValueError(f"Duplicate target window start: {dataset_name}/{sample_tag}")
        raw = np.asarray([pred_by_index[idx] for idx in indices], dtype=np.float32)
        corrected = moving_average(raw, temporal_window)
        residual = np.abs(raw - corrected)
        confidence = np.exp(-residual / max(float(confidence_temperature), 1e-6)).astype(np.float32)
        confidence = np.clip(confidence, float(min_weight), 1.0)
        for local_idx, row_idx in enumerate(indices):
            pseudo_bpm[row_idx] = corrected[local_idx]
            weights[row_idx] = confidence[local_idx]
        record_summary.append({
            "dataset": dataset_name,
            "group_key": group_key,
            "sample_tag": sample_tag,
            "count": len(indices),
            "raw_pred_mean_bpm": float(np.mean(raw)),
            "pseudo_mean_bpm": float(np.mean(corrected)),
            "mean_weight": float(np.mean(confidence)),
            "mean_temporal_residual_bpm": float(np.mean(residual)),
        })

    pseudo_norm = ((pseudo_bpm - float(stats.mean)) / float(stats.std)).astype(np.float32)
    summary = {
        "pseudo_label_mean_bpm": float(np.mean(pseudo_bpm)),
        "pseudo_label_std_bpm": float(np.std(pseudo_bpm)),
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "smoothing_unit": "dataset/group_key/sample_tag physical record",
        "n_records": len(record_summary),
        "by_record": record_summary,
    }
    return pseudo_norm, weights.astype(np.float32), summary


class WeightedPseudoLabelDataset(Dataset):
    def __init__(self, base: RadarUnlabeledWindowDataset, pseudo_norm: np.ndarray, weights: np.ndarray) -> None:
        self.base = base
        self.pseudo_norm = pseudo_norm.astype(np.float32)
        self.weights = weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = dict(self.base[index])
        item["pseudo_y"] = torch.tensor(float(self.pseudo_norm[index]), dtype=torch.float32)
        item["pseudo_weight"] = torch.tensor(float(self.weights[index]), dtype=torch.float32)
        return item


def weighted_adaptation_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    criterion = nn.SmoothL1Loss(beta=0.5, reduction="none")
    total_loss = 0.0
    total_weight = 0.0
    seen = 0
    for batch in loader:
        x_time = batch["x_time"].to(device, non_blocking=True)
        x_freq = batch["x_freq"].to(device, non_blocking=True)
        pseudo_y = batch["pseudo_y"].to(device, non_blocking=True)
        weights = batch["pseudo_weight"].to(device, non_blocking=True)
        pred = model(x_time, x_freq)
        per_sample = criterion(pred, pseudo_y)
        loss = torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * int(pred.numel())
        total_weight += float(torch.sum(weights.detach()))
        seen += int(pred.numel())
    return {
        "loss": total_loss / seen if seen else math.nan,
        "mean_weight": total_weight / seen if seen else math.nan,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(args.source_model_dir)
    require_single_source_dataset(run_config, args.expected_source_dataset)
    transfer_contract = require_transfer_feature_contract(run_config, args.export_dir)
    checkpoint_path = resolve_checkpoint(args.source_model_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = args.model or run_config.get("model")
    if model_name not in MODEL_CHOICES:
        raise ValueError("Cannot infer model architecture. Pass --model or provide run_config.json.")

    stats = LabelStats(**checkpoint["label_stats"])
    model = create_model(model_name, checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_datasets = set(args.target_datasets)
    if args.expected_source_dataset in target_datasets:
        raise ValueError("Target adaptation datasets must exclude the labeled source dataset")
    adapt_ds = RadarUnlabeledWindowDataset(args.export_dir, args.adapt_split, datasets=target_datasets)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    adapt_loader = DataLoader(adapt_ds, shuffle=False, drop_last=False, **loader_kwargs)
    pseudo_norm, weights, pseudo_summary = generate_temporal_pseudo_labels(
        model=model,
        dataset=adapt_ds,
        loader=adapt_loader,
        stats=stats,
        device=device,
        temporal_window=args.temporal_window,
        confidence_temperature=args.confidence_temperature,
        min_weight=args.min_weight,
    )

    pseudo_ds = WeightedPseudoLabelDataset(adapt_ds, pseudo_norm=pseudo_norm, weights=weights)
    pseudo_loader = DataLoader(pseudo_ds, shuffle=True, drop_last=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = weighted_adaptation_epoch(model, pseudo_loader, optimizer, device)
        row = {"epoch": epoch, **metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} | pseudo-label loss {metrics['loss']:.4f} | mean weight {metrics['mean_weight']:.3f}",
            flush=True,
        )

    serializable_args = vars(args) | {
        "source_model_dir": str(args.source_model_dir),
        "checkpoint": str(checkpoint_path),
        "export_dir": str(args.export_dir),
        "output_dir": str(args.output_dir),
    }
    checkpoint_out = {
        "model_state": model.state_dict(),
        "config": checkpoint["config"],
        "label_stats": checkpoint["label_stats"],
        "source_checkpoint": str(checkpoint_path),
        "transfer_feature_contract": transfer_contract,
        "adaptation": {
            "method": "Pseudo-label Adaptation + Temporal Confidence Weighting",
            "target_datasets": sorted(target_datasets),
            "adapt_split": args.adapt_split,
            "pseudo_summary": pseudo_summary,
            "args": serializable_args,
        },
    }
    torch.save(checkpoint_out, args.output_dir / "adapted.pt")
    torch.save(checkpoint_out, args.output_dir / "best.pt")
    summary = {
        "model": model_name,
        "parameters": count_parameters(model),
        "source_checkpoint": str(checkpoint_path),
        "transfer_feature_contract": transfer_contract,
        "method": "Pseudo-label Adaptation + Temporal Confidence Weighting",
        "pseudo_summary": pseudo_summary,
        "target_hr_labels_used": False,
        "evaluation": "run evaluate_model.py separately after adaptation is frozen",
        "history": history,
    }
    run_config = {
        "model": model_name,
        "config": checkpoint["config"],
        "label_stats": checkpoint["label_stats"],
        "parameters": count_parameters(model),
        "adaptation_method": "Pseudo-label Adaptation + Temporal Confidence Weighting",
        "source_checkpoint": str(checkpoint_path),
        "target_datasets": sorted(target_datasets),
        "args": serializable_args,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
