"""Training loop for radar heart-rate regression models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import (
    MODEL_CHOICES,
    ModelConfig,
    config_to_dict,
    count_parameters,
    create_model_and_config,
)
from src.training.common.metrics import aggregate, append_prediction_rows, mae_bpm, rmse_bpm, tolerance_metrics
from src.training.datasets import LabelStats, build_datasets
from src.training.export_gate import validate_export_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train radar HR regression models on training_exports.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="cycleformer",
        help="Model architecture to train.",
    )
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/cycleformer"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FTU"],
        choices=["FTU", "BGT60TR13C", "PhysDrive"],
        help="Optional dataset subset, for example: --datasets FTU",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=48, help="TCN hidden channels.")
    parser.add_argument("--num-blocks", type=int, default=4, help="TCN temporal blocks.")
    parser.add_argument("--kernel-size", type=int, default=7, help="TCN convolution kernel size.")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--patch-len", type=int, default=16, help="PatchTST patch length.")
    parser.add_argument("--patch-stride", type=int, default=8, help="PatchTST patch stride.")
    # TSLANet-specific hyperparameters (kept separate from shared args so other
    # backbones are unaffected). None of these enable the frequency branch (R8
    # is deferred); use_frequency_domain stays False for tslanet.
    parser.add_argument("--emb-dim", type=int, default=64, help="TSLANet embedding dimension.")
    parser.add_argument("--tslanet-depth", type=int, default=3, help="TSLANet encoder depth.")
    parser.add_argument("--tslanet-patch-len", type=int, default=16, help="TSLANet patch length.")
    parser.add_argument("--tslanet-patch-stride", type=int, default=8, help="TSLANet patch stride.")
    parser.add_argument(
        "--tslanet-dropout",
        type=float,
        default=0.5,
        help="TSLANet dropout (official default, higher than the shared --dropout).",
    )
    parser.add_argument(
        "--use-asb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet adaptive spectral block.",
    )
    parser.add_argument(
        "--use-icb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet interactive convolution block.",
    )
    parser.add_argument(
        "--adaptive-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet adaptive high-frequency mask.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RevIN-style input normalization in TSLANet.",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["joint", "official"],
        default="joint",
        help="TSLANet channel handling: 'joint' mixes channels in the patch "
        "projection; 'official' is channel-independent (F1 experimental factor).",
    )
    parser.add_argument(
        "--heart-periods",
        type=int,
        nargs="+",
        default=None,
        help="CycleFormer heart-period candidates in samples.",
    )
    parser.add_argument(
        "--respiration-periods",
        type=int,
        nargs="+",
        default=None,
        help="CycleFormer respiration-period candidates in samples.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--decomp-method", choices=["moving_avg", "dft"], default="moving_avg")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--down-sampling-layers", type=int, default=3)
    parser.add_argument("--time-only", action="store_true", help="Disable the frequency branch.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug only: cap batches per epoch.")
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate sealed test windows only after the method is frozen.",
    )
    parser.add_argument(
        "--allow-legacy-representation", action="store_true",
        help="Explicitly allow old proposed windows without the corrected decomposition version.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_root: Path) -> str:
    """Return the current git commit hash of the canonical upstream repo.

    The run copy under /app/work/run/upstream is a `cp -r` of /app/work/upstream,
    so both may carry a .git. We try the run root first, then fall back to the
    canonical upstream path so provenance always reflects the true source version.
    Git may refuse to run as root on a repo owned by another user ("dubious
    ownership"); we pass `-c safe.directory=*` and also parse .git/HEAD directly.
    """
    for root in (repo_root, Path("/app/work/upstream")):
        try:
            out = subprocess.run(
                ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
        try:
            head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head[4:].strip()
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            return parts[0]
                refp = root / ".git" / Path(*ref.split("/"))
                if refp.exists():
                    return refp.read_text(encoding="utf-8").strip()
            elif head:
                return head
        except Exception:
            pass
    return "unknown"


def sha256_of(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def read_representation(export_dir: Path) -> str:
    try:
        cfg = json.loads(Path(export_dir, "build_config.json").read_text(encoding="utf-8"))
        return cfg.get("representation", "unknown")
    except Exception:
        return "unknown"


def mae_bpm(pred_norm: torch.Tensor, y_norm: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    pred = pred_norm * stats.std + stats.mean
    y = y_norm * stats.std + stats.mean
    return torch.mean(torch.abs(pred - y))


def participant_id(dataset: str, group_key: str, sample_tag: str) -> str:
    if "/participant/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if "/session/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if dataset in {"FTU", "BGT60TR13C"}:
        match = re.match(r"p0?(\d+)", sample_tag)
        if match:
            return match.group(1)
    if dataset == "PhysDrive":
        return sample_tag.split("_", 1)[0]
    return group_key or sample_tag or "unknown"


def aggregate_errors(rows: list[Dict[str, float | str]], key: str) -> Dict[str, Dict[str, float | int]]:
    buckets: Dict[str, list[Dict[str, float | str]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)

    metrics: Dict[str, Dict[str, float | int]] = {}
    for name, values in buckets.items():
        errors = [float(row["abs_error_bpm"]) for row in values]
        labels = [float(row["label_bpm"]) for row in values]
        preds = [float(row["pred_bpm"]) for row in values]
        metrics[name] = {
            "count": len(values),
            "mae_bpm": sum(errors) / len(errors),
            "within_3bpm_percent": 100.0 * sum(error <= 3.0 for error in errors) / len(errors),
            "within_5bpm_percent": 100.0 * sum(error <= 5.0 for error in errors) / len(errors),
            "label_mean_bpm": sum(labels) / len(labels),
            "pred_mean_bpm": sum(preds) / len(preds),
        }
    return metrics


def tolerance_metrics(rows: list[Dict[str, float | str]]) -> Dict[str, float]:
    if not rows:
        return {"within_3bpm_percent": math.nan, "within_5bpm_percent": math.nan}
    errors = [float(row["abs_error_bpm"]) for row in rows]
    return {
        "within_3bpm_percent": 100.0 * sum(error <= 3.0 for error in errors) / len(errors),
        "within_5bpm_percent": 100.0 * sum(error <= 5.0 for error in errors) / len(errors),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: LabelStats,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
    collect_participant_metrics: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    seen = 0
    detail_rows: list[Dict[str, float | str]] = []

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x_time = batch["x_time"].to(device, non_blocking=True)
        x_freq = batch["x_freq"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(x_time, x_freq)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(y.numel())
        total_loss += float(loss.detach()) * batch_size
        total_mae += float(mae_bpm(pred.detach(), y.detach(), stats)) * batch_size
        total_rmse += float(rmse_bpm(pred.detach(), y.detach(), stats)) * batch_size
        seen += batch_size
        if collect_participant_metrics:
            append_prediction_rows(detail_rows, batch, pred, stats)

    if seen == 0:
        return {"loss": math.nan, "mae_bpm": math.nan, "rmse_bpm": math.nan}
    metrics: Dict[str, float | Dict[str, Dict[str, float | int]]] = {
        "loss": total_loss / seen,
        "mae_bpm": total_mae / seen,
        "rmse_bpm": total_rmse / seen,
    }
    if collect_participant_metrics:
        metrics.update(tolerance_metrics(detail_rows))
        metrics["by_participant"] = aggregate(detail_rows, "participant_id")
        metrics["by_group"] = aggregate(detail_rows, "group_key")
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: ModelConfig,
    stats: LabelStats,
    metrics: Dict[str, Any],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_to_dict(cfg),
            "label_stats": asdict(stats),
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_export_version(args.export_dir, args.allow_legacy_representation,
                            datasets=set(args.datasets))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = set(args.datasets) if args.datasets else None
    train_ds, val_ds, test_ds, stats = build_datasets(
        args.export_dir, datasets=dataset_filter, include_test=args.evaluate_test)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = (
        DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
        if test_ds is not None else None
    )

    model, cfg = create_model_and_config(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.SmoothL1Loss(beta=0.5)

    metadata = {
        "config": config_to_dict(cfg),
        "model": args.model,
        "label_stats": asdict(stats),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds) if test_ds is not None else None,
        "datasets": sorted(dataset_filter) if dataset_filter else "all",
        "parameters": count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args) | {"export_dir": str(args.export_dir), "output_dir": str(args.output_dir)},
    }
    metadata["provenance"] = {
        "git_commit": get_git_commit(ROOT),
        "representation": read_representation(args.export_dir),
        "dataset_dir": str(args.export_dir),
        "dataset_manifest_sha256": sha256_of(args.export_dir / "manifest.csv"),
        "seed": args.seed,
        "model": args.model,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_val = float("inf")
    stale_epochs = 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, stats, optimizer=optimizer, limit_batches=args.limit_batches
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, stats, limit_batches=args.limit_batches)
        scheduler.step(val_metrics["mae_bpm"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"epoch {epoch:03d} | train MAE {train_metrics['mae_bpm']:.3f} RMSE {train_metrics['rmse_bpm']:.3f} BPM "
            f"| val MAE {val_metrics['mae_bpm']:.3f} RMSE {val_metrics['rmse_bpm']:.3f} BPM | lr {row['lr']:.2e}",
            flush=True,
        )

        if val_metrics["mae_bpm"] < best_val:
            best_val = val_metrics["mae_bpm"]
            stale_epochs = 0
            save_checkpoint(args.output_dir / "best.pt", model, cfg, stats, val_metrics, epoch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = None
    if test_loader is not None:
        test_metrics = run_epoch(
            model, test_loader, criterion, device, stats,
            limit_batches=args.limit_batches, collect_participant_metrics=True,
        )
    save_checkpoint(
        args.output_dir / "final.pt", model, cfg, stats,
        test_metrics if test_metrics is not None else checkpoint["metrics"],
        int(checkpoint["epoch"]),
    )
    summary = {"best_val_mae_bpm": best_val, "test": test_metrics, "elapsed_sec": time.time() - start}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Per the experiment results-spec, also emit a standalone eval artifact.
    if test_metrics is not None:
        (args.output_dir / "eval_test.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)



if __name__ == "__main__":
    main()
