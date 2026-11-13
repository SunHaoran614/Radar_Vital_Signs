"""Training export eligibility checks without a PyTorch dependency."""

from __future__ import annotations

import json
from pathlib import Path

from src.features.hr_adavmd import DECOMPOSITION_ALGORITHM_VERSION


def validate_export_version(
    export_dir: Path, allow_legacy: bool = False, datasets: set[str] | None = None,
) -> None:
    """Require an FTU source gate for every representation; version VMD separately."""
    config = json.loads((export_dir / "build_config.json").read_text(encoding="utf-8"))
    if config.get("diagnostic_only"):
        raise ValueError("This export is diagnostic_only; source learnability gate has not passed.")
    selected_datasets = set(config.get("datasets", [])) if datasets is None else datasets
    if "FTU" in selected_datasets and config.get("source_gate_pass") is not True:
        raise ValueError(
            "FTU export has no passed source-input gate; finish the train-only "
            "physical/representation checks before model training."
        )
    if config.get("representation") not in {"proposed", "target_edacm_vmd"}:
        return
    actual = config.get("decomposition_algorithm_version")
    if actual != DECOMPOSITION_ALGORITHM_VERSION and not allow_legacy:
        raise ValueError(
            f"Export uses decomposition version {actual!r}; expected "
            f"{DECOMPOSITION_ALGORITHM_VERSION!r}. Regenerate versioned windows, "
            "or pass --allow-legacy-representation for an intentional old baseline."
        )
