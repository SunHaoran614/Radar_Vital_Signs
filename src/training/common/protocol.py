"""Dataset-role assertions for cross-domain experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def require_single_source_dataset(run_config: dict, expected: str) -> None:
    actual = run_config.get("datasets")
    if actual != [expected]:
        raise ValueError(f"Expected source trained only on {expected}, got {actual!r}")


def require_transfer_feature_contract(run_config: dict, target_export_dir: str | Path) -> dict:
    """Reject silent mixing of unversioned source and target feature exports.

    One joint export is accepted when its manifest still matches the source
    checkpoint. Separate exports require an explicit identical contract ID.
    The contract must cover radar preprocessing and channel/axis semantics;
    callers must not infer compatibility from tensor shape alone.
    """
    provenance = run_config.get("provenance") or {}
    source_dir_raw = provenance.get("dataset_dir")
    stored_digest = provenance.get("dataset_manifest_sha256")
    if not source_dir_raw or not stored_digest:
        raise ValueError("Source checkpoint lacks versioned export provenance")
    source_dir = Path(source_dir_raw)
    target_dir = Path(target_export_dir)
    source_manifest = source_dir / "manifest.csv"
    target_manifest = target_dir / "manifest.csv"
    if not source_manifest.is_file() or not target_manifest.is_file():
        raise ValueError("Source or target export manifest is missing")
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    target_digest = hashlib.sha256(target_manifest.read_bytes()).hexdigest()
    if source_digest != stored_digest:
        raise ValueError("Source export manifest changed since checkpoint training")
    source_cfg_path = source_dir / "build_config.json"
    target_cfg_path = target_dir / "build_config.json"
    if not source_cfg_path.is_file() or not target_cfg_path.is_file():
        raise ValueError("Source or target build_config.json is missing")
    source_cfg = json.loads(source_cfg_path.read_text(encoding="utf-8"))
    target_cfg = json.loads(target_cfg_path.read_text(encoding="utf-8"))
    source_rep = source_cfg.get("representation")
    target_rep = target_cfg.get("representation")
    if not source_rep or source_rep != target_rep:
        raise ValueError("Source and target representation versions differ")
    source_contract = source_cfg.get("transfer_feature_contract_id")
    target_contract = target_cfg.get("transfer_feature_contract_id")
    if source_digest != target_digest and (
        not source_contract or source_contract != target_contract
    ):
        raise ValueError("Separate source/target exports lack a matching transfer feature contract ID")
    return {"source_manifest_sha256": source_digest,
            "target_manifest_sha256": target_digest,
            "representation": source_rep,
            "transfer_feature_contract_id": source_contract,
            "same_manifest": source_digest == target_digest}
