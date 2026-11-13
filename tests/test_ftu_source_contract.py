"""The FTU source checkpoint must not include target labels."""

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from src.training.common.protocol import require_single_source_dataset, require_transfer_feature_contract


class FtuSourceContractTest(unittest.TestCase):
    def test_rejects_mixed_or_unknown_source(self) -> None:
        require_single_source_dataset({"datasets": ["FTU"]}, "FTU")
        for actual in (["FTU", "PhysDrive"], "all", None):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                require_single_source_dataset({"datasets": actual}, "FTU")

    def test_transfer_requires_matching_versioned_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory, "source"), Path(directory, "target")
            source.mkdir()
            target.mkdir()
            (source / "manifest.csv").write_text("source\n")
            (target / "manifest.csv").write_text("target\n")
            (source / "build_config.json").write_text(json.dumps({"representation": "proposed"}))
            (target / "build_config.json").write_text(json.dumps({"representation": "proposed"}))
            digest = hashlib.sha256((source / "manifest.csv").read_bytes()).hexdigest()
            run = {"provenance": {"dataset_dir": str(source),
                                  "dataset_manifest_sha256": digest}}
            same = require_transfer_feature_contract(run, source)
            self.assertTrue(same["same_manifest"])
            with self.assertRaisesRegex(ValueError, "matching transfer feature contract"):
                require_transfer_feature_contract(run, target)
            (source / "build_config.json").write_text(json.dumps({
                "representation": "proposed", "transfer_feature_contract_id": "front-v3"}))
            (target / "build_config.json").write_text(json.dumps({
                "representation": "proposed", "transfer_feature_contract_id": "front-v3"}))
            separate = require_transfer_feature_contract(run, target)
            self.assertFalse(separate["same_manifest"])
            (target / "build_config.json").write_text(json.dumps({
                "representation": "legacy", "transfer_feature_contract_id": "front-v3"}))
            with self.assertRaisesRegex(ValueError, "representation versions differ"):
                require_transfer_feature_contract(run, target)
            (source / "manifest.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                require_transfer_feature_contract(run, source)


if __name__ == "__main__":
    unittest.main()
