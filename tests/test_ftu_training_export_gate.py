"""FTU source eligibility cannot be bypassed by changing representation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.features.hr_adavmd import DECOMPOSITION_ALGORITHM_VERSION
from src.training.export_gate import validate_export_version


class FTUTrainingExportGateTest(unittest.TestCase):
    def check_config(self, config, *, allow_legacy=False, datasets=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "build_config.json").write_text(json.dumps(config), encoding="utf-8")
            validate_export_version(path, allow_legacy=allow_legacy, datasets=datasets)

    def test_gate_applies_to_every_ftu_representation_even_with_legacy_override(self):
        for representation in ("proposed", "target_edacm_vmd", "target_edacm",
                               "log_magnitude", "magnitude", "real_imag"):
            with self.subTest(representation=representation):
                config = {"datasets": ["FTU"], "representation": representation,
                          "source_gate_pass": False,
                          "decomposition_algorithm_version": DECOMPOSITION_ALGORITHM_VERSION}
                with self.assertRaisesRegex(ValueError, "source-input gate"):
                    self.check_config(config, allow_legacy=True)

    def test_diagnostic_export_is_blocked_even_if_gate_flag_is_true(self):
        config = {"datasets": ["FTU"], "representation": "target_edacm",
                  "source_gate_pass": True, "diagnostic_only": True}
        with self.assertRaisesRegex(ValueError, "diagnostic_only"):
            self.check_config(config)

    def test_vmd_version_is_checked_after_source_gate(self):
        config = {"datasets": ["FTU"], "representation": "proposed",
                  "source_gate_pass": True, "decomposition_algorithm_version": "old"}
        with self.assertRaisesRegex(ValueError, "decomposition version"):
            self.check_config(config)
        self.check_config(config, allow_legacy=True)

    def test_non_vmd_passes_only_after_source_gate(self):
        config = {"datasets": ["FTU"], "representation": "target_edacm",
                  "source_gate_pass": True}
        self.check_config(config)


if __name__ == "__main__":
    unittest.main()
