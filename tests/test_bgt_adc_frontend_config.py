"""BGT ADC candidate must preserve a known slow-time complex range target."""

import json
import csv
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.data.loaders import bgt60_loader

from src.data.loaders.export_all_datasets import convert_adc_cube_to_rda
from src.data.loaders import export_all_datasets as export
from src.radar.bgt_adc_config import BGTADCConfig


class BGTADCFrontendTest(unittest.TestCase):
    def test_candidate_selects_motion_target_and_keeps_zero_doppler(self) -> None:
        frames = 32
        native_sample = np.arange(512)
        target = np.cos(2 * np.pi * 20 * native_sample / 512)
        amplitude = 40 + 15 * np.sin(2 * np.pi * np.arange(frames) / frames)
        raw = np.full((frames, 3, 16, 512), 2048, dtype=np.float32)
        raw += amplitude[:, None, None, None] * target[None, None, None, :]
        raw += 100 * np.arange(16, dtype=np.float32)[None, None, :, None]
        candidate = BGTADCConfig()
        self.assertEqual(candidate.range_search_bins, (8, 60))
        cube, bins, center = candidate.convert(raw)
        self.assertEqual(center, 20)
        self.assertIn(20, bins)
        self.assertGreater(float(np.linalg.norm(cube[:, 4, :, 20 - bins[0]])), 1.)

        legacy, old_bins, old_center = convert_adc_cube_to_rda(raw)
        self.assertEqual(old_center, 0)
        self.assertNotIn(20, old_bins)
        self.assertEqual(legacy.shape, (frames, 8, 16, 8))

    def test_candidate_fingerprint_changes_with_search_range(self) -> None:
        candidate = BGTADCConfig()
        file_config = json.loads(Path("configs/bgt_adc_coherent_motion_v1.json").read_text())
        self.assertEqual(BGTADCConfig.from_dict(file_config), candidate)
        altered = BGTADCConfig.from_dict({**candidate.to_dict(), "range_search_bins": [8, 46]})
        self.assertNotEqual(candidate.fingerprint, altered.fingerprint)

    def test_cli_passes_candidate_and_rejects_existing_output(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            output = root / "new"
            arguments = ["export_all_datasets", "--bgt60", "--dataset-root", str(root),
                         "--output-dir", str(output), "--bgt-skip-long",
                         "--bgt-frontend-config", "configs/bgt_adc_coherent_motion_v1.json"]
            called = {}
            def fake_export(**kwargs):
                called.update(kwargs)
                return (1, 0)
            with patch.object(export.sys, "argv", arguments), patch.object(export, "export_bgt60", fake_export):
                export.main()
            self.assertEqual(called["frontend_config"], BGTADCConfig())
            self.assertFalse(called["include_long"])
            existing = output / "BGT60TR13C"
            existing.mkdir()
            (existing / "manifest.csv").write_text("existing\n")
            with patch.object(export.sys, "argv", arguments):
                with self.assertRaises(FileExistsError):
                    export.main()
            self.assertEqual((existing / "manifest.csv").read_text(), "existing\n")

    def test_export_writes_candidate_fingerprint_and_physical_roi(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            raw = np.full((32, 3, 16, 512), 2048, dtype=np.float32)
            carrier = np.cos(2 * np.pi * 20 * np.arange(512) / 512)
            motion = 50 + 15 * np.sin(2 * np.pi * np.arange(32) / 32)
            raw += motion[:, None, None, None] * carrier[None, None, None, :]

            class FakeLoader:
                def __init__(self, _root):
                    pass

                def list_participants(self):
                    return [2]

                def list_available_measurements(self, _pid):
                    return {"short": ["0.6m"]}

                def load_radar_data(self, **_kwargs):
                    return raw

                def load_reference_data(self, **_kwargs):
                    return {"time": np.arange(1, 33),
                            "heart_rate": np.full(32, 72.),
                            "respiration_rate": np.full(32, 15.)}

            with patch.object(bgt60_loader, "BGT60TR13CDataLoader", FakeLoader):
                result = export.export_bgt60(root, root / "out", compress=False,
                                             include_long=False,
                                             frontend_config=BGTADCConfig())
            self.assertEqual(result, (1, 0))
            meta = json.loads((root / "out/BGT60TR13C/meta/p02_0_6m_short.json").read_text())
            self.assertEqual(meta["range_center_bin"], 20)
            self.assertIn(20, meta["range_selected_bins"])
            self.assertEqual(meta["bgt_adc_frontend_fingerprint"], BGTADCConfig().fingerprint)
            with np.load(root / "out/BGT60TR13C/samples/p02_0_6m_short.npz") as saved:
                self.assertEqual(saved["radar"].shape, (32, 8, 16, 8))

    def test_candidate_cannot_resume_unversioned_legacy_sample(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            exported = root / "out/BGT60TR13C"
            (exported / "samples").mkdir(parents=True)
            (exported / "meta").mkdir()
            sample = exported / "samples/p02_0_6m_short.npz"
            np.savez(sample, radar=np.zeros((1, 8, 16, 8), dtype=np.complex64))
            (exported / "meta/p02_0_6m_short.json").write_text(json.dumps({
                "range_selection_method": "global_energy_center_contiguous_window"}))
            with (exported / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_tag", "status", "npz_path"])
                writer.writeheader()
                writer.writerow({"sample_tag": "p02_0_6m_short", "status": "ok",
                                 "npz_path": str(sample)})

            class FakeLoader:
                def __init__(self, _root):
                    pass
                def list_participants(self):
                    return [2]
                def list_available_measurements(self, _pid):
                    return {"short": ["0.6m"]}

            with patch.object(bgt60_loader, "BGT60TR13CDataLoader", FakeLoader):
                with self.assertRaisesRegex(ValueError, "mixed BGT frontends"):
                    export.export_bgt60(root, root / "out", compress=False,
                                        include_long=False, resume=True,
                                        frontend_config=BGTADCConfig())


if __name__ == "__main__":
    unittest.main()
