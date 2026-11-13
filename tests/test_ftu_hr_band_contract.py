"""The FTU high-HR band reaches both EDACM target scoring and VMD metadata."""

import unittest

import numpy as np

from src.data.build_training_dataset import SOURCE_HR_BAND_HZ
from src.features.hr_adavmd import hr_adavmd_initial_omega
from src.features.representations import radar_to_feature_bundle


class FTUHRBandContractTest(unittest.TestCase):
    def test_training_band_covers_elevated_reference_without_changing_vmd_init(self):
        self.assertGreaterEqual(SOURCE_HR_BAND_HZ["FTU"][1], 161 / 60)
        time = np.arange(256) / 20.0
        phase = 0.2 * np.sin(2 * np.pi * 2.7 * time)
        radar = np.exp(1j * phase).astype(np.complex64)[:, None, None, None]

        bundle = radar_to_feature_bundle(
            radar, "proposed", hr_band_hz=SOURCE_HR_BAND_HZ["FTU"])

        self.assertEqual(bundle["meta"]["hr_band_hz"], [0.75, 3.5])
        self.assertEqual(bundle["meta"]["vmd_init_profile"], "legacy")
        self.assertEqual(bundle["x_time"].shape, (7, 256))
        self.assertTrue(np.isfinite(bundle["x_time"]).all())
        self.assertAlmostEqual(bundle["meta"]["hr_adavmd_init_center_hz"][-1], 2.4, places=4)

    def test_expanded_initialization_is_explicit(self):
        legacy = hr_adavmd_initial_omega(7, init_profile="legacy")
        expanded = hr_adavmd_initial_omega(7, init_profile="expanded_3p5")
        self.assertAlmostEqual(float(legacy[-1] * 20), 2.4, places=4)
        self.assertAlmostEqual(float(expanded[-1] * 20), 3.2, places=4)


if __name__ == "__main__":
    unittest.main()
