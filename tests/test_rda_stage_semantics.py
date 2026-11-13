"""Physical sanity checks for RDA stages that act on slow time and range."""

import unittest

import numpy as np

from src.radar.clutter import clutter_mti, clutter_pca, clutter_temporal_highpass
from src.radar.localization import (
    localization_hr_band,
    localization_spatial_consistency,
    localization_temporal_consistency,
)
from src.radar.range_selection import range_hr_band, range_hr_band_coherence
from src.radar.hr_phase_score import hr_phase_power


class RDAStageSemanticsTest(unittest.TestCase):
    def test_temporal_highpass_cutoff_is_in_hz(self):
        fs = 20.0
        t = np.arange(8192) / fs
        low = np.sin(2 * np.pi * 0.02 * t)
        heart = np.sin(2 * np.pi * 1.5 * t)
        cube = (low + heart).astype(np.complex64)[:, None, None, None]
        out = clutter_temporal_highpass(cube, cutoff=0.1, fs=fs)[:, 0, 0, 0].real
        t, low, heart, out = (a[2048:] for a in (t, low, heart, out))
        def gain(reference):
            return 2 * abs(np.dot(out, reference)) / len(reference)
        self.assertLess(gain(low), 0.3)
        self.assertGreater(gain(heart), 0.9)
        with self.assertRaises(ValueError):
            clutter_temporal_highpass(cube, cutoff=0.0, fs=fs)
        with self.assertRaises(ValueError):
            clutter_temporal_highpass(cube, cutoff=fs / 2, fs=fs)

    def test_mti_has_no_first_frame_static_impulse(self):
        stationary = np.full((16, 1, 2, 3), 5 + 2j, dtype=np.complex64)
        np.testing.assert_array_equal(clutter_mti(stationary), np.zeros_like(stationary))
        np.testing.assert_array_equal(clutter_mti(stationary[:1]), np.zeros_like(stationary[:1]))

    def test_pca_removes_common_motion_and_preserves_unique_heartbeat(self):
        frames = 128
        time = np.arange(frames) / 20.0
        common_motion = 10 * np.sin(2 * np.pi * 0.15 * time)
        heartbeat = np.sin(2 * np.pi * 1.5 * time)
        cube = np.broadcast_to(common_motion[:, None, None, None],
                               (frames, 1, 2, 4)).astype(np.complex64).copy()
        cube[:, 0, 1, 3] += heartbeat

        cleaned = clutter_pca(cube)

        self.assertEqual(cleaned.shape, cube.shape)
        self.assertLess(np.std(cleaned[:, 0, 0, 0].real),
                        0.1 * np.std(cube[:, 0, 0, 0].real))
        self.assertGreater(np.corrcoef(cleaned[:, 0, 1, 3].real, heartbeat)[0, 1],
                           0.99)

    def test_pca_treats_phase_shifted_common_motion_as_one_complex_mode(self):
        frames = 256
        time = np.arange(frames) / 20.0
        common = 10 * np.sin(2 * np.pi * 0.17 * time)
        heart = np.sin(2 * np.pi * 1.5 * time)
        spatial = np.exp(1j * np.linspace(0, 2.4, 8)).reshape(1, 1, 2, 4)
        cube = (common[:, None, None, None] * spatial + (5 + 2j)).astype(np.complex64)
        cube[:, 0, 1, 3] += heart

        cleaned = clutter_pca(cube, n_components=1)

        self.assertLess(np.std(cleaned[:, 0, 0, 0]), 0.2)
        self.assertGreater(np.corrcoef(cleaned[:, 0, 1, 3].real, heart)[0, 1], 0.98)
        self.assertLess(abs(cleaned[:, 0, 0, 0].mean() - cube[:, 0, 0, 0].mean()), 1e-4)

    def test_consistency_localizers_keep_off_center_range_target(self):
        cube = np.zeros((32, 1, 4, 12), dtype=np.complex64)
        cube[:, 0, 2, 9] = 10
        cube[:, 0, 0, 2] = 2

        for localize in (localization_spatial_consistency,
                         localization_temporal_consistency):
            with self.subTest(method=localize.__name__):
                focused = localize(cube)
                self.assertEqual(focused.shape, cube.shape)
                self.assertGreater(np.abs(focused[:, 0, 2, 9]).mean(), 9)
                self.assertLess(np.abs(focused[:, 0, 0, 2]).mean(), 0.1)

    def test_hr_band_methods_include_elevated_heart_rate(self):
        frames = 200
        time = np.arange(frames) / 20.0
        cube = np.zeros((frames, 1, 2, 12), dtype=np.complex64)
        cube[:, 0, 0, 2] = 10 * np.exp(0.3j * np.sin(2 * np.pi * 0.4 * time))
        cube[:, 0, 1, 9] = 2 * np.exp(0.3j * np.sin(2 * np.pi * 2.7 * time))

        for select in (localization_hr_band, range_hr_band, range_hr_band_coherence):
            with self.subTest(method=select.__name__):
                focused = select(cube, fs=20.0)
                self.assertGreater(np.abs(focused[:, 0, 1, 9]).mean(), 1.9)
                self.assertLess(np.abs(focused[:, 0, 0, 2]).mean(), 0.1)

    def test_hr_phase_score_uses_exact_conjugate_increment(self):
        fs = 20.0
        t = np.arange(200) / fs
        phase = 1.7 * np.sin(2 * np.pi * 2.0 * t)
        cube = np.exp(1j * phase).astype(np.complex64)[:, None, None, None]
        observed = hr_phase_power(cube, fs, (.75, 3.5))[0, 0, 0]
        reference = np.abs(np.fft.rfft(
            (phase - phase.mean()) * np.hanning(len(t)))) ** 2
        frequencies = np.fft.rfftfreq(len(t), d=1 / fs)
        expected = reference[(frequencies >= .75) & (frequencies <= 3.5)].sum()
        self.assertAlmostEqual(float(observed / expected), 1.0, places=4)
        with self.assertRaises(ValueError):
            hr_phase_power(cube, fs, (.75, fs / 2))


if __name__ == "__main__":
    unittest.main()
