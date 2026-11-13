"""Native radar clock and 30-to-20 Hz window conversion checks."""

import unittest

import numpy as np

from src.data.build_training_dataset import (
    dataset_frame_times,
    label_window_at_target_rate,
    native_window_geometry,
    radar_window_at_target_rate,
)


class WindowRateAlignmentTest(unittest.TestCase):
    def test_ftu_reference_tail_does_not_stretch_radar_clock(self) -> None:
        ref_time = np.arange(65, dtype=np.float32)
        frame_time = dataset_frame_times("FTU", ref_time, 1200)
        self.assertAlmostEqual(float(frame_time[-1]), 59.95, places=4)

    def test_bgt_window_is_same_duration_as_twenty_hz_window(self) -> None:
        self.assertEqual(native_window_geometry("BGT60TR13C", 256, 128), (384, 192))
        native_time = dataset_frame_times("BGT60TR13C", np.arange(1, 601), 18000)
        self.assertAlmostEqual(float(native_time[-1]), 599.9667, places=3)
        tone = np.exp(2j * np.pi * 1.25 * np.arange(384) / 30.0).astype(np.complex64)
        radar = radar_window_at_target_rate(tone[:, None, None, None], "BGT60TR13C", 256)
        self.assertEqual(radar.shape, (256, 1, 1, 1))
        peak_hz = np.fft.rfftfreq(256, 1 / 20)[
            np.argmax(np.abs(np.fft.rfft(np.real(radar[:, 0, 0, 0]))))
        ]
        self.assertAlmostEqual(float(peak_hz), 1.25, delta=0.04)
        label = label_window_at_target_rate(np.arange(384) / 30.0, "BGT60TR13C", 256)
        self.assertEqual(label.shape, (256,))
        self.assertTrue(np.allclose(label, np.arange(256) / 20.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
