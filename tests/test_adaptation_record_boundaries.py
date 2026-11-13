"""Unlabeled adaptation must smooth within a recording, never across repeats."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.adapt_source_free import generate_temporal_pseudo_labels
from src.training.datasets import LabelStats, RadarUnlabeledWindowDataset


class FirstRadarValue(nn.Module):
    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        return x_time[:, 0, 0]


class AdaptationRecordBoundaryTest(unittest.TestCase):
    def test_interleaved_repeats_are_smoothed_separately_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for window_index, (a, b) in enumerate(zip((60, 70, 80), (120, 130, 140))):
                for record, value in (("record_a", a), ("record_b", b)):
                    relative = Path("windows") / f"{record}_{window_index}.npz"
                    (root / relative).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(root / relative,
                                        x_time=np.full((1, 256), value, dtype=np.float32),
                                        x_freq=np.zeros((1, 129), dtype=np.float32),
                                        label_heart_rate=np.asarray(999.0, dtype=np.float32))
                    rows.append({"window_path": relative.as_posix(), "dataset": "BGT60TR13C",
                                 "group_key": "BGT60TR13C/participant/1", "sample_tag": record,
                                 "window_start": str(window_index * 128), "window_end": str(window_index * 128 + 256),
                                 "split": "train", "label_heart_rate": "999"})
            with (root / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            dataset = RadarUnlabeledWindowDataset(root, "train", {"BGT60TR13C"})
            self.assertTrue(all("label_heart_rate" not in row for row in dataset.rows))
            pseudo, weights, summary = generate_temporal_pseudo_labels(
                FirstRadarValue(), dataset, DataLoader(dataset, batch_size=4),
                LabelStats(mean=0.0, std=1.0), torch.device("cpu"),
                temporal_window=3, confidence_temperature=3.0, min_weight=0.05)
            self.assertEqual(summary["n_records"], 2)
            self.assertEqual(summary["smoothing_unit"], "dataset/group_key/sample_tag physical record")
            np.testing.assert_allclose(pseudo, [65, 125, 70, 130, 75, 135])
            self.assertTrue(np.all(np.isfinite(weights)))
            self.assertTrue(np.all((weights >= 0.05) & (weights <= 1.0)))

    def test_even_temporal_window_is_rejected(self) -> None:
        class Empty:
            rows = []

            def __len__(self):
                return 0

        with self.assertRaisesRegex(ValueError, "positive odd"):
            generate_temporal_pseudo_labels(
                FirstRadarValue(), Empty(), [], LabelStats(0.0, 1.0),
                torch.device("cpu"), temporal_window=4,
                confidence_temperature=3.0, min_weight=0.05)


if __name__ == "__main__":
    unittest.main()
