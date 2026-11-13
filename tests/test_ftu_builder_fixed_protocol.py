"""Keep future FTU window exports on the frozen source subject protocol."""

import unittest

from src.data.build_training_dataset import (
    build_domain_split_map,
    default_participant_split,
)


class FTUBuilderFixedProtocolTest(unittest.TestCase):
    def test_builder_default_keeps_source_subject_protocol(self):
        # Explicit protocol fixture: no private export or historical manifest.
        expected = {f"FTU/participant/{pid}": split
                    for split, people in {"train": (1, 2, 4, 7, 8, 9),
                                          "val": (5, 6), "test": (3, 10)}.items()
                    for pid in people}

        actual = default_participant_split("FTU", list(expected))

        self.assertEqual(actual, expected)
        self.assertEqual({key.rsplit("/", 1)[-1] for key, split in actual.items()
                          if split == "test"}, {"3", "10"})

    def test_stable_mode_cannot_reassign_frozen_test_subject(self):
        samples = [{"dataset": "FTU", "sample_tag": f"p{pid:02d}",
                    "source_row": {"participant_id": str(pid)}}
                   for pid in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
        assignment = build_domain_split_map(samples, (.7, .15, .15), 42,
                                            set(), "stable_grouped")
        self.assertEqual(assignment["FTU/participant/3"], "test")
        self.assertEqual(assignment["FTU/participant/10"], "test")
        self.assertEqual(assignment["FTU/participant/5"], "val")

    def test_partial_ftu_export_keeps_observed_participants_frozen(self):
        samples = [{"dataset": "FTU", "sample_tag": f"p{pid:02d}",
                    "source_row": {"participant_id": str(pid)}}
                   for pid in (1, 3, 5)]
        assignment = build_domain_split_map(samples, (.7, .15, .15), 42,
                                            set(), "balanced_grouped")
        self.assertEqual(assignment, {
            "FTU/participant/1": "train",
            "FTU/participant/3": "test",
            "FTU/participant/5": "val",
        })


if __name__ == "__main__":
    unittest.main()
