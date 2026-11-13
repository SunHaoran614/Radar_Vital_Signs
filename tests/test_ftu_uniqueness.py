"""FTU physical-record identity must survive scenario aliasing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.ftu_uniqueness import check_manifest


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "sample_tag", "split"))
        writer.writeheader()
        writer.writerows(rows)


def test_aliases_are_not_counted_as_independent_records(tmp_path: Path) -> None:
    tags = ["p01_Distance_80_cm_r1", "p01_Orientation_80_cm__Front_side_r1",
            "p01_Angle_0_deg__80_cm_r1"]
    audit = {"full_byte_verification": [], "hr_sha256_groups": []}
    for participant in range(1, 11):
        for repeat in range(1, 5):
            group = [f"p{participant:02d}_Distance_80_cm_r{repeat}",
                     f"p{participant:02d}_Orientation_80_cm__Front_side_r{repeat}",
                     f"p{participant:02d}_Angle_0_deg__80_cm_r{repeat}"]
            audit["full_byte_verification"].append({"tags": group, "all_byte_equal": True})
            audit["hr_sha256_groups"].append(group)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [{"dataset": "FTU", "sample_tag": tags[0], "split": "train"},
                         {"dataset": "FTU", "sample_tag": tags[1], "split": "val"}])
    report = check_manifest(manifest, audit_path)
    assert not report["pass"]
    assert report["duplicate_physical_groups"] == 1
    assert report["split_conflicts"] == 1
    write_csv(manifest, [{"dataset": "FTU", "sample_tag": tags[0], "split": "train"}])
    assert check_manifest(manifest, audit_path)["pass"]
