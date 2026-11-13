"""Verify that FTU training manifests count physical recordings once.

The 4TU export contains exact aliases across three scenario folders for each
participant's 80-cm front-facing repeat. Use the full-byte-verified mapping
from the raw audit as the authority; do not infer identity from names alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def alias_map(audit_path: Path) -> dict[str, str]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    groups = audit.get("full_byte_verification", [])
    if len(groups) != 40 or not all(group.get("all_byte_equal") for group in groups):
        raise ValueError("FTU alias audit needs 40 full-byte-verified groups")
    hr_sets = {frozenset(tags) for tags in audit.get("hr_sha256_groups", [])}
    mapping = {}
    for group in groups:
        tags = group["tags"]
        if len(tags) != 3 or frozenset(tags) not in hr_sets:
            raise ValueError("ADC aliases and Polar CSV aliases disagree")
        canonical = next((tag for tag in tags if "_Distance_80_cm_" in tag), None)
        if canonical is None:
            raise ValueError("No Distance 80 cm canonical tag in alias group")
        for tag in tags:
            if tag in mapping:
                raise ValueError("FTU tag occurs in multiple alias groups")
            mapping[tag] = canonical
    return mapping


def check_manifest(manifest: Path, audit: Path) -> dict:
    aliases = alias_map(audit)
    by_canonical: dict[str, set[str]] = defaultdict(set)
    split_by_canonical: dict[str, set[str]] = defaultdict(set)
    ftu_rows = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset", "FTU") != "FTU":
                continue
            ftu_rows += 1
            tag = row.get("sample_tag") or row.get("canonical_sample_tag")
            if not tag:
                raise ValueError("FTU manifest row has no sample tag")
            canonical = aliases.get(tag, tag)
            by_canonical[canonical].add(tag)
            split = row.get("split")
            if split:
                split_by_canonical[canonical].add(split)
    duplicate_groups = {canonical: sorted(tags) for canonical, tags in by_canonical.items()
                        if len(tags) > 1}
    split_conflicts = {canonical: sorted(splits) for canonical, splits in split_by_canonical.items()
                       if len(splits) > 1}
    return {"manifest": str(manifest), "ftu_window_rows": ftu_rows,
            "unique_physical_recordings": len(by_canonical),
            "duplicate_physical_groups": len(duplicate_groups),
            "split_conflicts": len(split_conflicts),
            "duplicate_examples": dict(list(duplicate_groups.items())[:5]),
            "split_conflict_examples": dict(list(split_conflicts.items())[:5]),
            "pass": not duplicate_groups and not split_conflicts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = check_manifest(a.manifest, a.audit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
