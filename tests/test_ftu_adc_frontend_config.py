import json
import csv

import numpy as np
import pytest

from src.data.loaders import export_all_datasets as export
from src.radar.ftu_adc_config import FTUADCConfig
from src.data.loaders.ftu_loader import FTUDataLoader


def test_config_is_stable_and_validates_dependent_stages():
    baseline = FTUADCConfig()
    assert FTUADCConfig.from_dict(json.loads(json.dumps(baseline.to_dict()))) == baseline
    assert FTUADCConfig.from_dict(baseline.to_dict()).fingerprint == baseline.fingerprint
    assert FTUADCConfig.coherent_motion_candidate().fingerprint != baseline.fingerprint
    with pytest.raises(ValueError, match="requires full_center_bins"):
        FTUADCConfig(range_selection_method="motion_zero_doppler")


def test_manifest_accepts_failed_then_successful_sample(tmp_path):
    path = tmp_path / "manifest.csv"
    export.write_manifest(path, [{"sample_tag": "failed", "error": "bad"},
                                 {"sample_tag": "ok", "radar_shape": "(1,8,16,8)"}])
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["error"] == "bad"
    assert rows[1]["radar_shape"] == "(1,8,16,8)"


def test_ftu_export_uses_and_records_adc_config(tmp_path, monkeypatch):
    for part in ("Radar data", "HR_Ref_Values", "BR_Ref_Values"):
        (tmp_path / part).mkdir()
    time = np.arange(4) / 20.0
    fast = np.exp(2j * np.pi * 8 * np.arange(32) / 32)
    radar = np.zeros((4, 4, 16, 32), dtype=np.complex64)
    for frame in range(4):
        radar[frame] = (1.0 + 0.2 * np.sin(2 * np.pi * frame / 4)) * fast

    class FakeLoader:
        def __init__(self, _root):
            pass

        def list_available_samples(self, **_kwargs):
            return [{"participant_id": 1, "scenario": "Distance", "distance": "40 cm", "repeat": 1}]

        def load_radar_data(self, **_kwargs):
            return radar

        def open_radar_frame_view(self, **_kwargs):
            return radar

        def load_reference_data(self, **_kwargs):
            return {"timestamps": time, "heart_rate": np.full(4, 70.0)}

    monkeypatch.setattr(export, "FTUDataLoader", FakeLoader)
    monkeypatch.setattr(export, "UNIFY_TO_RDA", True)
    for name, config in (("legacy", FTUADCConfig()),
                         ("candidate", FTUADCConfig.coherent_motion_candidate())):
        output = tmp_path / name
        assert export.export_ftu(tmp_path, output, compress=False, frontend_config=config) == (1, 0)
        tag = "p01_Distance_40_cm_r1"
        with np.load(output / "FTU/samples" / f"{tag}.npz") as sample:
            expected, _, _ = export.convert_adc_cube_to_rda(radar, **config.converter_kwargs())
            np.testing.assert_allclose(sample["radar"], expected)
        meta = json.loads((output / "FTU/meta" / f"{tag}.json").read_text())
        assert meta["ftu_adc_frontend_config"] == json.loads(json.dumps(config.to_dict()))
        assert meta["ftu_adc_frontend_fingerprint"] == config.fingerprint
        assert meta["range_selection_method"] == config.range_selection_method


def test_ftu_cli_passes_versioned_frontend_config(tmp_path, monkeypatch):
    config = FTUADCConfig.coherent_motion_candidate()
    config_path = tmp_path / "front_end.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    output = tmp_path / "export"
    called = {}

    def fake_export_ftu(**kwargs):
        called.update(kwargs)
        return 1, 0

    monkeypatch.setattr(export, "export_ftu", fake_export_ftu)
    monkeypatch.setattr(export.sys, "argv", [
        "export_all_datasets", "--ftu", "--dataset-root", str(tmp_path),
        "--output-dir", str(output), "--ftu-frontend-config", str(config_path),
        "--ftu-alias-audit", str(tmp_path / "aliases.json"),
    ])
    export.main()
    assert called["frontend_config"] == config
    assert json.loads((output / "export_summary.json").read_text())["FTU"] == {
        "success": 1, "failed": 0,
    }


def test_ftu_cli_rejects_existing_version_output(tmp_path, monkeypatch):
    config_path = tmp_path / "front_end.json"
    config_path.write_text(json.dumps(FTUADCConfig().to_dict()), encoding="utf-8")
    output = tmp_path / "export"
    existing = output / "FTU"
    existing.mkdir(parents=True)
    (existing / "manifest.csv").write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(export.sys, "argv", [
        "export_all_datasets", "--ftu", "--dataset-root", str(tmp_path),
        "--output-dir", str(output), "--ftu-frontend-config", str(config_path),
        "--ftu-alias-audit", str(tmp_path / "aliases.json"),
    ])
    with pytest.raises(FileExistsError, match="fresh output directory"):
        export.main()
    assert (existing / "manifest.csv").read_text() == "existing\n"


def test_ftu_frame_view_matches_eager_decode_and_both_converter_modes(tmp_path, monkeypatch):
    loader = FTUDataLoader.__new__(FTUDataLoader)
    loader.NUM_FRAMES = 4
    loader.NUM_ADC = 32
    loader.NUM_CHIRPS = 16
    loader.NUM_RX = 4
    monkeypatch.setattr(loader, "_validate_parameters", lambda *_args: None)
    monkeypatch.setattr(loader, "_build_radar_file_path", lambda *_args: tmp_path)
    words = loader.NUM_FRAMES * loader.NUM_ADC * loader.NUM_CHIRPS * loader.NUM_RX * 2
    rng = np.random.default_rng(20261008)
    raw = rng.integers(-1000, 1000, size=words, dtype=np.int16)
    raw.tofile(tmp_path / "data_Raw_0.bin")
    kwargs = dict(participant_id=1, scenario="Distance", distance="40 cm", repeat=1)
    eager = loader.load_radar_data(**kwargs)
    view = loader.open_radar_frame_view(**kwargs)
    assert view.shape == eager.shape and view.dtype == eager.dtype
    for frame in range(loader.NUM_FRAMES):
        np.testing.assert_array_equal(view[frame], eager[frame])
    for config in (FTUADCConfig(), FTUADCConfig.coherent_motion_candidate()):
        expected = config.convert(eager)
        actual = config.convert(view)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


def test_versioned_ftu_export_keeps_one_full_byte_verified_alias(tmp_path, monkeypatch):
    for part in ("Radar data", "HR_Ref_Values", "BR_Ref_Values"):
        (tmp_path / "4TU" / part).mkdir(parents=True)
    time = np.arange(4) / 20.0
    radar = np.zeros((4, 4, 16, 32), dtype=np.complex64)

    class FakeLoader:
        def __init__(self, root):
            assert root == str(tmp_path / "4TU")

        def list_available_samples(self, **_kwargs):
            return [
                {"participant_id": 1, "scenario": "Distance", "distance": "80 cm", "repeat": 1},
                {"participant_id": 1, "scenario": "Orientation", "distance": "80 cm- Front side", "repeat": 1},
                {"participant_id": 1, "scenario": "Angle", "distance": "0 deg- 80 cm", "repeat": 1},
            ]

        def open_radar_frame_view(self, **_kwargs):
            return radar

        def load_reference_data(self, **_kwargs):
            return {"timestamps": time, "heart_rate": np.full(4, 70.0)}

    monkeypatch.setattr(export, "FTUDataLoader", FakeLoader)
    monkeypatch.setattr(export, "UNIFY_TO_RDA", True)
    # Synthetic identity fixture; a fresh checkout needs no historical audit.
    groups = [[f"p{pid:02d}_{scenario}_r{repeat}"
               for scenario in ("Distance_80_cm", "Orientation_80_cm_Front_side", "Angle_0_deg_80_cm")]
              for pid in range(1, 11) for repeat in range(1, 5)]
    audit = tmp_path / "aliases.json"
    audit.write_text(json.dumps({
        "full_byte_verification": [{"tags": tags, "all_byte_equal": True} for tags in groups],
        "hr_sha256_groups": groups,
    }))
    output = tmp_path / "versioned"
    assert export.export_ftu(tmp_path, output, compress=False,
                             participant_id=1, frontend_config=FTUADCConfig(),
                             alias_audit=audit) == (3, 0)
    with (output / "FTU/manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert len(rows[0]["ftu_alias_audit_sha256"]) == 64
