from src.data.loaders.ftu_loader import FTUDataLoader


def test_elevated_reference_is_flat_while_radar_is_nested(tmp_path):
    radar = tmp_path / "Radar data" / "Participant 2" / "4. Elevated" / "80 cm" / "1"
    reference = tmp_path / "HR_Ref_Values" / "Participant 2" / "4. Elevated"
    radar.mkdir(parents=True)
    reference.mkdir(parents=True)
    (tmp_path / "BR_Ref_Values").mkdir()
    (radar / "data_Raw_0.bin").write_bytes(b"adc")
    (reference / "R1.CSV").write_text(
        "meta\nmeta\nmeta\n1,00:00:00,120\n,00:00:01,121\n", encoding="utf-8")

    loader = FTUDataLoader(str(tmp_path))
    samples = loader.list_available_samples(participant_id=2, scenario="Elevated")
    assert len(samples) == 1
    assert samples[0]["distance"] == "80 cm"
    assert samples[0]["hr_file"] == str(reference / "R1.CSV")
    assert loader._build_hr_file_path(2, "Elevated", "80 cm", 1) == reference / "R1.CSV"
    ref = loader.load_reference_data(2, "Elevated", "80 cm", 1)
    assert ref["heart_rate"].tolist() == [120, 121]
    assert ref["timestamps"].tolist() == [0, 1]
