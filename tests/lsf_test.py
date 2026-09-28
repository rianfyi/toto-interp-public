from __future__ import annotations

from pathlib import Path

from toto_interp.lsf import (
    default_lsf_data_path,
    ensure_lsf_datasets,
    normalize_lsf_layout,
    required_archives_for_lsf_datasets,
    validate_lsf_layout,
)


def test_required_archives_for_lsf_datasets_maps_dataset_names():
    archive_keys = required_archives_for_lsf_datasets(["ETTh1", "weather"])
    assert archive_keys == ("ett", "weather")


def test_normalize_lsf_layout_copies_files_into_expected_structure(tmp_path: Path):
    source_dir = tmp_path / "downloads" / "weird_folder"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "ETTh1.csv").write_text("date,OT\n2024-01-01,1\n")
    (source_dir / "electricity.csv").write_text("date,OT\n2024-01-01,1\n")

    normalize_lsf_layout(tmp_path, archive_keys=("ett", "electricity"))

    assert (tmp_path / "ETT-small" / "ETTh1.csv").exists()
    assert (tmp_path / "electricity" / "electricity.csv").exists()


def test_ensure_lsf_datasets_raises_helpful_error_for_missing_layout(tmp_path: Path):
    try:
        ensure_lsf_datasets(tmp_path, archive_keys=("weather",), download=False)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ensure_lsf_datasets to fail on missing files.")

    assert "scripts/download_lsf_datasets.py" in message
    assert "weather/weather.csv" in message


def test_validate_lsf_layout_and_default_path(tmp_path: Path):
    data_root = default_lsf_data_path(tmp_path)
    (data_root / "weather").mkdir(parents=True, exist_ok=True)
    (data_root / "weather" / "weather.csv").write_text("date,OT\n2024-01-01,1\n")

    ok, missing = validate_lsf_layout(data_root, archive_keys=("weather",))
    assert ok is True
    assert missing == []
