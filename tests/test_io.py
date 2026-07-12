"""Tests for EEG loading and path discovery utilities."""

from pathlib import Path
import shutil
import uuid

import pytest

from eeg_pipeline.io import discover_eeg_files, inspect_recording_path, load_raw_eeg


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_discover_eeg_files_finds_mff_directories() -> None:
    root = _workspace_tmp()
    try:
        recording = root / "EL0001" / "EL0001_DIS1_20220208_100620.mff"
        recording.mkdir(parents=True)

        assert discover_eeg_files(root) == [recording]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_inspect_recording_path_does_not_load_payload() -> None:
    info = inspect_recording_path(Path("data/raw/demo/EL0001/EL0001_DIS1.mff"))

    assert info["participant_id"] == "EL0001"
    assert info["run_label"] == "DIS1"
    assert info["format"] == "mff"


def test_load_raw_eeg_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_eeg("data/raw/example.edf")
