"""Tests for configuration and safety helpers."""

from pathlib import Path
import shutil
import uuid

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.safety import find_private_data


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_default_config_matches_planned_sampling() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.sampling.original_frequency_hz == 1000.0
    assert config.sampling.target_frequency_hz == 250.0
    assert config.filtering.analysis_high_pass_hz == 0.1
    assert config.filtering.ica_high_pass_hz == 1.0


def test_find_private_data_detects_mff_without_reading_payload() -> None:
    root = _workspace_tmp()
    try:
        recording = root / "data" / "raw" / "EL3001" / "EL3001_DIS1.mff"
        recording.mkdir(parents=True)

        assert find_private_data(root) == [recording]
    finally:
        shutil.rmtree(root, ignore_errors=True)
