"""Tests for package API exports and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import eeg_pipeline
from eeg_pipeline.cli import main


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_package_exports_core_api() -> None:
    assert eeg_pipeline.__version__ == "0.1.0"
    assert hasattr(eeg_pipeline, "Pipeline")
    assert hasattr(eeg_pipeline, "PipelineConfig")
    assert hasattr(eeg_pipeline, "BatchConfig")
    assert hasattr(eeg_pipeline, "process_dataset")
    assert hasattr(eeg_pipeline, "generate_recording_dashboard")


def test_cli_batch_summary_is_read_only(capsys) -> None:
    root = _workspace_tmp()
    try:
        recording = root / "input" / "EL0000" / "EL0000_DIS1.mff"
        recording.mkdir(parents=True)

        exit_code = main(["batch-summary", str(root / "input")])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert exit_code == 0
        assert payload == {"participants": 1, "recordings": 1}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cli_preprocess_recording_reports_unimplemented(capsys) -> None:
    exit_code = main(["preprocess-recording", "missing.mff"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["implemented"] is False
