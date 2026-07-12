"""Tests for resumable dataset batch processing."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
from typing import Any
import uuid

from eeg_pipeline.batch import BatchConfig, process_dataset, summarize_batch
from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.records import Recording


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _make_recording(root: Path, participant: str, name: str) -> Path:
    recording = root / participant / f"{name}.mff"
    recording.mkdir(parents=True)
    return recording


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_process_dataset_writes_status_log_and_summary_csv() -> None:
    root = _workspace_tmp()
    try:
        input_dir = root / "input"
        output_dir = root / "out"
        _make_recording(input_dir, "EL0001", "EL0001_DIS1")
        _make_recording(input_dir, "EL0002", "EL0002_DIS2")

        def processor(recording: Recording, recording_output_dir: Path) -> dict[str, Any]:
            assert recording.path.exists()
            assert recording_output_dir.exists()
            return {"n_bad_channels": 2}

        report = process_dataset(
            input_dir,
            output_dir=output_dir,
            processor=processor,
            config=BatchConfig(show_progress=False),
        )

        rows = _read_summary(report.summary_csv_path)
        assert report.total_recordings == 2
        assert report.completed == 2
        assert report.failed == 0
        assert report.skipped == 0
        assert report.log_path.exists()
        assert len(rows) == 2
        assert rows[0]["participant_key"].startswith("participant_")
        assert rows[0]["metric_n_bad_channels"] == "2"
        assert "EL0001" not in report.summary_csv_path.read_text(encoding="utf-8")
        assert all((result.output_dir / "status.json").exists() for result in report.results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_process_dataset_resumes_completed_recordings() -> None:
    root = _workspace_tmp()
    try:
        input_dir = root / "input"
        output_dir = root / "out"
        _make_recording(input_dir, "EL0001", "EL0001_DIS1")
        calls = {"count": 0}

        def processor(recording: Recording, recording_output_dir: Path) -> dict[str, Any]:
            _ = (recording, recording_output_dir)
            calls["count"] += 1
            return {"processed": 1}

        first = process_dataset(
            input_dir,
            output_dir=output_dir,
            processor=processor,
            config={"show_progress": False, "resume": True},
        )
        second = process_dataset(
            input_dir,
            output_dir=output_dir,
            processor=processor,
            config={"show_progress": False, "resume": True},
        )

        assert first.completed == 1
        assert second.completed == 0
        assert second.skipped == 1
        assert second.results[0].resumed
        assert calls["count"] == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_process_dataset_records_failures_without_leaking_private_identifiers() -> None:
    root = _workspace_tmp()
    try:
        input_dir = root / "input"
        output_dir = root / "out"
        recording_path = _make_recording(input_dir, "EL0001", "EL0001_DIS1")

        def processor(recording: Recording, recording_output_dir: Path) -> dict[str, Any]:
            _ = recording_output_dir
            raise RuntimeError(f"Failed for {recording.path} and {recording.participant_id}")

        report = process_dataset(
            input_dir,
            output_dir=output_dir,
            processor=processor,
            config={"show_progress": False},
        )

        summary_text = report.summary_csv_path.read_text(encoding="utf-8")
        assert report.failed == 1
        assert "[recording-path]" in report.results[0].message
        assert "[participant]" in report.results[0].message
        assert str(recording_path) not in summary_text
        assert "EL0001" not in summary_text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_process_dataset_supports_thread_parallelization() -> None:
    root = _workspace_tmp()
    try:
        input_dir = root / "input"
        output_dir = root / "out"
        _make_recording(input_dir, "EL0001", "EL0001_DIS1")
        _make_recording(input_dir, "EL0002", "EL0002_DIS2")
        _make_recording(input_dir, "EL0003", "EL0003_DIS3")

        def processor(recording: Recording, recording_output_dir: Path) -> dict[str, Any]:
            _ = (recording, recording_output_dir)
            return {"ok": 1}

        report = process_dataset(
            input_dir,
            output_dir=output_dir,
            processor=processor,
            config={"show_progress": False, "n_jobs": 2, "parallel_backend": "thread"},
        )

        assert report.completed == 3
        assert [result.recording_key for result in report.results] == sorted(
            result.recording_key for result in report.results
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_summarize_batch_counts_recordings_without_loading_payloads() -> None:
    root = _workspace_tmp()
    try:
        input_dir = root / "input"
        _make_recording(input_dir, "EL0001", "EL0001_DIS1")
        _make_recording(input_dir, "EL0001", "EL0001_DIS2")

        assert summarize_batch(input_dir) == {"participants": 1, "recordings": 2}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_config_exposes_batch_settings() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.batch.enabled
    assert config.batch.resume
    assert config.batch.output_dir == Path("outputs/batch")
    assert config.batch.anonymize_identifiers
