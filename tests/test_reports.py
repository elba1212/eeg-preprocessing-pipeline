"""Tests for HTML quality-control dashboards."""

from __future__ import annotations

import sys
import types
from pathlib import Path
import shutil
from typing import Any
import uuid

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.reports import (
    ReportConfig,
    build_recording_sections,
    generate_participant_dashboard,
    generate_recording_dashboard,
    generate_report,
)


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


class FakeMneReport:
    """Small MNE Report-like object for testing optional integration."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *, title: str) -> None:
        self.title = title
        self.html = ""

    def add_html(self, *, html: str, title: str) -> None:
        self.html = html
        FakeMneReport.calls.append({"title": title, "html": html})

    def save(self, output_path, *, overwrite: bool, open_browser: bool) -> None:
        assert overwrite
        assert not open_browser
        output_path.write_text(self.html, encoding="utf-8")


def test_recording_dashboard_contains_required_sections_and_redacts_private_fields() -> None:
    root = _workspace_tmp()
    try:
        output_path = root / "recording.html"

        report = generate_recording_dashboard(
            output_path,
            recording_label="Recording A",
            metadata={
                "sampling_frequency_hz": 250,
                "participant_id": "EL0001",
                "raw_path": r"C:\private\data\EL0001.mff",
            },
            preprocessing_summary={"filtering": "0.1-40 Hz", "event_id": {"face": 1}},
            bad_channels={"final_bad_channels": ["E1", "E2"]},
            interpolation={"interpolated_channels": ["E1", "E2"]},
            noisy_windows={"n_noisy_windows": 3},
            ica={"removed_components": [0, 2]},
            psd={"alpha_power": 1.2},
            quality_metrics={"n_epochs_dropped": 1},
            config=ReportConfig(use_mne_report=False),
        )

        html = output_path.read_text(encoding="utf-8")
        assert report.output_path == output_path
        assert report.sections == (
            "Metadata",
            "Preprocessing Summary",
            "Bad Channels",
            "Interpolation",
            "Noisy Windows",
            "ICA",
            "PSD",
            "Quality Metrics",
        )
        assert "Recording A Quality-Control Dashboard" in html
        assert "0.1-40 Hz" in html
        assert "EL0001" not in html
        assert r"C:\private" not in html
        assert "[redacted]" in html
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_participant_dashboard_summarizes_recordings() -> None:
    root = _workspace_tmp()
    try:
        output_path = root / "participant.html"

        report = generate_participant_dashboard(
            output_path,
            [
                {"recording": "run-1", "bad_channels": 2, "n_noisy_windows": 4},
                {"recording": "run-2", "bad_channels": 1, "n_noisy_windows": 0},
            ],
            participant_label="Participant A",
            participant_summary={"n_recordings": 2, "participant_id": "EL0001"},
            config={"use_mne_report": False},
        )

        html = output_path.read_text(encoding="utf-8")
        assert report.n_recordings == 2
        assert "Participant A Quality-Control Summary" in html
        assert "run-1" in html
        assert "run-2" in html
        assert "EL0001" not in html
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_recording_sections_is_modular() -> None:
    sections = build_recording_sections(
        metadata={"sfreq": 250},
        bad_channels={"final_bad_channels": ["E1"]},
    )

    assert [section.title for section in sections] == [
        "Metadata",
        "Preprocessing Summary",
        "Bad Channels",
        "Interpolation",
        "Noisy Windows",
        "ICA",
        "PSD",
        "Quality Metrics",
    ]


def test_recording_dashboard_can_use_mne_report(monkeypatch) -> None:
    FakeMneReport.calls = []
    fake_mne = types.ModuleType("mne")
    fake_mne.Report = FakeMneReport
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    root = _workspace_tmp()
    try:
        report = generate_recording_dashboard(
            root / "mne-report.html",
            metadata={"sampling_frequency_hz": 250},
            config=ReportConfig(use_mne_report=True),
        )

        assert report.used_mne_report
        assert FakeMneReport.calls
        assert report.output_path.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generate_report_compatibility_helper() -> None:
    root = _workspace_tmp()
    try:
        output_path = root / "compat.html"

        result = generate_report(
            {"n_bad_channels": 2},
            output_path,
            figures=["ignored-private-path-name.png"],
            title="Compat",
        )

        assert result == output_path
        html = output_path.read_text(encoding="utf-8")
        assert "n_bad_channels" in html
        assert "ignored-private-path-name.png" not in html
        assert "figure_001" in html
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_config_exposes_report_privacy_settings() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.reports.use_mne_report
    assert config.reports.redact_private_fields
    assert config.reports.include_empty_sections
