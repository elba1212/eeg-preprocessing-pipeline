"""Tests for automatic bad-channel detection."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from eeg_pipeline.bad_channels import (
    BadChannelDetectionConfig,
    BadChannelReport,
    detect_bad_channels,
)
from eeg_pipeline.config import load_pipeline_config


class RawLike:
    """Synthetic raw-like object for pyPREP wrapper tests."""


class FakeNoisyChannels:
    """Fake pyPREP detector with deterministic detector outputs."""

    calls: list[str] = []

    def __init__(self, raw: Any, random_state: int | None = None) -> None:
        self.raw = raw
        self.random_state = random_state
        self.bad_by_deviation: list[str] = []
        self.bad_by_correlation: list[str] = []
        self.bad_by_hf_noise: list[str] = []
        self.bad_by_ransac: list[str] = []

    def find_bad_by_deviation(self) -> None:
        self.calls.append("deviation")
        self.bad_by_deviation = ["E1", "E2"]

    def find_bad_by_correlation(self) -> None:
        self.calls.append("correlation")
        self.bad_by_correlation = ["E2", "E3"]

    def find_bad_by_hfnoise(self) -> None:
        self.calls.append("high_frequency_noise")
        self.bad_by_hf_noise = ["E4"]

    def find_bad_by_ransac(self) -> None:
        self.calls.append("ransac")
        self.bad_by_ransac = ["E5"]


def _install_fake_pyprep(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNoisyChannels.calls = []
    pyprep = types.ModuleType("pyprep")
    find_noisy_channels = types.ModuleType("pyprep.find_noisy_channels")
    find_noisy_channels.NoisyChannels = FakeNoisyChannels
    monkeypatch.setitem(sys.modules, "pyprep", pyprep)
    monkeypatch.setitem(sys.modules, "pyprep.find_noisy_channels", find_noisy_channels)


def test_detect_bad_channels_returns_structured_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pyprep(monkeypatch)

    report = detect_bad_channels(RawLike())

    assert isinstance(report, BadChannelReport)
    assert report.detector == "pyprep"
    assert report.detector_results == {
        "deviation": ("E1", "E2"),
        "correlation": ("E2", "E3"),
        "high_frequency_noise": ("E4",),
        "ransac": ("E5",),
    }
    assert report.final_bad_channels == ("E1", "E2", "E3", "E4", "E5")
    assert report.processing_time_seconds >= 0.0
    assert FakeNoisyChannels.calls == [
        "deviation",
        "correlation",
        "high_frequency_noise",
        "ransac",
    ]


def test_detect_bad_channels_can_disable_individual_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pyprep(monkeypatch)

    report = detect_bad_channels(
        RawLike(),
        BadChannelDetectionConfig(correlation=False, ransac=False),
    )

    assert set(report.detector_results) == {"deviation", "high_frequency_noise"}
    assert report.final_bad_channels == ("E1", "E2", "E4")
    assert FakeNoisyChannels.calls == ["deviation", "high_frequency_noise"]


def test_detect_bad_channels_disabled_returns_empty_report() -> None:
    report = detect_bad_channels(RawLike(), {"enabled": False})

    assert report.detector_results == {}
    assert report.final_bad_channels == ()


def test_bad_channel_config_exposes_requested_detectors() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.bad_channels.method == "pyprep"
    assert config.bad_channels.deviation
    assert config.bad_channels.correlation
    assert config.bad_channels.high_frequency_noise
    assert config.bad_channels.ransac
