"""Tests for continuous noisy-window annotation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eeg_pipeline.annotations import NoisyWindowConfig, annotate_noisy_windows, detect_noisy_windows
from eeg_pipeline.config import load_pipeline_config


class RawLike:
    """Synthetic Raw-like object for annotation tests."""

    def __init__(self, data: np.ndarray, sfreq: float = 10.0) -> None:
        self.data = data
        self.info: dict[str, Any] = {"sfreq": sfreq}
        self.annotations: list[dict[str, Any]] = []

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)

    def get_data(self, picks: str | None = None) -> np.ndarray:
        _ = picks
        return self.data

    def add_annotations(self, annotations: tuple[dict[str, Any], ...]) -> None:
        self.annotations.extend(annotations)


def _base_data() -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, 140)
    return np.vstack([np.sin(t), np.cos(t)])


def _config() -> NoisyWindowConfig:
    return NoisyWindowConfig(
        window_seconds=2.0,
        zscore_threshold=2.0,
        flat_peak_to_peak_threshold=1e-9,
        flat_variance_threshold=1e-18,
        clip_threshold=5.0,
        clipped_fraction_threshold=0.05,
    )


def test_detect_noisy_windows_uses_two_second_windows() -> None:
    decisions = detect_noisy_windows(RawLike(_base_data()), _config())

    assert len(decisions) == 7
    assert all(decision.duration_seconds == 2.0 for decision in decisions)


def test_detect_noisy_windows_records_requested_reasons() -> None:
    data = _base_data()
    data[:, 20:40] *= 20.0  # peak-to-peak and variance
    data[0, 40:60] = 0.0  # flat signal
    data[1, 60:80:2] = 4.0  # high-frequency noise and abrupt jumps
    data[:, 80:100] = 6.0  # clipped samples and flat signal
    data[0, 100:120] = np.nan  # NaNs

    decisions = detect_noisy_windows(RawLike(data), _config())
    reasons_by_window = [set(decision.reasons) for decision in decisions]

    assert {"peak_to_peak", "variance"} <= reasons_by_window[1]
    assert "flat_signal" in reasons_by_window[2]
    assert {"high_frequency_noise", "abrupt_jumps"} <= reasons_by_window[3]
    assert {"clipped_samples", "flat_signal"} <= reasons_by_window[4]
    assert "nans" in reasons_by_window[5]


def test_annotate_noisy_windows_returns_report_and_preserves_input() -> None:
    data = _base_data()
    data[:, 20:40] *= 20.0
    raw = RawLike(data)

    report = annotate_noisy_windows(raw, _config())

    assert report.raw is not raw
    assert raw.annotations == []
    assert any(annotation["onset"] == 2.0 for annotation in report.annotations)
    assert all(annotation["description"] == "BAD_noisy_window" for annotation in report.annotations)
    assert report.raw.annotations == list(report.annotations)
    assert "decisions" in report.raw.info["temp"]["eeg_pipeline"]["noisy_windows"]


def test_annotate_noisy_windows_does_not_reject_data() -> None:
    data = _base_data()
    data[:, 20:40] *= 20.0

    report = annotate_noisy_windows(RawLike(data), _config())

    assert report.raw.data.shape == data.shape


def test_disabled_noisy_window_detection_records_empty_decisions() -> None:
    report = annotate_noisy_windows(RawLike(_base_data()), NoisyWindowConfig(enabled=False))

    assert report.decisions == ()
    assert report.annotations == ()


def test_default_config_exposes_annotation_settings() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.annotations.window_seconds == 2.0
    assert config.annotations.description == "BAD_noisy_window"
