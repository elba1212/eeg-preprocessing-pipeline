"""Tests for transformation helpers without reading private data."""

from __future__ import annotations

import copy

import numpy as np

from eeg_pipeline.events import compare_event_timing
from eeg_pipeline.metrics import compute_quality_metrics
from eeg_pipeline.reference import set_eeg_reference
from eeg_pipeline.resampling import resample_with_event_report


class FakeAnnotations:
    """Minimal annotation container for unit tests."""

    def __len__(self) -> int:
        return 0


class FakeRaw:
    """Tiny Raw-like object for fast transform unit tests."""

    def __init__(self, sfreq: float = 1000.0) -> None:
        self.info = {"sfreq": sfreq, "bads": [], "custom_ref_applied": False}
        self.ch_names = ["E1", "E2", "E3"]
        self.annotations = FakeAnnotations()
        self.times = np.arange(2000) / sfreq
        self._data = np.vstack(
            [
                np.sin(np.linspace(0, 10, 2000)),
                np.cos(np.linspace(0, 10, 2000)),
                np.zeros(2000),
            ]
        )

    def copy(self) -> "FakeRaw":
        return copy.deepcopy(self)

    def resample(self, sfreq: float, events: np.ndarray | None = None):
        old_sfreq = self.info["sfreq"]
        self.info["sfreq"] = sfreq
        self.times = np.arange(self._data.shape[1]) / sfreq
        if events is None:
            return self
        new_events = events.copy()
        new_events[:, 0] = np.round(events[:, 0] * sfreq / old_sfreq).astype(int)
        return self, new_events

    def get_data(self, picks: str | None = None) -> np.ndarray:
        _ = picks
        return self._data

    def set_eeg_reference(self, ref_channels="average", projection: bool = False) -> "FakeRaw":
        _ = (ref_channels, projection)
        self.info["custom_ref_applied"] = True
        return self


def test_resample_with_event_report_preserves_event_timing() -> None:
    raw = FakeRaw()
    events = np.array([[100, 0, 1], [1000, 0, 2]])

    resampled, resampled_events, report = resample_with_event_report(raw, 250.0, events)

    assert resampled.info["sfreq"] == 250.0
    assert resampled_events.tolist() == [[25, 0, 1], [250, 0, 2]]
    assert report.ok


def test_event_timing_detects_count_mismatch() -> None:
    report = compare_event_timing(np.array([[1, 0, 1]]), 1000.0, np.empty((0, 3)), 250.0)

    assert not report.ok


def test_quality_metrics_on_raw_like_object() -> None:
    metrics = compute_quality_metrics(FakeRaw())

    assert metrics["sfreq_hz"] == 1000.0
    assert metrics["n_eeg_channels"] == 3


def test_average_reference_returns_copy() -> None:
    raw = FakeRaw()

    referenced = set_eeg_reference(raw)

    assert referenced is not raw
    assert referenced.info["custom_ref_applied"]
