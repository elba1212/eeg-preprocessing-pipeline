"""Tests for resampling with event timing preservation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eeg_pipeline.resampling import resample_raw, resample_with_event_report


class RawLike:
    """Tiny MNE Raw-like object for resampling tests."""

    def __init__(self, sfreq: float = 1000.0) -> None:
        self.info: dict[str, Any] = {"sfreq": sfreq}
        self.calls: list[dict[str, Any]] = []

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)

    def resample(self, sfreq: float, **kwargs: Any) -> Any:
        self.calls.append({"sfreq": sfreq, **kwargs})
        events = kwargs.get("events")
        old_sfreq = float(self.info["sfreq"])
        self.info["sfreq"] = sfreq
        if events is not None:
            resampled_events = np.array(events, copy=True)
            resampled_events[:, 0] = np.rint(resampled_events[:, 0] * sfreq / old_sfreq)
            return self, resampled_events
        return self


def test_resample_with_event_report_preserves_event_timing() -> None:
    raw = RawLike()
    events = np.array([[1000, 0, 1], [2000, 0, 2]])

    result = resample_with_event_report(raw, 250.0, events)

    assert result.raw is not raw
    assert result.raw.info["sfreq"] == 250.0
    assert result.events.tolist() == [[250, 0, 1], [500, 0, 2]]
    assert result.report.n_events == 2
    assert result.report.max_event_time_error_seconds == 0.0
    assert "resampling" in result.raw.info["temp"]["eeg_pipeline"]


def test_resample_raw_returns_only_raw_object() -> None:
    raw = RawLike()

    result = resample_raw(raw, 250.0)

    assert result.info["sfreq"] == 250.0
