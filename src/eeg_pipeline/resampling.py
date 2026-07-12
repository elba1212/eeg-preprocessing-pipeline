"""Resampling helpers that preserve event timing."""

from __future__ import annotations

from typing import Any

import numpy as np

from eeg_pipeline.events import EventTimingReport, compare_event_timing


def resample_raw(raw: Any, target_sfreq_hz: float = 250.0, events: np.ndarray | None = None) -> Any:
    """Return a resampled copy of a raw object."""

    resampled = raw.copy()
    if events is None:
        return resampled.resample(target_sfreq_hz)
    result = resampled.resample(target_sfreq_hz, events=events)
    if isinstance(result, tuple):
        return result
    return result, events


def resample_with_event_report(
    raw: Any,
    target_sfreq_hz: float,
    events: np.ndarray,
) -> tuple[Any, np.ndarray, EventTimingReport]:
    """Resample raw data and report event-timing preservation."""

    before_sfreq = float(raw.info["sfreq"])
    resampled, resampled_events = resample_raw(raw, target_sfreq_hz, events)
    report = compare_event_timing(events, before_sfreq, resampled_events, float(resampled.info["sfreq"]))
    return resampled, resampled_events, report
