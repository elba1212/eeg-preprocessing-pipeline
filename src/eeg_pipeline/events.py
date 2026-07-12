"""Event extraction and timing validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventTimingReport:
    """Comparison of event timing before and after a transform."""

    ok: bool
    max_abs_time_error_seconds: float
    event_count_before: int
    event_count_after: int


def extract_events(raw: Any, event_id: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Extract events from annotations, falling back to stim channels when available."""

    import mne

    if len(raw.annotations) > 0:
        events, ids = mne.events_from_annotations(raw, event_id=event_id)
        return events, ids
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    return events, event_id or {}


def event_times_seconds(events: np.ndarray, sfreq: float) -> np.ndarray:
    """Convert MNE event samples to seconds."""

    if events.size == 0:
        return np.array([], dtype=float)
    return events[:, 0].astype(float) / float(sfreq)


def compare_event_timing(
    before_events: np.ndarray,
    before_sfreq: float,
    after_events: np.ndarray,
    after_sfreq: float,
    *,
    tolerance_seconds: float = 1 / 250,
) -> EventTimingReport:
    """Check that event times remain stable across resampling."""

    before = event_times_seconds(before_events, before_sfreq)
    after = event_times_seconds(after_events, after_sfreq)
    if len(before) != len(after):
        return EventTimingReport(False, float("inf"), len(before), len(after))
    if len(before) == 0:
        return EventTimingReport(True, 0.0, 0, 0)
    max_error = float(np.max(np.abs(before - after)))
    return EventTimingReport(max_error <= tolerance_seconds, max_error, len(before), len(after))
