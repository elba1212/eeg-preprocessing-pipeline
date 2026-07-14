"""Resampling transforms with event-timing checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResamplingReport:
    """Timing and provenance for a resampling operation."""

    original_sfreq_hz: float
    target_sfreq_hz: float
    output_sfreq_hz: float
    n_events: int
    max_event_time_error_seconds: float
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ResamplingResult:
    """Resampled raw object, optional resampled events, and timing report."""

    raw: Any
    events: np.ndarray | None
    report: ResamplingReport

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result mapping without embedding raw data."""

        return {
            "events_present": self.events is not None,
            "report": self.report.to_dict(),
        }


def _as_events(events: Any | None) -> np.ndarray | None:
    if events is None:
        return None
    array = np.asarray(events, dtype=int)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3).")
    return array


def _event_time_error_seconds(
    before_events: np.ndarray | None,
    after_events: np.ndarray | None,
    *,
    before_sfreq: float,
    after_sfreq: float,
) -> float:
    if before_events is None or after_events is None or before_events.size == 0:
        return 0.0
    before_times = before_events[:, 0].astype(float) / before_sfreq
    after_times = after_events[:, 0].astype(float) / after_sfreq
    return float(np.max(np.abs(before_times - after_times)))


def _record_resampling_provenance(raw: Any, report: ResamplingReport) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["resampling"] = report.to_dict()


def resample_with_event_report(
    raw: Any,
    target_sfreq_hz: float,
    events: Any | None = None,
    *,
    npad: str | int = "auto",
    verbose: bool | str | int | None = None,
) -> ResamplingResult:
    """Resample a raw copy and preserve event timing when events are supplied."""

    if target_sfreq_hz <= 0:
        raise ValueError("target_sfreq_hz must be positive.")
    before_events = _as_events(events)
    before_sfreq = float(raw.info["sfreq"])
    start = perf_counter()
    resampled = raw.copy()
    if before_events is None:
        resampled.resample(target_sfreq_hz, npad=npad, verbose=verbose)
        after_events = None
    else:
        result = resampled.resample(
            target_sfreq_hz,
            events=before_events,
            npad=npad,
            verbose=verbose,
        )
        if isinstance(result, tuple):
            resampled, after_events = result
        else:
            after_events = np.array(before_events, copy=True)
            after_events[:, 0] = np.rint(after_events[:, 0] * target_sfreq_hz / before_sfreq)
    output_sfreq = float(resampled.info["sfreq"])
    report = ResamplingReport(
        original_sfreq_hz=before_sfreq,
        target_sfreq_hz=float(target_sfreq_hz),
        output_sfreq_hz=output_sfreq,
        n_events=0 if before_events is None else int(before_events.shape[0]),
        max_event_time_error_seconds=_event_time_error_seconds(
            before_events,
            after_events,
            before_sfreq=before_sfreq,
            after_sfreq=output_sfreq,
        ),
        processing_time_seconds=perf_counter() - start,
    )
    _record_resampling_provenance(resampled, report)
    return ResamplingResult(raw=resampled, events=after_events, report=report)


def resample_raw(raw: Any, target_sfreq_hz: float = 250.0, events: Any | None = None) -> Any:
    """Backward-compatible helper returning only the resampled raw object."""

    return resample_with_event_report(raw, target_sfreq_hz, events).raw
