"""Resampling placeholders."""

from __future__ import annotations

from typing import Any


def resample_raw(raw: Any, target_sfreq_hz: float = 250.0, events: Any | None = None) -> Any:
    """Resample raw data once event-preservation behavior is validated."""

    _ = (raw, target_sfreq_hz, events)
    raise NotImplementedError("Resampling is not implemented yet.")


def resample_with_event_report(
    raw: Any, target_sfreq_hz: float, events: Any
) -> tuple[Any, Any, Any]:
    """Resample and report event timing once resampling is implemented."""

    _ = (raw, target_sfreq_hz, events)
    raise NotImplementedError("Resampling with event reporting is not implemented yet.")
