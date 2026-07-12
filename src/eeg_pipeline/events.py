"""Event handling placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventTimingReport:
    """Planned event-timing validation result."""

    ok: bool
    max_abs_time_error_seconds: float
    event_count_before: int
    event_count_after: int


def extract_events(raw: Any, event_id: dict[str, int] | None = None) -> tuple[Any, dict[str, int]]:
    """Extract events once EGI event streams are validated."""

    _ = (raw, event_id)
    raise NotImplementedError("Event extraction is not implemented yet.")


def compare_event_timing(*args: Any, **kwargs: Any) -> EventTimingReport:
    """Compare event timing once resampling behavior is validated."""

    _ = (args, kwargs)
    raise NotImplementedError("Event-timing comparison is not implemented yet.")
