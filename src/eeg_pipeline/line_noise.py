"""Line-noise diagnostics and optional notch filtering."""

from __future__ import annotations

from typing import Any


def notch_frequencies(base_frequency_hz: float = 50.0, harmonics: int = 1) -> list[float]:
    """Return base line frequency and requested harmonics."""

    return [base_frequency_hz * index for index in range(1, harmonics + 1)]


def apply_notch_filter(
    raw: Any,
    *,
    frequency_hz: float = 50.0,
    harmonics: int = 1,
    enabled: bool = False,
) -> Any:
    """Optionally return a notch-filtered copy of raw data."""

    filtered = raw.copy()
    if not enabled:
        return filtered
    return filtered.notch_filter(freqs=notch_frequencies(frequency_hz, harmonics))
