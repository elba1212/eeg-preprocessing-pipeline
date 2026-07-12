"""Filtering transforms for analysis and ICA copies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilteredCopies:
    """Separate filtered views used by the pipeline."""

    analysis: Any
    ica: Any


def filter_raw(
    raw: Any,
    high_pass_hz: float | None,
    low_pass_hz: float | None,
    *,
    method: str = "fir",
) -> Any:
    """Return a filtered copy of a raw object."""

    filtered = raw.copy()
    return filtered.filter(l_freq=high_pass_hz, h_freq=low_pass_hz, method=method)


def make_filtered_copies(
    raw: Any,
    *,
    analysis_high_pass_hz: float = 0.1,
    ica_high_pass_hz: float = 1.0,
    low_pass_hz: float = 40.0,
    method: str = "fir",
) -> FilteredCopies:
    """Create analysis and ICA-fitting filtered copies."""

    return FilteredCopies(
        analysis=filter_raw(raw, analysis_high_pass_hz, low_pass_hz, method=method),
        ica=filter_raw(raw, ica_high_pass_hz, low_pass_hz, method=method),
    )
