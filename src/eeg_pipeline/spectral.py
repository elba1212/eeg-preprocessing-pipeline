"""Spectral quality-control placeholders."""

from __future__ import annotations

from typing import Any


def compute_psd(raw: Any, fmin_hz: float = 1.0, fmax_hz: float = 45.0) -> Any:
    """Compute PSD once QC requirements are validated."""

    _ = (raw, fmin_hz, fmax_hz)
    raise NotImplementedError("PSD computation is not implemented yet.")


def summarize_band_power(psd: Any, bands: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Summarize band power once PSD computation is implemented."""

    _ = (psd, bands)
    raise NotImplementedError("Band-power summaries are not implemented yet.")
