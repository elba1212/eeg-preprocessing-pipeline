"""Spectral analysis utilities for EEG quality control."""

from __future__ import annotations

from typing import Any


def compute_psd(raw: Any, fmin_hz: float = 1.0, fmax_hz: float = 45.0) -> Any | None:
    """Compute power spectral density for raw EEG data."""

    return raw.compute_psd(fmin=fmin_hz, fmax=fmax_hz, picks="eeg")


def summarize_band_power(psd: Any, bands: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Summarize spectral power within named frequency bands."""
    import numpy as np

    freqs = psd.freqs
    data = psd.get_data()
    summary: dict[str, float] = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        summary[name] = float(np.mean(data[..., mask])) if mask.any() else float("nan")
    return summary
