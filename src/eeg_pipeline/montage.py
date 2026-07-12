"""Montage and channel metadata validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MontageValidation:
    """Result of validating channel metadata."""

    ok: bool
    eeg_channel_count: int
    sampling_frequency_hz: float
    problems: tuple[str, ...]


def validate_montage(
    raw: Any,
    *,
    expected_eeg_channels: int = 256,
    expected_sfreq_hz: float | None = 1000.0,
) -> MontageValidation:
    """Validate channel count, types, sampling rate, and montage availability."""

    problems: list[str] = []
    eeg_picks = raw.copy().pick("eeg").ch_names if hasattr(raw, "copy") else []
    sfreq = float(raw.info.get("sfreq", 0.0))
    if expected_eeg_channels and len(eeg_picks) != expected_eeg_channels:
        problems.append(f"Expected {expected_eeg_channels} EEG channels, found {len(eeg_picks)}.")
    if expected_sfreq_hz is not None and abs(sfreq - expected_sfreq_hz) > 1e-6:
        problems.append(f"Expected sampling rate {expected_sfreq_hz:g} Hz, found {sfreq:g} Hz.")
    if not raw.get_montage():
        problems.append("No montage is set on the raw object.")
    return MontageValidation(
        ok=not problems,
        eeg_channel_count=len(eeg_picks),
        sampling_frequency_hz=sfreq,
        problems=tuple(problems),
    )


def apply_egi_montage(raw: Any, montage_name: str = "GSN-HydroCel-256", on_missing: str = "warn") -> Any:
    """Apply a standard EGI HydroCel montage in-place and return the raw object."""

    raw.set_montage(montage_name, on_missing=on_missing)
    return raw
