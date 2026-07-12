"""Line-noise filtering helpers."""

from __future__ import annotations

from typing import Any

from eeg_pipeline.filtering import FirFilterConfig, NotchFilterConfig, filter_raw


def notch_frequencies(base_frequency_hz: float = 50.0, harmonics: int = 1) -> list[float]:
    """Return line frequency and requested harmonics."""

    if base_frequency_hz <= 0:
        raise ValueError("base_frequency_hz must be positive.")
    if harmonics < 1:
        raise ValueError("harmonics must be at least 1.")
    return [base_frequency_hz * index for index in range(1, harmonics + 1)]


def apply_notch_filter(
    raw: Any,
    *,
    line_frequency_hz: float = 50.0,
    harmonics: int = 1,
    fir_config: FirFilterConfig | None = None,
    picks: str | list[str] = "eeg",
    verbose: bool | str | int | None = None,
) -> Any:
    """Create a copy with only line-noise notch filtering applied."""

    result = filter_raw(
        raw,
        high_pass_hz=None,
        low_pass_hz=None,
        copy_name="notch",
        fir_config=fir_config,
        notch_config=NotchFilterConfig(
            enabled=True,
            line_frequency_hz=line_frequency_hz,
            harmonics=harmonics,
        ),
        picks=picks,
        verbose=verbose,
    )
    return result.raw
