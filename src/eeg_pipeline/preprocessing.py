"""Compatibility preprocessing facade.

The pipeline is implemented as focused transformations in dedicated modules. This module remains
as a small facade for older imports.
"""

from __future__ import annotations

from typing import Any

from eeg_pipeline.config import PipelineConfig, config_from_dict
from eeg_pipeline.filtering import filter_raw as _filter_raw
from eeg_pipeline.line_noise import apply_notch_filter
from eeg_pipeline.resampling import resample_raw


def preprocess_raw(raw: Any, config: dict[str, Any] | PipelineConfig | None = None) -> Any:
    """Apply the early deterministic transforms used by the continuous pipeline."""

    cfg = config if isinstance(config, PipelineConfig) else config_from_dict(config)
    processed = resample_raw(raw, cfg.sampling.target_frequency_hz)
    processed = apply_notch_filter(
        processed,
        frequency_hz=cfg.line_noise.frequency_hz,
        harmonics=cfg.line_noise.harmonics,
        enabled=cfg.line_noise.enabled,
    )
    return _filter_raw(
        processed,
        cfg.filtering.analysis_high_pass_hz,
        cfg.filtering.low_pass_hz,
        method=cfg.filtering.method,
    )


def filter_raw(raw: Any, high_pass_hz: float | None, low_pass_hz: float | None) -> Any:
    """Apply high-pass and low-pass filters to raw EEG data."""

    return _filter_raw(raw, high_pass_hz, low_pass_hz)


def notch_filter_raw(raw: Any, notch_frequency_hz: float | list[float] | None) -> Any:
    """Apply notch filtering to remove line noise."""

    filtered = raw.copy()
    if notch_frequency_hz is None:
        return filtered
    return filtered.notch_filter(freqs=notch_frequency_hz)
