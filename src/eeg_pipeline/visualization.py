"""Visualization helpers for EEG preprocessing quality control."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_psd_summary(psd: Any, output_path: str | Path | None = None) -> Path | None:
    """Create a PSD summary figure."""
    # TODO: Plot channel-level and aggregate spectral summaries.
    _ = psd
    return Path(output_path) if output_path is not None else None


def plot_bad_channel_summary(
    bad_channels: list[str],
    output_path: str | Path | None = None,
) -> Path | None:
    """Create a summary plot for detected bad channels."""
    # TODO: Visualize bad-channel counts and topographic distribution when montage is available.
    _ = bad_channels
    return Path(output_path) if output_path is not None else None
