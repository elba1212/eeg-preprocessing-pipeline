"""Visualization placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_psd_summary(psd: Any, output_path: str | Path | None = None) -> Path | None:
    """Plot PSD summaries once visualization requirements are validated."""

    _ = (psd, output_path)
    raise NotImplementedError("PSD visualization is not implemented yet.")


def plot_bad_channel_summary(
    bad_channels: list[str],
    output_path: str | Path | None = None,
) -> Path | None:
    """Plot bad-channel summaries once visualization requirements are validated."""

    _ = (bad_channels, output_path)
    raise NotImplementedError("Bad-channel visualization is not implemented yet.")
