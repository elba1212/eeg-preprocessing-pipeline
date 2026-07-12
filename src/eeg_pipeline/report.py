"""Backward-compatible report imports.

New code should import from :mod:`eeg_pipeline.reports`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.reports import generate_report as _generate_report


def generate_report(
    metrics: dict[str, Any],
    output_path: str | Path,
    figures: list[str | Path] | None = None,
) -> Path:
    """Generate a preprocessing quality-control report."""

    return _generate_report(metrics, output_path, figures)
