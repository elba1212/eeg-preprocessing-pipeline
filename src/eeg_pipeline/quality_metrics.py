"""Backward-compatible quality metric imports.

New code should import from :mod:`eeg_pipeline.metrics`.
"""

from __future__ import annotations

from typing import Any

from eeg_pipeline.metrics import compare_pre_post_quality as _compare_pre_post_quality
from eeg_pipeline.metrics import compute_quality_metrics as _compute_quality_metrics


def compute_quality_metrics(raw: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute EEG quality metrics for a recording."""

    return _compute_quality_metrics(raw, config)


def compare_pre_post_quality(before: Any, after: Any) -> dict[str, Any]:
    """Compare quality metrics before and after preprocessing."""

    return _compare_pre_post_quality(before, after)
