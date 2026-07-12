"""Quality-metric placeholders."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_reference_quality_metrics(raw: Any) -> dict[str, Any]:
    """Compute focused metrics for auditing EEG reference changes."""

    bad_channels = tuple(str(channel) for channel in raw.info.get("bads", ()))
    metrics: dict[str, Any] = {
        "n_bad_channels": len(bad_channels),
        "bad_channels": bad_channels,
        "custom_ref_applied": bool(raw.info.get("custom_ref_applied", False)),
    }
    if hasattr(raw, "get_data"):
        data = raw.get_data(picks="eeg")
        if data.size:
            metrics["mean_abs_channel_mean"] = float(np.mean(np.abs(np.mean(data, axis=1))))
            metrics["global_mean"] = float(np.mean(data))
    return metrics


def compute_epoch_rejection_quality_metrics(report_or_stats: Any) -> dict[str, Any]:
    """Compute focused metrics for auditing epoch rejection."""

    stats = getattr(report_or_stats, "stats", report_or_stats)
    n_before = int(getattr(stats, "n_epochs_before", 0))
    n_after = int(getattr(stats, "n_epochs_after", 0))
    n_dropped = int(getattr(stats, "n_epochs_dropped", max(n_before - n_after, 0)))
    reason_counts = dict(getattr(stats, "reason_counts", {}))
    dropped_fraction = getattr(stats, "dropped_fraction", None)
    if dropped_fraction is None:
        dropped_fraction = float(n_dropped / n_before) if n_before else 0.0
    return {
        "n_epochs_before": n_before,
        "n_epochs_after": n_after,
        "n_epochs_dropped": n_dropped,
        "dropped_fraction": float(dropped_fraction),
        "reason_counts": reason_counts,
    }


def compute_quality_metrics(raw: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute quality metrics once QC definitions are validated."""

    _ = (raw, config)
    raise NotImplementedError("Quality metrics are not implemented yet.")


def compare_pre_post_quality(before: Any, after: Any) -> dict[str, Any]:
    """Compare quality metrics once QC definitions are validated."""

    _ = (before, after)
    raise NotImplementedError("Quality metric comparison is not implemented yet.")
