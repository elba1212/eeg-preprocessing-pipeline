"""Quality metrics for continuous and epoched EEG data."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_quality_metrics(raw: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute lightweight quality metrics for an MNE Raw object."""

    _ = config
    if raw is None:
        return {}
    picks = "eeg"
    data = raw.get_data(picks=picks)
    if data.size == 0:
        return {"sfreq_hz": float(raw.info["sfreq"]), "n_eeg_channels": 0}
    channel_std = np.std(data, axis=1)
    return {
        "sfreq_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1]) if len(raw.times) else 0.0,
        "n_channels": int(len(raw.ch_names)),
        "n_eeg_channels": int(data.shape[0]),
        "n_bad_channels": int(len(raw.info.get("bads", []))),
        "n_annotations": int(len(raw.annotations)),
        "median_channel_std": float(np.median(channel_std)),
        "max_channel_std": float(np.max(channel_std)),
    }


def compare_pre_post_quality(before: Any, after: Any) -> dict[str, Any]:
    """Compare core metrics before and after preprocessing."""

    before_metrics = compute_quality_metrics(before)
    after_metrics = compute_quality_metrics(after)
    return {"before": before_metrics, "after": after_metrics}
