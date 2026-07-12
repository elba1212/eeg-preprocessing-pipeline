"""Noisy continuous-window annotation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_noisy_windows(
    raw: Any,
    *,
    zscore_threshold: float = 6.0,
    window_seconds: float = 2.0,
    min_bad_fraction: float = 0.25,
    description: str = "BAD_noisy_window",
) -> Any:
    """Annotate high-amplitude windows using a simple robust z-score heuristic."""

    import mne

    annotated = raw.copy()
    data = annotated.get_data(picks="eeg")
    if data.size == 0:
        return annotated
    sfreq = float(annotated.info["sfreq"])
    window_samples = max(1, int(round(window_seconds * sfreq)))
    channel_scale = np.median(np.abs(data - np.median(data, axis=1, keepdims=True)), axis=1)
    channel_scale[channel_scale == 0] = np.nan
    robust_z = np.abs(data - np.nanmedian(data, axis=1, keepdims=True)) / (
        1.4826 * channel_scale[:, None]
    )
    bad_mask = np.nanmean(robust_z > zscore_threshold, axis=0) >= min_bad_fraction

    onsets: list[float] = []
    durations: list[float] = []
    start: int | None = None
    for index, is_bad in enumerate(bad_mask):
        if is_bad and start is None:
            start = index
        if start is not None and (not is_bad or index == len(bad_mask) - 1):
            stop = index if not is_bad else index + 1
            if stop - start >= window_samples:
                onsets.append(start / sfreq)
                durations.append((stop - start) / sfreq)
            start = None
    if onsets:
        annotated.set_annotations(
            annotated.annotations
            + mne.Annotations(onset=onsets, duration=durations, description=[description] * len(onsets))
        )
    return annotated
