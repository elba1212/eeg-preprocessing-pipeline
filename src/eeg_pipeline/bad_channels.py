"""Bad-channel detection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_bad_channels(raw: Any, config: dict[str, Any] | None = None) -> list[str]:
    """Detect noisy, flat, or otherwise unusable EEG channels.

    Args:
        raw: Raw EEG object.
        config: Optional bad-channel detection settings.

    Returns:
        Channel names marked as bad.
    """
    config = config or {}
    method = str(config.get("method", "pyprep")).lower()
    if method == "pyprep":
        try:
            from pyprep.find_noisy_channels import NoisyChannels

            noisy = NoisyChannels(raw, random_state=int(config.get("random_state", 42)))
            noisy.find_all_bads()
            return sorted(noisy.get_bads())
        except ImportError:
            method = "robust_zscore"

    data = raw.get_data(picks="eeg")
    names = raw.copy().pick("eeg").ch_names
    if data.size == 0:
        return []
    channel_std = np.std(data, axis=1)
    median = np.median(channel_std)
    mad = np.median(np.abs(channel_std - median)) or 1.0
    zscores = np.abs(channel_std - median) / (1.4826 * mad)
    flat = channel_std <= float(config.get("flat_std_threshold", 1e-15))
    noisy = zscores >= float(config.get("zscore_threshold", 8.0))
    return sorted(name for name, is_bad in zip(names, flat | noisy, strict=True) if is_bad)


def mark_bad_channels(raw: Any, bad_channels: list[str]) -> Any:
    """Mark detected bad channels on the raw EEG object."""

    existing = set(raw.info.get("bads", []))
    raw.info["bads"] = sorted(existing | set(bad_channels))
    return raw
