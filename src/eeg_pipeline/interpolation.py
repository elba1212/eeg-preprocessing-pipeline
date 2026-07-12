"""Bad-channel interpolation utilities."""

from __future__ import annotations

from typing import Any


def interpolate_bad_channels(raw: Any, reset_bads: bool = False) -> Any:
    """Interpolate channels listed in the raw object's bad-channel metadata.

    Args:
        raw: Raw EEG object.
        reset_bads: Whether to clear the bad-channel list after interpolation.

    Returns:
        Interpolated raw data once implemented; ``None`` for now.
    """
    interpolated = raw.copy()
    if not interpolated.info.get("bads"):
        return interpolated
    return interpolated.interpolate_bads(reset_bads=reset_bads)
