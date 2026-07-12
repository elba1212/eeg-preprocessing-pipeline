"""EEG re-referencing helpers."""

from __future__ import annotations

from typing import Any


def set_eeg_reference(
    raw: Any,
    method: str = "average",
    channels: list[str] | None = None,
) -> Any:
    """Apply an EEG reference strategy."""

    referenced = raw.copy()
    if method == "average":
        referenced.set_eeg_reference("average", projection=False)
    elif method == "custom":
        if not channels:
            raise ValueError("Custom reference requires at least one channel.")
        referenced.set_eeg_reference(ref_channels=channels, projection=False)
    else:
        raise ValueError(f"Unsupported reference method: {method}")
    return referenced
