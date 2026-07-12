"""Legacy preprocessing facade placeholders."""

from __future__ import annotations

from typing import Any

from eeg_pipeline.filtering import filter_raw
from eeg_pipeline.line_noise import apply_notch_filter as notch_filter_raw


def preprocess_raw(raw: Any, config: dict[str, Any] | None = None) -> Any:
    """Run future preprocessing for legacy callers."""

    _ = (raw, config)
    raise NotImplementedError("Preprocessing is not implemented yet.")


__all__ = ["preprocess_raw", "filter_raw", "notch_filter_raw"]
