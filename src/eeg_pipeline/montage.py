"""Montage validation placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MontageValidation:
    """Planned channel and montage validation result."""

    ok: bool
    eeg_channel_count: int
    sampling_frequency_hz: float
    problems: tuple[str, ...]


def validate_montage(raw: Any, **kwargs: Any) -> MontageValidation:
    """Validate channel metadata once EGI handling is finalized."""

    _ = (raw, kwargs)
    raise NotImplementedError("Montage validation is not implemented yet.")


def apply_egi_montage(
    raw: Any, montage_name: str = "GSN-HydroCel-256", on_missing: str = "warn"
) -> Any:
    """Apply an EGI montage once channel naming policy is validated."""

    _ = (raw, montage_name, on_missing)
    raise NotImplementedError("EGI montage application is not implemented yet.")
