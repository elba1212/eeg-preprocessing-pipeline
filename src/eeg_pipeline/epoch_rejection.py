"""Epoch rejection helpers."""

from __future__ import annotations

from typing import Any


def reject_epochs_by_peak_to_peak(
    epochs: Any,
    *,
    eeg_threshold_v: float | None = None,
) -> Any:
    """Drop epochs exceeding a peak-to-peak EEG threshold."""

    cleaned = epochs.copy()
    if eeg_threshold_v is None:
        return cleaned
    return cleaned.drop_bad(reject={"eeg": eeg_threshold_v})


def summarize_epoch_rejection(epochs: Any) -> dict[str, int]:
    """Summarize retained and dropped epochs."""

    drop_log = getattr(epochs, "drop_log", ())
    dropped = sum(1 for item in drop_log if item)
    return {"total": len(drop_log), "dropped": dropped, "retained": len(drop_log) - dropped}
