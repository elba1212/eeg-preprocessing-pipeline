"""Batch discovery and orchestration helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from eeg_pipeline.io import discover_recordings
from eeg_pipeline.records import Recording


def group_recordings_by_participant(input_dir: str | Path) -> dict[str, list[Recording]]:
    """Discover recordings and group them by participant identifier."""

    grouped: dict[str, list[Recording]] = defaultdict(list)
    for recording in discover_recordings(input_dir):
        grouped[recording.participant_id].append(recording)
    return dict(sorted(grouped.items()))


def summarize_batch(input_dir: str | Path) -> dict[str, int]:
    """Return path-level batch counts without loading raw EEG signals."""

    grouped = group_recordings_by_participant(input_dir)
    return {
        "participants": len(grouped),
        "recordings": sum(len(recordings) for recordings in grouped.values()),
    }
