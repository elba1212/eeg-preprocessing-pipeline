"""Recording metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


PARTICIPANT_PATTERN = re.compile(r"(EL\d{4})", re.IGNORECASE)


@dataclass(frozen=True)
class Recording:
    """A single EEG recording discovered on disk."""

    path: Path
    participant_id: str
    recording_id: str
    run_label: str | None = None


def infer_participant_id(path: str | Path) -> str:
    """Infer a participant identifier from a path without reading private payloads."""

    path = Path(path)
    for part in reversed(path.parts):
        match = PARTICIPANT_PATTERN.search(part)
        if match:
            return match.group(1).upper()
    return "unknown"


def infer_run_label(path: str | Path) -> str | None:
    """Infer a task/run label from common recording-name conventions."""

    stem = Path(path).stem.lower()
    for label in ("dis1", "dis2", "dis3", "lea", "lea1"):
        if label in stem:
            return label.upper()
    return None


def recording_from_path(path: str | Path) -> Recording:
    """Create a lightweight recording descriptor from a file or MFF directory path."""

    path = Path(path)
    return Recording(
        path=path,
        participant_id=infer_participant_id(path),
        recording_id=path.stem,
        run_label=infer_run_label(path),
    )


def output_stem(recording: Recording) -> str:
    """Return a privacy-aware output stem for derived artifacts."""

    run = f"_{recording.run_label}" if recording.run_label else ""
    return f"{recording.participant_id}_{recording.recording_id}{run}"
