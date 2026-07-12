"""Input/output helpers for EEG recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.records import Recording, recording_from_path


SUPPORTED_FILE_SUFFIXES = {".edf", ".bdf", ".fif", ".fif.gz", ".set", ".vhdr"}
SUPPORTED_DIRECTORY_SUFFIXES = {".mff"}


def _has_suffix(path: Path, suffix: str) -> bool:
    return path.name.lower().endswith(suffix)


def discover_eeg_files(input_dir: str | Path) -> list[Path]:
    """Discover supported EEG recordings below a directory.

    MFF recordings are directories, so they are returned as directory paths. This function only
    inspects paths and does not read recording payloads.
    """

    root = Path(input_dir)
    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        lowered = path.name.lower()
        if path.is_dir() and any(
            lowered.endswith(suffix) for suffix in SUPPORTED_DIRECTORY_SUFFIXES
        ):
            matches.append(path)
        elif path.is_file() and any(lowered.endswith(suffix) for suffix in SUPPORTED_FILE_SUFFIXES):
            matches.append(path)
    return sorted(matches)


def discover_recordings(input_dir: str | Path) -> list[Recording]:
    """Discover supported recordings and return lightweight metadata."""

    return [recording_from_path(path) for path in discover_eeg_files(input_dir)]


def load_raw_eeg(file_path: str | Path, *, preload: bool = False, **kwargs: Any) -> Any:
    """Load a raw EEG recording with MNE-Python.

    The reader is selected from the path suffix. Private data are never loaded by discovery or
    inspection commands; this function is used only by explicit preprocessing commands.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    import mne

    name = path.name.lower()
    if path.is_dir() and _has_suffix(path, ".mff"):
        return mne.io.read_raw_egi(path, preload=preload, **kwargs)
    if _has_suffix(path, ".fif") or _has_suffix(path, ".fif.gz"):
        return mne.io.read_raw_fif(path, preload=preload, **kwargs)
    if _has_suffix(path, ".edf"):
        return mne.io.read_raw_edf(path, preload=preload, **kwargs)
    if _has_suffix(path, ".bdf"):
        return mne.io.read_raw_bdf(path, preload=preload, **kwargs)
    if _has_suffix(path, ".set"):
        return mne.io.read_raw_eeglab(path, preload=preload, **kwargs)
    if _has_suffix(path, ".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=preload, **kwargs)
    raise ValueError(f"Unsupported EEG file format for {path} ({name}).")


def inspect_recording_path(path: str | Path) -> dict[str, Any]:
    """Return path-level recording metadata without reading raw signal payloads."""

    recording = recording_from_path(path)
    return {
        "path": str(recording.path),
        "participant_id": recording.participant_id,
        "recording_id": recording.recording_id,
        "run_label": recording.run_label,
        "format": "mff" if recording.path.name.lower().endswith(".mff") else recording.path.suffix,
    }
