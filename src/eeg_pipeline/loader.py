"""Backward-compatible data-loading imports.

New code should import from :mod:`eeg_pipeline.io`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.io import discover_eeg_files as _discover_eeg_files
from eeg_pipeline.io import load_raw_eeg as _load_raw_eeg


def load_raw_eeg(file_path: str | Path, **kwargs: Any) -> Any:
    """Load a raw EEG recording with MNE-Python."""

    return _load_raw_eeg(file_path, **kwargs)


def discover_eeg_files(input_dir: str | Path) -> list[Path]:
    """Discover supported EEG files in a directory."""

    return _discover_eeg_files(input_dir)
