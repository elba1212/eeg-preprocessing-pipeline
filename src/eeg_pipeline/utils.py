"""Shared utility functions for configuration, paths, and logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eeg_pipeline.config import load_config_dict


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Parsed configuration values.
    """
    return load_config_dict(config_path)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
