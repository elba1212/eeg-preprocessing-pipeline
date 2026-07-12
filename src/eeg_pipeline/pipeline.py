"""High-level pipeline orchestration placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.config import PipelineConfig, load_pipeline_config
from eeg_pipeline.io import discover_recordings, inspect_recording_path
from eeg_pipeline.safety import run_safety_preflight


class Pipeline:
    """Coordinate package configuration, inspection, and future preprocessing."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the pipeline with an optional configuration file."""

        self.config_path = Path(config_path) if config_path is not None else None
        self.config: PipelineConfig = load_pipeline_config(self.config_path)

    def inspect(self, input_path: str | Path) -> dict[str, Any]:
        """Inspect recording paths without reading EEG signal payloads."""

        path = Path(input_path)
        if path.is_dir() and not path.name.lower().endswith(".mff"):
            recordings = [
                inspect_recording_path(recording.path) for recording in discover_recordings(path)
            ]
            return {
                "input_path": str(path),
                "recordings": recordings,
                "n_recordings": len(recordings),
            }
        return inspect_recording_path(path)

    def preflight(self, root: str | Path = ".") -> dict[str, Any]:
        """Run repository safety checks."""

        report = run_safety_preflight(root)
        return {
            "git_ok": report.git_ok,
            "private_data_present": report.private_data_present,
            "tracked_private_paths": list(report.tracked_private_paths),
            "warnings": list(report.warnings),
        }

    def run(self, input_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Run future preprocessing once algorithms are implemented."""

        _ = (input_path, output_dir)
        raise NotImplementedError("Full EEG preprocessing is not implemented yet.")
