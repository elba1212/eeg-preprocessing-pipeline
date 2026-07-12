"""Pipeline orchestration for EEG preprocessing workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.annotations import annotate_noisy_windows
from eeg_pipeline.bad_channels import detect_bad_channels, mark_bad_channels
from eeg_pipeline.config import PipelineConfig, load_pipeline_config
from eeg_pipeline.events import extract_events
from eeg_pipeline.filtering import make_filtered_copies
from eeg_pipeline.interpolation import interpolate_bad_channels
from eeg_pipeline.io import discover_recordings, inspect_recording_path, load_raw_eeg
from eeg_pipeline.line_noise import apply_notch_filter
from eeg_pipeline.metrics import compute_quality_metrics
from eeg_pipeline.montage import apply_egi_montage, validate_montage
from eeg_pipeline.reference import set_eeg_reference
from eeg_pipeline.reports import generate_report
from eeg_pipeline.resampling import resample_with_event_report
from eeg_pipeline.safety import run_safety_preflight


class Pipeline:
    """Coordinate EEG loading, preprocessing, quality control, and reporting.

    This class is intentionally minimal at the scaffold stage. Future development should add
    explicit step methods and structured state for raw data, intermediate outputs, metrics, and
    generated reports.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the pipeline with an optional configuration file."""
        self.config_path = Path(config_path) if config_path is not None else None
        self.config: PipelineConfig = load_pipeline_config(self.config_path)

    def inspect(self, input_path: str | Path) -> dict[str, Any]:
        """Inspect paths and configuration without reading raw signal data."""

        path = Path(input_path)
        if path.is_dir() and not path.name.lower().endswith(".mff"):
            recordings = [inspect_recording_path(recording.path) for recording in discover_recordings(path)]
            return {"input_path": str(path), "recordings": recordings, "n_recordings": len(recordings)}
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
        """Run the full EEG preprocessing pipeline.

        Args:
            input_path: Raw EEG file or directory to process.
            output_dir: Optional destination for processed data, reports, and figures.

        Returns:
            A dictionary describing generated outputs and quality metrics.
        """
        cfg = self.config
        output_root = Path(output_dir) if output_dir is not None else cfg.data.output_dir
        raw = load_raw_eeg(input_path, preload=cfg.data.preload)
        apply_egi_montage(raw, cfg.montage.name, cfg.montage.on_missing)
        validation = validate_montage(
            raw,
            expected_eeg_channels=cfg.montage.expected_eeg_channels,
            expected_sfreq_hz=cfg.sampling.original_frequency_hz,
        )
        events, event_id = extract_events(raw)
        resampled, resampled_events, timing = resample_with_event_report(
            raw,
            cfg.sampling.target_frequency_hz,
            events,
        )
        line_clean = apply_notch_filter(
            resampled,
            frequency_hz=cfg.line_noise.frequency_hz,
            harmonics=cfg.line_noise.harmonics,
            enabled=cfg.line_noise.enabled,
        )
        filtered = make_filtered_copies(
            line_clean,
            analysis_high_pass_hz=cfg.filtering.analysis_high_pass_hz,
            ica_high_pass_hz=cfg.filtering.ica_high_pass_hz,
            low_pass_hz=cfg.filtering.low_pass_hz,
            method=cfg.filtering.method,
        )
        bads = detect_bad_channels(
            filtered.analysis,
            {"method": cfg.bad_channels.method, "random_state": cfg.bad_channels.random_state},
        )
        analysis = mark_bad_channels(filtered.analysis, bads)
        analysis = interpolate_bad_channels(analysis)
        analysis = set_eeg_reference(analysis, method="average")
        annotated_ica = annotate_noisy_windows(
            filtered.ica,
            zscore_threshold=cfg.annotations.zscore_threshold,
            window_seconds=cfg.annotations.window_seconds,
            min_bad_fraction=cfg.annotations.min_bad_fraction,
        )
        metrics = compute_quality_metrics(analysis)
        metrics.update(
            {
                "input_path": str(input_path),
                "montage_ok": validation.ok,
                "montage_problems": "; ".join(validation.problems),
                "event_count": int(len(resampled_events)),
                "event_timing_ok": timing.ok,
                "event_id_count": len(event_id),
                "bad_channels": ", ".join(bads),
                "ica_annotation_count": len(annotated_ica.annotations),
            }
        )
        report_path: Path | None = None
        if cfg.reports.enabled:
            report_path = generate_report(metrics, output_root / "qc_report.html")
        return {"metrics": metrics, "report_path": str(report_path) if report_path else None}
