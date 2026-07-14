"""High-level pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from eeg_pipeline.annotations import annotate_noisy_windows
from eeg_pipeline.bad_channels import detect_bad_channels
from eeg_pipeline.config import PipelineConfig, config_from_dict, load_pipeline_config
from eeg_pipeline.demo import (
    create_demo_events,
    create_synthetic_raw,
    demo_pipeline_overrides,
    save_psd_figure,
)
from eeg_pipeline.epoch_rejection import reject_epochs
from eeg_pipeline.epoching import create_epochs
from eeg_pipeline.filtering import FirFilterConfig, NotchFilterConfig, make_filtered_copies
from eeg_pipeline.ica import apply_ica, fit_ica
from eeg_pipeline.interpolation import interpolate_bad_channels
from eeg_pipeline.io import discover_recordings, inspect_recording_path, load_raw_eeg
from eeg_pipeline.reference import set_eeg_reference
from eeg_pipeline.reports import generate_recording_dashboard
from eeg_pipeline.resampling import resample_with_event_report
from eeg_pipeline.safety import run_safety_preflight


LOGGER = logging.getLogger(__name__)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_ready(value.to_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_with_overrides(config: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    return config_from_dict(_deep_update(asdict(config), overrides))


def _extract_events(raw: Any) -> np.ndarray | None:
    try:
        import mne

        events = mne.find_events(raw, shortest_event=1, verbose=False)
    except Exception:
        return None
    return events if len(events) else None


def _basic_psd_metrics(raw: Any) -> dict[str, Any]:
    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    psd = np.mean(np.abs(np.fft.rfft(data, axis=1)) ** 2, axis=0)
    return {
        "sfreq_hz": sfreq,
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "frequency_resolution_hz": float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0,
        "mean_power": float(np.mean(psd)),
    }


def _stage_config(config: Any) -> dict[str, Any]:
    return asdict(config)


class Pipeline:
    """Coordinate configuration, inspection, and preprocessing orchestration."""

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

    def _run_raw(
        self,
        raw: Any,
        *,
        output_dir: Path,
        recording_label: str,
        events: Any | None = None,
        event_id: dict[str, int] | None = None,
        metadata: Any = None,
        config: PipelineConfig | None = None,
    ) -> dict[str, Any]:
        """Run implemented preprocessing stages on an already-loaded raw object."""

        cfg = config or self.config
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        report_path = output_dir / "qc_report.html"
        provenance_path = output_dir / "provenance.json"

        LOGGER.info("Starting preprocessing for %s", recording_label)
        supplied_events = None if events is None else np.asarray(events, dtype=int)
        if supplied_events is None:
            supplied_events = _extract_events(raw)

        resampled = resample_with_event_report(
            raw,
            cfg.sampling.target_frequency_hz,
            supplied_events,
            verbose=False,
        )
        fir_config = FirFilterConfig(
            method=cfg.filtering.method,  # type: ignore[arg-type]
            phase=cfg.filtering.phase,
            fir_window=cfg.filtering.fir_window,
            fir_design=cfg.filtering.fir_design,
            filter_length=cfg.filtering.filter_length,
        )
        notch_config = NotchFilterConfig(
            enabled=cfg.line_noise.enabled,
            line_frequency_hz=cfg.line_noise.frequency_hz,
            harmonics=cfg.line_noise.harmonics,
            method=cfg.line_noise.method,  # type: ignore[arg-type]
        )
        filtered = make_filtered_copies(
            resampled.raw,
            analysis_high_pass_hz=cfg.filtering.analysis_high_pass_hz,
            ica_high_pass_hz=cfg.filtering.ica_high_pass_hz,
            low_pass_hz=cfg.filtering.low_pass_hz,
            fir_config=fir_config,
            notch_config=notch_config,
            verbose=False,
        )

        bad_channel_report = detect_bad_channels(
            filtered.analysis.raw,
            _stage_config(cfg.bad_channels),
        )
        analysis_for_interpolation = filtered.analysis.raw.copy()
        ica_for_interpolation = filtered.ica.raw.copy()
        analysis_for_interpolation.info["bads"] = list(bad_channel_report.final_bad_channels)
        ica_for_interpolation.info["bads"] = list(bad_channel_report.final_bad_channels)

        interpolation_report = interpolate_bad_channels(analysis_for_interpolation)
        ica_interpolation_report = interpolate_bad_channels(ica_for_interpolation)
        reference_report = set_eeg_reference(
            interpolation_report.raw,
            method=cfg.reference.method,
            projection=cfg.reference.projection,
        )
        ica_reference_report = set_eeg_reference(
            ica_interpolation_report.raw,
            method=cfg.reference.method,
            projection=cfg.reference.projection,
        )
        noisy_report = annotate_noisy_windows(reference_report.raw, _stage_config(cfg.annotations))
        ica_noisy_report = annotate_noisy_windows(
            ica_reference_report.raw,
            _stage_config(cfg.annotations),
        )
        ica_report = fit_ica(ica_noisy_report.raw, _stage_config(cfg.ica))
        cleaned_analysis = (
            apply_ica(noisy_report.raw, ica_report.ica, list(ica_report.removed_components))
            if ica_report.ica is not None
            else noisy_report.raw
        )

        selected_event_id = dict(event_id or cfg.epoching.event_id)
        epoch_report = None
        epoch_rejection_report = None
        if selected_event_id and resampled.events is not None:
            epoch_report = create_epochs(
                cleaned_analysis,
                resampled.events,
                selected_event_id,
                metadata=metadata,
                config=_stage_config(cfg.epoching),
            )
            if cfg.epoch_rejection.enabled:
                epoch_rejection_report = reject_epochs(
                    epoch_report.epochs,
                    config=_stage_config(cfg.epoch_rejection),
                )

        psd_metrics = _basic_psd_metrics(cleaned_analysis)
        psd_figure = save_psd_figure(cleaned_analysis, figures_dir / "psd.png")
        quality_metrics = {
            "n_bad_channels": len(bad_channel_report.final_bad_channels),
            "n_noisy_windows": sum(1 for decision in noisy_report.decisions if decision.is_noisy),
            "n_ica_removed_components": len(ica_report.removed_components),
            "epochs_created": epoch_report is not None,
            "epochs_rejected": (
                None
                if epoch_rejection_report is None
                else epoch_rejection_report.stats.n_epochs_dropped
            ),
        }
        preprocessing_summary = {
            "resampled_to_hz": cfg.sampling.target_frequency_hz,
            "analysis_filter_hz": [
                cfg.filtering.analysis_high_pass_hz,
                cfg.filtering.low_pass_hz,
            ],
            "ica_filter_hz": [cfg.filtering.ica_high_pass_hz, cfg.filtering.low_pass_hz],
            "event_handling": "epochs created" if epoch_report is not None else "epochs skipped",
        }
        dashboard = generate_recording_dashboard(
            report_path,
            recording_label=recording_label,
            metadata={
                "recording_label": recording_label,
                "sfreq_hz": float(cleaned_analysis.info["sfreq"]),
                "n_channels": len(cleaned_analysis.ch_names),
            },
            preprocessing_summary=preprocessing_summary,
            bad_channels=bad_channel_report,
            interpolation=interpolation_report,
            noisy_windows=noisy_report,
            ica=ica_report,
            psd={"metrics": psd_metrics, "figures": [psd_figure.name]},
            quality_metrics=quality_metrics,
            config={"use_mne_report": cfg.reports.use_mne_report},
        )

        summary: dict[str, Any] = {
            "recording_label": recording_label,
            "output_dir": output_dir,
            "provenance_path": provenance_path,
            "report_path": dashboard.output_path,
            "figures": [psd_figure],
            "preprocessing_summary": preprocessing_summary,
            "quality_metrics": quality_metrics,
            "stages": {
                "resampling": resampled.to_dict(),
                "filtering": filtered.provenance,
                "bad_channels": bad_channel_report.to_dict(),
                "interpolation": interpolation_report.to_dict(),
                "reference": reference_report.to_dict(),
                "noisy_windows": noisy_report.to_dict(),
                "ica": ica_report.to_dict(),
                "epoching": None if epoch_report is None else epoch_report.to_dict(),
                "epoch_rejection": (
                    None if epoch_rejection_report is None else epoch_rejection_report.to_dict()
                ),
            },
        }
        provenance_path.write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        LOGGER.info("Finished preprocessing for %s", recording_label)
        return _json_ready(summary)

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        events: Any | None = None,
        event_id: dict[str, int] | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        """Run implemented preprocessing stages for one recording.

        Use ``input_path="demo"`` to run the public synthetic demo pipeline. Private EEG files are
        loaded only when explicitly passed here.
        """

        if str(input_path).lower() == "demo":
            return self.run_demo(output_dir)

        path = Path(input_path)
        raw = load_raw_eeg(path, preload=True)
        destination = Path(output_dir) if output_dir is not None else self.config.data.output_dir
        return self._run_raw(
            raw,
            output_dir=destination / path.stem,
            recording_label=path.stem,
            events=events,
            event_id=event_id,
            metadata=metadata,
        )

    def run_demo(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Run the end-to-end pipeline on public synthetic EEG data."""

        raw = create_synthetic_raw()
        events, event_id, metadata = create_demo_events(raw)
        demo_config = _config_with_overrides(self.config, demo_pipeline_overrides())
        destination = Path(output_dir) if output_dir is not None else Path("outputs/demo")
        return self._run_raw(
            raw,
            output_dir=destination,
            recording_label="synthetic-demo",
            events=events,
            event_id=event_id,
            metadata=metadata,
            config=demo_config,
        )
