"""Configuration models and loading helpers for the EEG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Input and output path settings."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    file_format: str = "auto"
    preload: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling-rate expectations and resampling target."""

    original_frequency_hz: float = 1000.0
    target_frequency_hz: float = 250.0


@dataclass(frozen=True)
class FilterConfig:
    """Continuous-data filter settings."""

    analysis_high_pass_hz: float = 0.1
    ica_high_pass_hz: float = 1.0
    low_pass_hz: float = 40.0
    method: str = "fir"
    phase: str = "zero"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    filter_length: str = "auto"


@dataclass(frozen=True)
class LineNoiseConfig:
    """Line-noise handling settings."""

    enabled: bool = False
    frequency_hz: float = 50.0
    harmonics: int = 1
    method: str = "fir"


@dataclass(frozen=True)
class MontageConfig:
    """EGI montage and channel validation settings."""

    name: str = "GSN-HydroCel-256"
    expected_eeg_channels: int = 256
    on_missing: str = "warn"


@dataclass(frozen=True)
class BadChannelConfig:
    """Global bad-channel detection settings."""

    enabled: bool = True
    method: str = "pyprep"
    random_state: int = 42
    deviation: bool = True
    correlation: bool = True
    high_frequency_noise: bool = True
    ransac: bool = True


@dataclass(frozen=True)
class ReferenceConfig:
    """EEG reference settings."""

    method: str = "average"
    projection: bool = False


@dataclass(frozen=True)
class AnnotationConfig:
    """Continuous noisy-window annotation settings."""

    enabled: bool = True
    zscore_threshold: float = 6.0
    window_seconds: float = 2.0
    min_bad_fraction: float = 0.25
    flat_peak_to_peak_threshold: float = 1e-12
    flat_variance_threshold: float = 1e-24
    clip_threshold: float | None = None
    clipped_fraction_threshold: float = 0.01
    description: str = "BAD_noisy_window"


@dataclass(frozen=True)
class IcaConfig:
    """ICA fitting and component classification settings."""

    enabled: bool = True
    n_components: float | int | None = 0.99
    method: str = "picard"
    random_state: int = 42
    max_iterations: int | str = "auto"
    reject_by_annotation: bool = True
    classifier: str = "iclabel"
    artifact_labels: tuple[str, ...] = ("eye blink", "heart beat", "muscle artifact")
    label_probability_threshold: float = 0.8
    require_ica_filter: bool = True
    require_annotations: bool = True


@dataclass(frozen=True)
class EpochConfig:
    """Task epoch settings."""

    enabled: bool = True
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] = (None, 0.0)
    preload: bool = True
    reject_by_annotation: bool = True
    event_id: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochRejectionConfig:
    """Epoch-level artifact rejection settings."""

    enabled: bool = True
    method: str = "manual"
    reject: dict[str, float] = field(default_factory=dict)
    flat: dict[str, float] = field(default_factory=dict)
    autoreject_n_interpolate: tuple[int, ...] | None = None
    autoreject_consensus: tuple[float, ...] | None = None
    random_state: int = 42
    cv: int = 10
    n_jobs: int = 1


@dataclass(frozen=True)
class ReportConfig:
    """Quality-control report settings."""

    enabled: bool = True
    output_dir: Path = Path("reports")
    format: str = "html"
    use_mne_report: bool = True
    redact_private_fields: bool = True
    include_empty_sections: bool = True


@dataclass(frozen=True)
class BatchConfig:
    """Dataset-level batch processing settings."""

    enabled: bool = True
    output_dir: Path = Path("outputs/batch")
    resume: bool = True
    n_jobs: int = 1
    parallel_backend: str = "thread"
    show_progress: bool = True
    summary_csv_name: str = "dataset_summary.csv"
    log_file_name: str = "batch.log"
    anonymize_identifiers: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Typed pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    line_noise: LineNoiseConfig = field(default_factory=LineNoiseConfig)
    montage: MontageConfig = field(default_factory=MontageConfig)
    bad_channels: BadChannelConfig = field(default_factory=BadChannelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    annotations: AnnotationConfig = field(default_factory=AnnotationConfig)
    ica: IcaConfig = field(default_factory=IcaConfig)
    epoching: EpochConfig = field(default_factory=EpochConfig)
    epoch_rejection: EpochRejectionConfig = field(default_factory=EpochRejectionConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section {key!r} must be a mapping.")
    return value


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def config_from_dict(data: dict[str, Any] | None) -> PipelineConfig:
    """Build a typed config from a nested mapping."""

    data = data or {}
    data_section = _section(data, "data")
    sampling = _section(data, "sampling")
    filtering = _section(data, "filtering")
    line_noise = _section(data, "line_noise")
    montage = _section(data, "montage")
    bad_channels = _section(data, "bad_channels")
    reference = _section(data, "reference")
    annotations = _section(data, "annotations")
    ica = _section(data, "ica")
    epoching = _section(data, "epoching")
    epoch_rejection = _section(data, "epoch_rejection")
    reports = _section(data, "reports")
    batch = _section(data, "batch")

    return PipelineConfig(
        data=DataConfig(
            input_dir=_path(data_section.get("input_dir", DataConfig.input_dir)),
            output_dir=_path(data_section.get("output_dir", DataConfig.output_dir)),
            file_format=str(data_section.get("file_format", DataConfig.file_format)),
            preload=bool(data_section.get("preload", DataConfig.preload)),
        ),
        sampling=SamplingConfig(
            original_frequency_hz=float(
                sampling.get("original_frequency_hz", SamplingConfig.original_frequency_hz)
            ),
            target_frequency_hz=float(
                sampling.get("target_frequency_hz", SamplingConfig.target_frequency_hz)
            ),
        ),
        filtering=FilterConfig(
            analysis_high_pass_hz=float(
                filtering.get("analysis_high_pass_hz", FilterConfig.analysis_high_pass_hz)
            ),
            ica_high_pass_hz=float(
                filtering.get("ica_high_pass_hz", FilterConfig.ica_high_pass_hz)
            ),
            low_pass_hz=float(filtering.get("low_pass_hz", FilterConfig.low_pass_hz)),
            method=str(filtering.get("method", FilterConfig.method)),
            phase=str(filtering.get("phase", FilterConfig.phase)),
            fir_window=str(filtering.get("fir_window", FilterConfig.fir_window)),
            fir_design=str(filtering.get("fir_design", FilterConfig.fir_design)),
            filter_length=str(filtering.get("filter_length", FilterConfig.filter_length)),
        ),
        line_noise=LineNoiseConfig(
            enabled=bool(line_noise.get("enabled", LineNoiseConfig.enabled)),
            frequency_hz=float(line_noise.get("frequency_hz", LineNoiseConfig.frequency_hz)),
            harmonics=int(line_noise.get("harmonics", LineNoiseConfig.harmonics)),
            method=str(line_noise.get("method", LineNoiseConfig.method)),
        ),
        montage=MontageConfig(
            name=str(montage.get("name", MontageConfig.name)),
            expected_eeg_channels=int(
                montage.get("expected_eeg_channels", MontageConfig.expected_eeg_channels)
            ),
            on_missing=str(montage.get("on_missing", MontageConfig.on_missing)),
        ),
        bad_channels=BadChannelConfig(
            enabled=bool(bad_channels.get("enabled", BadChannelConfig.enabled)),
            method=str(bad_channels.get("method", BadChannelConfig.method)),
            random_state=int(bad_channels.get("random_state", BadChannelConfig.random_state)),
            deviation=bool(bad_channels.get("deviation", BadChannelConfig.deviation)),
            correlation=bool(bad_channels.get("correlation", BadChannelConfig.correlation)),
            high_frequency_noise=bool(
                bad_channels.get("high_frequency_noise", BadChannelConfig.high_frequency_noise)
            ),
            ransac=bool(bad_channels.get("ransac", BadChannelConfig.ransac)),
        ),
        reference=ReferenceConfig(
            method=str(reference.get("method", ReferenceConfig.method)),
            projection=bool(reference.get("projection", ReferenceConfig.projection)),
        ),
        annotations=AnnotationConfig(
            enabled=bool(annotations.get("enabled", AnnotationConfig.enabled)),
            zscore_threshold=float(
                annotations.get("zscore_threshold", AnnotationConfig.zscore_threshold)
            ),
            window_seconds=float(
                annotations.get("window_seconds", AnnotationConfig.window_seconds)
            ),
            min_bad_fraction=float(
                annotations.get("min_bad_fraction", AnnotationConfig.min_bad_fraction)
            ),
            flat_peak_to_peak_threshold=float(
                annotations.get(
                    "flat_peak_to_peak_threshold",
                    AnnotationConfig.flat_peak_to_peak_threshold,
                )
            ),
            flat_variance_threshold=float(
                annotations.get(
                    "flat_variance_threshold",
                    AnnotationConfig.flat_variance_threshold,
                )
            ),
            clip_threshold=annotations.get("clip_threshold", AnnotationConfig.clip_threshold),
            clipped_fraction_threshold=float(
                annotations.get(
                    "clipped_fraction_threshold",
                    AnnotationConfig.clipped_fraction_threshold,
                )
            ),
            description=str(annotations.get("description", AnnotationConfig.description)),
        ),
        ica=IcaConfig(
            enabled=bool(ica.get("enabled", IcaConfig.enabled)),
            n_components=ica.get("n_components", IcaConfig.n_components),
            method=str(ica.get("method", IcaConfig.method)),
            random_state=int(ica.get("random_state", IcaConfig.random_state)),
            max_iterations=ica.get("max_iterations", IcaConfig.max_iterations),
            reject_by_annotation=bool(
                ica.get("reject_by_annotation", IcaConfig.reject_by_annotation)
            ),
            classifier=str(ica.get("classifier", IcaConfig.classifier)),
            artifact_labels=tuple(ica.get("artifact_labels", IcaConfig.artifact_labels)),
            label_probability_threshold=float(
                ica.get("label_probability_threshold", IcaConfig.label_probability_threshold)
            ),
            require_ica_filter=bool(ica.get("require_ica_filter", IcaConfig.require_ica_filter)),
            require_annotations=bool(ica.get("require_annotations", IcaConfig.require_annotations)),
        ),
        epoching=EpochConfig(
            enabled=bool(epoching.get("enabled", EpochConfig.enabled)),
            tmin=float(epoching.get("tmin", EpochConfig.tmin)),
            tmax=float(epoching.get("tmax", EpochConfig.tmax)),
            baseline=tuple(epoching.get("baseline", EpochConfig.baseline)),
            preload=bool(epoching.get("preload", EpochConfig.preload)),
            reject_by_annotation=bool(
                epoching.get("reject_by_annotation", EpochConfig.reject_by_annotation)
            ),
            event_id=dict(epoching.get("event_id", {})),
        ),
        epoch_rejection=EpochRejectionConfig(
            enabled=bool(epoch_rejection.get("enabled", EpochRejectionConfig.enabled)),
            method=str(epoch_rejection.get("method", EpochRejectionConfig.method)),
            reject=dict(epoch_rejection.get("reject", {})),
            flat=dict(epoch_rejection.get("flat", {})),
            autoreject_n_interpolate=(
                tuple(epoch_rejection["autoreject_n_interpolate"])
                if epoch_rejection.get("autoreject_n_interpolate") is not None
                else None
            ),
            autoreject_consensus=(
                tuple(epoch_rejection["autoreject_consensus"])
                if epoch_rejection.get("autoreject_consensus") is not None
                else None
            ),
            random_state=int(
                epoch_rejection.get("random_state", EpochRejectionConfig.random_state)
            ),
            cv=int(epoch_rejection.get("cv", EpochRejectionConfig.cv)),
            n_jobs=int(epoch_rejection.get("n_jobs", EpochRejectionConfig.n_jobs)),
        ),
        reports=ReportConfig(
            enabled=bool(reports.get("enabled", ReportConfig.enabled)),
            output_dir=_path(reports.get("output_dir", ReportConfig.output_dir)),
            format=str(reports.get("format", ReportConfig.format)),
            use_mne_report=bool(reports.get("use_mne_report", ReportConfig.use_mne_report)),
            redact_private_fields=bool(
                reports.get("redact_private_fields", ReportConfig.redact_private_fields)
            ),
            include_empty_sections=bool(
                reports.get("include_empty_sections", ReportConfig.include_empty_sections)
            ),
        ),
        batch=BatchConfig(
            enabled=bool(batch.get("enabled", BatchConfig.enabled)),
            output_dir=_path(batch.get("output_dir", BatchConfig.output_dir)),
            resume=bool(batch.get("resume", BatchConfig.resume)),
            n_jobs=max(1, int(batch.get("n_jobs", BatchConfig.n_jobs))),
            parallel_backend=str(batch.get("parallel_backend", BatchConfig.parallel_backend)),
            show_progress=bool(batch.get("show_progress", BatchConfig.show_progress)),
            summary_csv_name=str(batch.get("summary_csv_name", BatchConfig.summary_csv_name)),
            log_file_name=str(batch.get("log_file_name", BatchConfig.log_file_name)),
            anonymize_identifiers=bool(
                batch.get("anonymize_identifiers", BatchConfig.anonymize_identifiers)
            ),
        ),
    )


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML."""

    if config_path is None:
        return PipelineConfig()
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Pipeline configuration file must contain a YAML mapping.")
    return config_from_dict(data)


def load_config_dict(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config as a plain dictionary for legacy callers."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Configuration file must contain a YAML mapping.")
    return dict(data)
