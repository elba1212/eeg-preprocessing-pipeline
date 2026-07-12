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


@dataclass(frozen=True)
class LineNoiseConfig:
    """Line-noise handling settings."""

    enabled: bool = False
    frequency_hz: float = 50.0
    harmonics: int = 1


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


@dataclass(frozen=True)
class AnnotationConfig:
    """Continuous noisy-window annotation settings."""

    enabled: bool = True
    zscore_threshold: float = 6.0
    window_seconds: float = 2.0
    min_bad_fraction: float = 0.25


@dataclass(frozen=True)
class IcaConfig:
    """ICA fitting and component classification settings."""

    enabled: bool = True
    n_components: float | int | None = 0.99
    method: str = "fastica"
    random_state: int = 42
    max_iterations: int | str = "auto"
    classifier: str = "mne-icalabel"
    artifact_labels: tuple[str, ...] = ("eye blink", "heart beat", "muscle artifact")


@dataclass(frozen=True)
class EpochConfig:
    """Task epoch settings."""

    enabled: bool = False
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float | None, float | None] = (None, 0.0)


@dataclass(frozen=True)
class ReportConfig:
    """Quality-control report settings."""

    enabled: bool = True
    output_dir: Path = Path("reports")
    format: str = "html"


@dataclass(frozen=True)
class PipelineConfig:
    """Typed pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    line_noise: LineNoiseConfig = field(default_factory=LineNoiseConfig)
    montage: MontageConfig = field(default_factory=MontageConfig)
    bad_channels: BadChannelConfig = field(default_factory=BadChannelConfig)
    annotations: AnnotationConfig = field(default_factory=AnnotationConfig)
    ica: IcaConfig = field(default_factory=IcaConfig)
    epoching: EpochConfig = field(default_factory=EpochConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)


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
    annotations = _section(data, "annotations")
    ica = _section(data, "ica")
    epoching = _section(data, "epoching")
    reports = _section(data, "reports")

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
            ica_high_pass_hz=float(filtering.get("ica_high_pass_hz", FilterConfig.ica_high_pass_hz)),
            low_pass_hz=float(filtering.get("low_pass_hz", FilterConfig.low_pass_hz)),
            method=str(filtering.get("method", FilterConfig.method)),
        ),
        line_noise=LineNoiseConfig(
            enabled=bool(line_noise.get("enabled", LineNoiseConfig.enabled)),
            frequency_hz=float(line_noise.get("frequency_hz", LineNoiseConfig.frequency_hz)),
            harmonics=int(line_noise.get("harmonics", LineNoiseConfig.harmonics)),
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
        ),
        annotations=AnnotationConfig(
            enabled=bool(annotations.get("enabled", AnnotationConfig.enabled)),
            zscore_threshold=float(
                annotations.get("zscore_threshold", AnnotationConfig.zscore_threshold)
            ),
            window_seconds=float(annotations.get("window_seconds", AnnotationConfig.window_seconds)),
            min_bad_fraction=float(
                annotations.get("min_bad_fraction", AnnotationConfig.min_bad_fraction)
            ),
        ),
        ica=IcaConfig(
            enabled=bool(ica.get("enabled", IcaConfig.enabled)),
            n_components=ica.get("n_components", IcaConfig.n_components),
            method=str(ica.get("method", IcaConfig.method)),
            random_state=int(ica.get("random_state", IcaConfig.random_state)),
            max_iterations=ica.get("max_iterations", IcaConfig.max_iterations),
            classifier=str(ica.get("classifier", IcaConfig.classifier)),
            artifact_labels=tuple(ica.get("artifact_labels", IcaConfig.artifact_labels)),
        ),
        epoching=EpochConfig(
            enabled=bool(epoching.get("enabled", EpochConfig.enabled)),
            tmin=float(epoching.get("tmin", EpochConfig.tmin)),
            tmax=float(epoching.get("tmax", EpochConfig.tmax)),
            baseline=tuple(epoching.get("baseline", EpochConfig.baseline)),  # type: ignore[arg-type]
        ),
        reports=ReportConfig(
            enabled=bool(reports.get("enabled", ReportConfig.enabled)),
            output_dir=_path(reports.get("output_dir", ReportConfig.output_dir)),
            format=str(reports.get("format", ReportConfig.format)),
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
