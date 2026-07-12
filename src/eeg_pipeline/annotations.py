"""Continuous noisy-window detection and annotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoisyWindowConfig:
    """Configuration for fixed-window noisy segment detection."""

    enabled: bool = True
    window_seconds: float = 2.0
    zscore_threshold: float = 6.0
    flat_peak_to_peak_threshold: float = 1e-12
    flat_variance_threshold: float = 1e-24
    clip_threshold: float | None = None
    clipped_fraction_threshold: float = 0.01
    description: str = "BAD_noisy_window"

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        return asdict(self)


@dataclass(frozen=True)
class WindowMetrics:
    """Metrics computed for one continuous data window."""

    peak_to_peak: float
    variance: float
    flat_signal: bool
    high_frequency_noise: float
    clipped_fraction: float
    abrupt_jump: float
    nan_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable metrics mapping."""

        return asdict(self)


@dataclass(frozen=True)
class WindowDecision:
    """Decision for one analyzed window."""

    index: int
    onset_seconds: float
    duration_seconds: float
    metrics: WindowMetrics
    reasons: tuple[str, ...]

    @property
    def is_noisy(self) -> bool:
        """Whether this window should be annotated as noisy."""

        return bool(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable decision mapping."""

        data = asdict(self)
        data["is_noisy"] = self.is_noisy
        return data


@dataclass(frozen=True)
class NoisyWindowProvenance:
    """Provenance for one noisy-window annotation run."""

    method: str
    processing_time_seconds: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class NoisyWindowReport:
    """Structured output from continuous noisy-window annotation."""

    raw: Any
    decisions: tuple[WindowDecision, ...]
    annotations: tuple[dict[str, Any], ...]
    provenance: NoisyWindowProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding the raw object."""

        return {
            "decisions": tuple(decision.to_dict() for decision in self.decisions),
            "annotations": self.annotations,
            "provenance": self.provenance.to_dict(),
        }


def _config_from_mapping(config: dict[str, Any] | NoisyWindowConfig | None) -> NoisyWindowConfig:
    if isinstance(config, NoisyWindowConfig):
        return config
    data = config or {}
    return NoisyWindowConfig(
        enabled=bool(data.get("enabled", NoisyWindowConfig.enabled)),
        window_seconds=float(data.get("window_seconds", NoisyWindowConfig.window_seconds)),
        zscore_threshold=float(data.get("zscore_threshold", NoisyWindowConfig.zscore_threshold)),
        flat_peak_to_peak_threshold=float(
            data.get("flat_peak_to_peak_threshold", NoisyWindowConfig.flat_peak_to_peak_threshold)
        ),
        flat_variance_threshold=float(
            data.get("flat_variance_threshold", NoisyWindowConfig.flat_variance_threshold)
        ),
        clip_threshold=data.get("clip_threshold", NoisyWindowConfig.clip_threshold),
        clipped_fraction_threshold=float(
            data.get("clipped_fraction_threshold", NoisyWindowConfig.clipped_fraction_threshold)
        ),
        description=str(data.get("description", NoisyWindowConfig.description)),
    )


def _robust_zscores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values, dtype=float)
    return np.abs(values - median) / (1.4826 * mad)


def _window_slices(n_samples: int, window_samples: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + window_samples, n_samples))
        for start in range(0, n_samples, window_samples)
        if min(start + window_samples, n_samples) > start
    ]


def _window_metrics(window: np.ndarray, config: NoisyWindowConfig) -> WindowMetrics:
    finite_window = np.where(np.isfinite(window), window, np.nan)
    peak_to_peak_by_channel = np.array(
        [
            (
                float(np.max(channel[np.isfinite(channel)]) - np.min(channel[np.isfinite(channel)]))
                if np.any(np.isfinite(channel))
                else 0.0
            )
            for channel in finite_window
        ]
    )
    variance_by_channel = np.array(
        [
            float(np.var(channel[np.isfinite(channel)])) if np.any(np.isfinite(channel)) else 0.0
            for channel in finite_window
        ]
    )
    diffs = (
        np.diff(finite_window, axis=1)
        if finite_window.shape[1] > 1
        else np.zeros_like(finite_window)
    )
    high_frequency_by_channel = np.array(
        [
            (
                float(np.sqrt(np.mean(np.square(channel[np.isfinite(channel)]))))
                if np.any(np.isfinite(channel))
                else 0.0
            )
            for channel in diffs
        ]
    )
    abrupt_jump_by_channel = (
        np.array(
            [
                (
                    float(np.max(np.abs(channel[np.isfinite(channel)])))
                    if np.any(np.isfinite(channel))
                    else 0.0
                )
                for channel in diffs
            ]
        )
        if diffs.size
        else np.zeros(window.shape[0])
    )
    if config.clip_threshold is None:
        clipped_fraction = 0.0
    else:
        clipped_fraction = float(np.nanmean(np.abs(finite_window) >= config.clip_threshold))
    nan_fraction = float(np.mean(~np.isfinite(window)))
    flat_signal = bool(
        np.any(peak_to_peak_by_channel <= config.flat_peak_to_peak_threshold)
        or np.any(variance_by_channel <= config.flat_variance_threshold)
    )
    return WindowMetrics(
        peak_to_peak=float(np.nanmax(peak_to_peak_by_channel)),
        variance=float(np.nanmax(variance_by_channel)),
        flat_signal=flat_signal,
        high_frequency_noise=float(np.nanmax(high_frequency_by_channel)),
        clipped_fraction=clipped_fraction,
        abrupt_jump=float(np.nanmax(abrupt_jump_by_channel)),
        nan_fraction=nan_fraction,
    )


def detect_noisy_windows(
    raw: Any,
    config: dict[str, Any] | NoisyWindowConfig | None = None,
) -> tuple[WindowDecision, ...]:
    """Compute noisy-window decisions without modifying the input raw object."""

    detection_config = _config_from_mapping(config)
    if detection_config.window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if not detection_config.enabled:
        return ()

    sfreq = float(raw.info["sfreq"])
    window_samples = max(1, int(round(detection_config.window_seconds * sfreq)))
    data = raw.get_data(picks="eeg")
    slices = _window_slices(data.shape[1], window_samples)
    metrics = tuple(
        _window_metrics(data[:, start:stop], detection_config) for start, stop in slices
    )

    peak_to_peak_z = _robust_zscores(np.array([item.peak_to_peak for item in metrics]))
    variance_z = _robust_zscores(np.array([item.variance for item in metrics]))
    high_frequency_z = _robust_zscores(np.array([item.high_frequency_noise for item in metrics]))
    abrupt_jump_z = _robust_zscores(np.array([item.abrupt_jump for item in metrics]))

    decisions: list[WindowDecision] = []
    for index, ((start, stop), item) in enumerate(zip(slices, metrics)):
        reasons: list[str] = []
        if peak_to_peak_z[index] > detection_config.zscore_threshold:
            reasons.append("peak_to_peak")
        if variance_z[index] > detection_config.zscore_threshold:
            reasons.append("variance")
        if item.flat_signal:
            reasons.append("flat_signal")
        if high_frequency_z[index] > detection_config.zscore_threshold:
            reasons.append("high_frequency_noise")
        if (
            item.clipped_fraction >= detection_config.clipped_fraction_threshold
            and item.clipped_fraction > 0
        ):
            reasons.append("clipped_samples")
        if abrupt_jump_z[index] > detection_config.zscore_threshold:
            reasons.append("abrupt_jumps")
        if item.nan_fraction > 0:
            reasons.append("nans")
        decisions.append(
            WindowDecision(
                index=index,
                onset_seconds=start / sfreq,
                duration_seconds=(stop - start) / sfreq,
                metrics=item,
                reasons=tuple(reasons),
            )
        )
    return tuple(decisions)


def _annotation_dicts(
    decisions: tuple[WindowDecision, ...],
    description: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "onset": decision.onset_seconds,
            "duration": decision.duration_seconds,
            "description": description,
            "reasons": decision.reasons,
        }
        for decision in decisions
        if decision.is_noisy
    )


def _add_annotations(raw: Any, annotations: tuple[dict[str, Any], ...]) -> None:
    if not annotations:
        return
    if hasattr(raw, "add_annotations"):
        raw.add_annotations(annotations)
        return
    onsets = [annotation["onset"] for annotation in annotations]
    durations = [annotation["duration"] for annotation in annotations]
    descriptions = [annotation["description"] for annotation in annotations]
    try:
        import mne

        new_annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )
        raw.set_annotations(raw.annotations + new_annotations)
    except ImportError:
        raise


def _record_annotation_provenance(
    raw: Any,
    provenance: NoisyWindowProvenance,
    decisions: tuple[WindowDecision, ...],
) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["noisy_windows"] = {
        "provenance": provenance.to_dict(),
        "decisions": tuple(decision.to_dict() for decision in decisions),
    }


def annotate_noisy_windows(
    raw: Any,
    config: dict[str, Any] | NoisyWindowConfig | None = None,
) -> NoisyWindowReport:
    """Annotate noisy continuous windows without rejecting data."""

    detection_config = _config_from_mapping(config)
    start = perf_counter()
    annotated = raw.copy()
    decisions = detect_noisy_windows(raw, detection_config)
    annotations = _annotation_dicts(decisions, detection_config.description)
    _add_annotations(annotated, annotations)
    provenance = NoisyWindowProvenance(
        method="fixed_window_noisy_signal_metrics",
        processing_time_seconds=perf_counter() - start,
        config=detection_config.to_dict(),
    )
    _record_annotation_provenance(annotated, provenance, decisions)
    return NoisyWindowReport(
        raw=annotated,
        decisions=decisions,
        annotations=annotations,
        provenance=provenance,
    )
