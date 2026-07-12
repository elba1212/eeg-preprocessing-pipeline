"""Epoch-level artifact rejection for already-created task epochs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from eeg_pipeline.metrics import compute_epoch_rejection_quality_metrics


Thresholds = dict[str, float]


@dataclass(frozen=True)
class EpochRejectionConfig:
    """Configuration for epoch rejection."""

    enabled: bool = True
    method: str = "manual"
    reject: Thresholds = field(default_factory=dict)
    flat: Thresholds = field(default_factory=dict)
    autoreject_n_interpolate: tuple[int, ...] | None = None
    autoreject_consensus: tuple[float, ...] | None = None
    random_state: int = 42
    cv: int = 10
    n_jobs: int = 1
    verbose: bool | str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochRejectionStats:
    """Detailed epoch rejection statistics."""

    n_epochs_before: int
    n_epochs_after: int
    n_epochs_dropped: int
    dropped_fraction: float
    retained_fraction: float
    rejected_indices: tuple[int, ...]
    reason_counts: dict[str, int]
    drop_log: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable statistics mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochRejectionProvenance:
    """Provenance for an epoch rejection run."""

    method: str
    reject: Thresholds
    flat: Thresholds
    autoreject_n_interpolate: tuple[int, ...] | None
    autoreject_consensus: tuple[float, ...] | None
    random_state: int
    cv: int
    n_jobs: int
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochRejectionReport:
    """Structured output from epoch rejection."""

    epochs: Any
    stats: EpochRejectionStats
    provenance: EpochRejectionProvenance
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    autoreject_log: Any = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding epochs."""

        return {
            "stats": self.stats.to_dict(),
            "provenance": self.provenance.to_dict(),
            "quality_metrics": self.quality_metrics,
            "autoreject_log_present": self.autoreject_log is not None,
            "config": self.config,
        }


def _config_from_mapping(
    config: dict[str, Any] | EpochRejectionConfig | None,
) -> EpochRejectionConfig:
    if isinstance(config, EpochRejectionConfig):
        return config
    data = config or {}
    n_interpolate = data.get(
        "autoreject_n_interpolate",
        EpochRejectionConfig.autoreject_n_interpolate,
    )
    consensus = data.get("autoreject_consensus", EpochRejectionConfig.autoreject_consensus)
    return EpochRejectionConfig(
        enabled=bool(data.get("enabled", EpochRejectionConfig.enabled)),
        method=str(data.get("method", EpochRejectionConfig.method)),
        reject=_thresholds(data.get("reject", {})),
        flat=_thresholds(data.get("flat", {})),
        autoreject_n_interpolate=tuple(n_interpolate) if n_interpolate is not None else None,
        autoreject_consensus=tuple(consensus) if consensus is not None else None,
        random_state=int(data.get("random_state", EpochRejectionConfig.random_state)),
        cv=int(data.get("cv", EpochRejectionConfig.cv)),
        n_jobs=int(data.get("n_jobs", EpochRejectionConfig.n_jobs)),
        verbose=data.get("verbose", EpochRejectionConfig.verbose),
    )


def _thresholds(value: Any) -> Thresholds:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Epoch rejection thresholds must be mappings.")
    return {str(key): float(threshold) for key, threshold in value.items() if threshold is not None}


def _drop_log(epochs: Any) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(reason) for reason in item) for item in getattr(epochs, "drop_log", ()))


def _stats_from_drop_log(
    drop_log: tuple[tuple[str, ...], ...],
    *,
    n_epochs_before: int | None = None,
) -> EpochRejectionStats:
    total = len(drop_log) if n_epochs_before is None else n_epochs_before
    rejected_indices = tuple(index for index, item in enumerate(drop_log) if item)
    reason_counts: dict[str, int] = {}
    for item in drop_log:
        for reason in item:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    dropped = len(rejected_indices)
    retained = max(total - dropped, 0)
    dropped_fraction = float(dropped / total) if total else 0.0
    retained_fraction = float(retained / total) if total else 0.0
    return EpochRejectionStats(
        n_epochs_before=int(total),
        n_epochs_after=int(retained),
        n_epochs_dropped=int(dropped),
        dropped_fraction=dropped_fraction,
        retained_fraction=retained_fraction,
        rejected_indices=rejected_indices,
        reason_counts=reason_counts,
        drop_log=drop_log,
    )


def _stats_from_autoreject_log(
    epochs_before: Any, cleaned_epochs: Any, reject_log: Any
) -> EpochRejectionStats:
    bad_epochs = np.asarray(getattr(reject_log, "bad_epochs", ()), dtype=bool)
    if bad_epochs.size:
        autoreject_drop_log: tuple[tuple[str, ...], ...] = tuple(
            ("AUTOREJECT",) if bad else () for bad in bad_epochs
        )
        return _stats_from_drop_log(autoreject_drop_log, n_epochs_before=int(bad_epochs.size))

    cleaned_drop_log = _drop_log(cleaned_epochs)
    if cleaned_drop_log:
        return _stats_from_drop_log(cleaned_drop_log)

    before = len(epochs_before)
    after = len(cleaned_epochs)
    dropped = max(before - after, 0)
    synthetic_log: tuple[tuple[str, ...], ...] = tuple(
        ("AUTOREJECT",) if index < dropped else () for index in range(before)
    )
    return _stats_from_drop_log(synthetic_log, n_epochs_before=before)


def _record_epoch_rejection_provenance(report: EpochRejectionReport) -> None:
    info = getattr(report.epochs, "info", None)
    if not isinstance(info, dict):
        return
    temp = info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["epoch_rejection"] = report.to_dict()


def _provenance(
    config: EpochRejectionConfig, processing_time_seconds: float
) -> EpochRejectionProvenance:
    return EpochRejectionProvenance(
        method=config.method,
        reject=dict(config.reject),
        flat=dict(config.flat),
        autoreject_n_interpolate=config.autoreject_n_interpolate,
        autoreject_consensus=config.autoreject_consensus,
        random_state=config.random_state,
        cv=config.cv,
        n_jobs=config.n_jobs,
        processing_time_seconds=processing_time_seconds,
    )


def reject_epochs(
    epochs: Any,
    *,
    config: dict[str, Any] | EpochRejectionConfig | None = None,
) -> EpochRejectionReport:
    """Reject epochs with manual thresholds or AutoReject."""

    rejection_config = _config_from_mapping(config)
    if not rejection_config.enabled:
        raise ValueError("Epoch rejection is disabled by configuration.")

    method = rejection_config.method.lower().replace("-", "_")
    if method in {"manual", "threshold", "thresholds"}:
        return reject_epochs_manual(epochs, config=rejection_config)
    if method == "autoreject":
        return reject_epochs_autoreject(epochs, config=rejection_config)
    raise ValueError("Epoch rejection method must be 'manual' or 'autoreject'.")


def reject_epochs_manual(
    epochs: Any,
    *,
    config: dict[str, Any] | EpochRejectionConfig | None = None,
) -> EpochRejectionReport:
    """Reject epochs using explicit MNE reject and flat thresholds."""

    rejection_config = _config_from_mapping(config)
    cleaned = epochs.copy()
    start = perf_counter()
    cleaned.drop_bad(
        reject=rejection_config.reject or None,
        flat=rejection_config.flat or None,
    )
    processing_time = perf_counter() - start
    stats = _stats_from_drop_log(_drop_log(cleaned))
    provenance = _provenance(rejection_config, processing_time)
    quality_metrics = compute_epoch_rejection_quality_metrics(stats)
    report = EpochRejectionReport(
        epochs=cleaned,
        stats=stats,
        provenance=provenance,
        quality_metrics=quality_metrics,
        config=rejection_config.to_dict(),
    )
    _record_epoch_rejection_provenance(report)
    return report


def reject_epochs_autoreject(
    epochs: Any,
    *,
    config: dict[str, Any] | EpochRejectionConfig | None = None,
) -> EpochRejectionReport:
    """Reject or repair epochs using the optional AutoReject package."""

    rejection_config = _config_from_mapping(config)
    try:
        from autoreject import AutoReject
    except ImportError as error:
        raise ImportError(
            "AutoReject epoch rejection requires the optional 'autoreject' package."
        ) from error

    autoreject_kwargs = {
        "n_interpolate": rejection_config.autoreject_n_interpolate,
        "consensus": rejection_config.autoreject_consensus,
        "cv": rejection_config.cv,
        "random_state": rejection_config.random_state,
        "n_jobs": rejection_config.n_jobs,
        "verbose": rejection_config.verbose,
    }
    autoreject_kwargs = {
        key: value for key, value in autoreject_kwargs.items() if value is not None
    }
    estimator = AutoReject(**autoreject_kwargs)
    epochs_before = epochs.copy()
    start = perf_counter()
    cleaned, reject_log = estimator.fit_transform(epochs_before, return_log=True)
    processing_time = perf_counter() - start
    stats = _stats_from_autoreject_log(epochs, cleaned, reject_log)
    provenance = _provenance(rejection_config, processing_time)
    quality_metrics = compute_epoch_rejection_quality_metrics(stats)
    report = EpochRejectionReport(
        epochs=cleaned,
        stats=stats,
        provenance=provenance,
        quality_metrics=quality_metrics,
        autoreject_log=reject_log,
        config=rejection_config.to_dict(),
    )
    _record_epoch_rejection_provenance(report)
    return report


def reject_epochs_by_peak_to_peak(
    epochs: Any,
    *,
    eeg_threshold_v: float | None = None,
) -> Any:
    """Backward-compatible helper for manual EEG peak-to-peak thresholding."""

    config = EpochRejectionConfig(reject={"eeg": eeg_threshold_v} if eeg_threshold_v else {})
    return reject_epochs_manual(epochs, config=config).epochs


def summarize_epoch_rejection(epochs: Any) -> dict[str, int]:
    """Summarize retained and dropped epochs from an MNE-style drop log."""

    stats = _stats_from_drop_log(_drop_log(epochs))
    return {
        "total": stats.n_epochs_before,
        "dropped": stats.n_epochs_dropped,
        "retained": stats.n_epochs_after,
    }
