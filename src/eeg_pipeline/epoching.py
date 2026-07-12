"""Task epoch creation without epoch rejection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Optional, Tuple

import numpy as np


Baseline = Optional[Tuple[Optional[float], Optional[float]]]


@dataclass(frozen=True)
class EpochConfig:
    """Configuration for task epoch creation."""

    enabled: bool = True
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Baseline = (None, 0.0)
    preload: bool = True
    reject_by_annotation: bool = True
    event_id: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochTimingReport:
    """Validation result for event timing and epoch boundaries."""

    ok: bool
    n_events: int
    invalid_event_indices: tuple[int, ...]
    first_epoch_start_seconds: float | None
    last_epoch_stop_seconds: float | None
    raw_duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable validation mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochProvenance:
    """Provenance for epoch creation."""

    tmin: float
    tmax: float
    baseline: Baseline
    event_id: dict[str, int]
    reject_by_annotation: bool
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpochReport:
    """Structured output from epoch creation."""

    epochs: Any
    timing: EpochTimingReport
    provenance: EpochProvenance
    metadata: Any = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding epochs."""

        return {
            "timing": self.timing.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metadata_present": self.metadata is not None,
            "config": self.config,
        }


def _config_from_mapping(config: dict[str, Any] | EpochConfig | None) -> EpochConfig:
    if isinstance(config, EpochConfig):
        return config
    data = config or {}
    baseline = data.get("baseline", EpochConfig.baseline)
    if baseline is not None:
        baseline = tuple(baseline)
    return EpochConfig(
        enabled=bool(data.get("enabled", EpochConfig.enabled)),
        tmin=float(data.get("tmin", EpochConfig.tmin)),
        tmax=float(data.get("tmax", EpochConfig.tmax)),
        baseline=baseline,
        preload=bool(data.get("preload", EpochConfig.preload)),
        reject_by_annotation=bool(
            data.get("reject_by_annotation", EpochConfig.reject_by_annotation)
        ),
        event_id=dict(data.get("event_id", {})),
    )


def _raw_duration_seconds(raw: Any) -> float:
    if hasattr(raw, "times") and len(raw.times):
        return float(raw.times[-1])
    n_times = getattr(raw, "n_times", None)
    if n_times is not None:
        return float(n_times - 1) / float(raw.info["sfreq"])
    raise ValueError("Raw object must expose times or n_times for epoch timing validation.")


def validate_epoch_timing(
    raw: Any,
    events: np.ndarray,
    *,
    tmin: float,
    tmax: float,
) -> EpochTimingReport:
    """Validate that requested epoch windows fit inside the raw recording."""

    if tmin >= tmax:
        raise ValueError("tmin must be earlier than tmax.")
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3).")

    sfreq = float(raw.info["sfreq"])
    event_times = events[:, 0].astype(float) / sfreq
    starts = event_times + tmin
    stops = event_times + tmax
    raw_duration = _raw_duration_seconds(raw)
    invalid = tuple(
        int(index)
        for index, (start, stop) in enumerate(zip(starts, stops))
        if start < 0 or stop > raw_duration
    )
    return EpochTimingReport(
        ok=not invalid,
        n_events=int(events.shape[0]),
        invalid_event_indices=invalid,
        first_epoch_start_seconds=float(np.min(starts)) if starts.size else None,
        last_epoch_stop_seconds=float(np.max(stops)) if stops.size else None,
        raw_duration_seconds=raw_duration,
    )


def _event_id(event_id: dict[str, int] | None, config: EpochConfig) -> dict[str, int]:
    mapping = dict(event_id or config.event_id)
    if not mapping:
        raise ValueError("event_id mapping is required for behavioral task epochs.")
    return mapping


def _record_epoch_provenance(raw: Any, report: EpochReport) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["epoching"] = report.to_dict()


def create_epochs(
    raw: Any,
    events: Any,
    event_id: dict[str, int] | None = None,
    *,
    metadata: Any = None,
    config: dict[str, Any] | EpochConfig | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    baseline: Baseline = None,
    preload: bool | None = None,
    reject_by_annotation: bool | None = None,
) -> EpochReport:
    """Create behavioral-task epochs without applying rejection."""

    epoch_config = _config_from_mapping(config)
    if not epoch_config.enabled:
        raise ValueError("Epoch creation is disabled by configuration.")
    chosen_tmin = epoch_config.tmin if tmin is None else tmin
    chosen_tmax = epoch_config.tmax if tmax is None else tmax
    chosen_baseline = epoch_config.baseline if baseline is None else baseline
    chosen_preload = epoch_config.preload if preload is None else preload
    chosen_reject_by_annotation = (
        epoch_config.reject_by_annotation if reject_by_annotation is None else reject_by_annotation
    )
    events_array = np.asarray(events, dtype=int)
    mapping = _event_id(event_id, epoch_config)
    timing = validate_epoch_timing(raw, events_array, tmin=chosen_tmin, tmax=chosen_tmax)
    if not timing.ok:
        raise ValueError(
            "Epoch timing is outside raw data bounds for event indices: "
            + ", ".join(str(index) for index in timing.invalid_event_indices)
        )

    import mne

    start = perf_counter()
    epochs = mne.Epochs(
        raw,
        events_array,
        event_id=mapping,
        tmin=chosen_tmin,
        tmax=chosen_tmax,
        baseline=chosen_baseline,
        metadata=metadata,
        preload=chosen_preload,
        reject=None,
        flat=None,
        reject_by_annotation=chosen_reject_by_annotation,
    )
    provenance = EpochProvenance(
        tmin=chosen_tmin,
        tmax=chosen_tmax,
        baseline=chosen_baseline,
        event_id=mapping,
        reject_by_annotation=chosen_reject_by_annotation,
        processing_time_seconds=perf_counter() - start,
    )
    report = EpochReport(
        epochs=epochs,
        timing=timing,
        provenance=provenance,
        metadata=metadata,
        config=epoch_config.to_dict(),
    )
    _record_epoch_provenance(raw, report)
    return report
