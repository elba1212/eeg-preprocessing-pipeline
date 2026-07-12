"""Common-average EEG referencing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from eeg_pipeline.metrics import compute_reference_quality_metrics


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration for the referencing stage."""

    method: str = "average"
    projection: bool = False

    def normalized_method(self) -> str:
        """Normalize accepted common-average reference names."""

        method = self.method.lower().replace("-", "_")
        if method == "common_average":
            return "average"
        return method

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable config mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ReferenceProvenance:
    """Provenance for one reference operation."""

    method: str
    projection: bool
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class ReferenceReport:
    """Structured output from the referencing stage."""

    raw: Any
    before: dict[str, Any]
    after: dict[str, Any]
    provenance: ReferenceProvenance
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding the raw object."""

        return {
            "before": self.before,
            "after": self.after,
            "provenance": self.provenance.to_dict(),
            "config": self.config,
        }


def _config_from_args(method: str | ReferenceConfig, projection: bool) -> ReferenceConfig:
    if isinstance(method, ReferenceConfig):
        return method
    return ReferenceConfig(method=method, projection=projection)


def _interpolated_channels(raw: Any) -> tuple[str, ...]:
    interpolation = raw.info.get("temp", {}).get("eeg_pipeline", {}).get("interpolation", {})
    return tuple(str(channel) for channel in interpolation.get("interpolated_channels", ()))


def validate_bad_channels_interpolated(raw: Any) -> None:
    """Require marked bad channels to have interpolation provenance before referencing."""

    bad_channels = tuple(str(channel) for channel in raw.info.get("bads", ()))
    if not bad_channels:
        return
    interpolated = set(_interpolated_channels(raw))
    missing = tuple(channel for channel in bad_channels if channel not in interpolated)
    if missing:
        raise ValueError(
            "Cannot apply common-average reference before bad channels are interpolated: "
            + ", ".join(missing)
        )


def _record_reference_provenance(raw: Any, provenance: ReferenceProvenance) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["reference"] = provenance.to_dict()


def set_eeg_reference(
    raw: Any,
    method: str | ReferenceConfig = "average",
    channels: list[str] | None = None,
    *,
    projection: bool = False,
) -> ReferenceReport:
    """Apply common-average EEG reference to a copy of ``raw``.

    Bad channels must either be absent from ``raw.info["bads"]`` or have interpolation provenance.
    Custom channel references are intentionally not implemented in this stage.
    """

    if channels:
        raise ValueError("Custom reference channels are not implemented in this stage.")
    config = _config_from_args(method, projection)
    normalized_method = config.normalized_method()
    if normalized_method != "average":
        raise ValueError("Only common-average EEG reference is implemented.")

    validate_bad_channels_interpolated(raw)
    before = compute_reference_quality_metrics(raw)
    start = perf_counter()
    referenced = raw.copy()
    referenced.set_eeg_reference(ref_channels="average", projection=config.projection)
    after = compute_reference_quality_metrics(referenced)
    provenance = ReferenceProvenance(
        method="average",
        projection=config.projection,
        processing_time_seconds=perf_counter() - start,
    )
    _record_reference_provenance(referenced, provenance)
    return ReferenceReport(
        raw=referenced,
        before=before,
        after=after,
        provenance=provenance,
        config=config.to_dict(),
    )
