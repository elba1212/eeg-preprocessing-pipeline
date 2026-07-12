"""Bad-channel interpolation for marked EEG channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InterpolationMetrics:
    """Before/after metrics for bad-channel interpolation."""

    bad_channels: tuple[str, ...]
    n_bad_channels: int
    n_channels: int
    channel_rms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable metrics mapping."""

        return asdict(self)


@dataclass(frozen=True)
class InterpolationProvenance:
    """Provenance for one interpolation operation."""

    method: str
    reset_bads: bool
    interpolated_channels: tuple[str, ...]
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class InterpolationReport:
    """Structured interpolation output."""

    raw: Any
    before: InterpolationMetrics
    after: InterpolationMetrics
    provenance: InterpolationProvenance
    figures: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding the raw object."""

        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "provenance": self.provenance.to_dict(),
            "figures": tuple(str(path) for path in self.figures),
        }


def _bad_channels(raw: Any) -> tuple[str, ...]:
    return tuple(str(channel) for channel in raw.info.get("bads", ()))


def _channel_rms(raw: Any) -> dict[str, float]:
    if not hasattr(raw, "get_data"):
        return {}
    data = raw.get_data(picks="eeg")
    channel_names = list(getattr(raw, "ch_names", ()))[: data.shape[0]]
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    return {name: float(value) for name, value in zip(channel_names, rms)}


def compute_interpolation_metrics(raw: Any) -> InterpolationMetrics:
    """Compute focused metrics needed to audit interpolation."""

    bad_channels = _bad_channels(raw)
    return InterpolationMetrics(
        bad_channels=bad_channels,
        n_bad_channels=len(bad_channels),
        n_channels=len(getattr(raw, "ch_names", ())),
        channel_rms=_channel_rms(raw),
    )


def _record_interpolation_provenance(raw: Any, provenance: InterpolationProvenance) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["interpolation"] = provenance.to_dict()


def interpolate_bad_channels(
    raw: Any, reset_bads: bool = False, *, mode: str = "accurate"
) -> InterpolationReport:
    """Interpolate channels already marked in ``raw.info["bads"]``.

    The input raw object is never modified. Average reference is intentionally not applied here.
    """

    before = compute_interpolation_metrics(raw)
    start = perf_counter()
    interpolated = raw.copy()
    if before.bad_channels:
        interpolated.interpolate_bads(reset_bads=reset_bads, mode=mode)
    after = compute_interpolation_metrics(interpolated)
    provenance = InterpolationProvenance(
        method="mne.io.Raw.interpolate_bads",
        reset_bads=reset_bads,
        interpolated_channels=before.bad_channels,
        processing_time_seconds=perf_counter() - start,
    )
    _record_interpolation_provenance(interpolated, provenance)
    return InterpolationReport(
        raw=interpolated,
        before=before,
        after=after,
        provenance=provenance,
    )


def _topomap_values(raw: Any) -> np.ndarray:
    metrics = compute_interpolation_metrics(raw)
    if not metrics.channel_rms:
        raise ValueError("Cannot plot interpolation topomap without channel-level data.")
    return np.array(
        [metrics.channel_rms[channel] for channel in raw.ch_names if channel in metrics.channel_rms]
    )


def save_interpolation_topomaps(
    before_raw: Any,
    after_raw: Any,
    output_path: str | Path,
    *,
    title: str = "Bad-channel interpolation",
) -> Path:
    """Save before/after channel-RMS topomap figures for interpolation QC."""

    import matplotlib.pyplot as plt
    import mne

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    before_values = _topomap_values(before_raw)
    after_values = _topomap_values(after_raw)
    before_info = before_raw.copy().pick("eeg").info
    after_info = after_raw.copy().pick("eeg").info

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle(title)
    mne.viz.plot_topomap(before_values, before_info, axes=axes[0], show=False)
    axes[0].set_title("Before")
    mne.viz.plot_topomap(after_values, after_info, axes=axes[1], show=False)
    axes[1].set_title("After")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
