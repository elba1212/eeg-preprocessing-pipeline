"""Filtering transforms for continuous EEG data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal


FilterMethod = Literal["fir", "iir"]


@dataclass(frozen=True)
class FirFilterConfig:
    """Configurable MNE FIR band-pass settings."""

    method: FilterMethod = "fir"
    phase: str = "zero"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    filter_length: str = "auto"


@dataclass(frozen=True)
class NotchFilterConfig:
    """Optional line-noise notch settings."""

    enabled: bool = False
    line_frequency_hz: float = 50.0
    harmonics: int = 1
    method: FilterMethod = "fir"


@dataclass(frozen=True)
class FilterProvenance:
    """Filtering decisions applied to one raw copy."""

    copy_name: str
    high_pass_hz: float | None
    low_pass_hz: float | None
    notch_frequencies_hz: tuple[float, ...]
    fir_config: FirFilterConfig
    notch_config: NotchFilterConfig
    input_sfreq_hz: float
    output_sfreq_hz: float
    mne_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class FilteredRaw:
    """One filtered raw copy and the provenance for that copy."""

    raw: Any
    provenance: FilterProvenance


@dataclass(frozen=True)
class FilteredCopies:
    """Analysis and ICA filtered copies."""

    analysis: FilteredRaw
    ica: FilteredRaw
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validate_band(high_pass_hz: float | None, low_pass_hz: float | None) -> None:
    if high_pass_hz is not None and high_pass_hz < 0:
        raise ValueError("high_pass_hz must be non-negative or None.")
    if low_pass_hz is not None and low_pass_hz <= 0:
        raise ValueError("low_pass_hz must be positive or None.")
    if high_pass_hz is not None and low_pass_hz is not None and high_pass_hz >= low_pass_hz:
        raise ValueError("high_pass_hz must be lower than low_pass_hz.")


def _notch_frequencies(config: NotchFilterConfig) -> tuple[float, ...]:
    if not config.enabled:
        return ()
    if config.line_frequency_hz <= 0:
        raise ValueError("line_frequency_hz must be positive.")
    if config.harmonics < 1:
        raise ValueError("harmonics must be at least 1.")
    return tuple(config.line_frequency_hz * index for index in range(1, config.harmonics + 1))


def _record_filtering_provenance(raw: Any, provenance: FilterProvenance) -> None:
    """Store filtering provenance in the MNE temporary info namespace."""

    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    filtering = eeg_pipeline.setdefault("filtering", {})
    filtering[provenance.copy_name] = provenance.to_dict()


def _mne_version() -> str:
    try:
        return version("mne")
    except PackageNotFoundError:
        return "unknown"


def filter_raw(
    raw: Any,
    high_pass_hz: float | None,
    low_pass_hz: float | None,
    *,
    copy_name: str = "filtered",
    fir_config: FirFilterConfig | None = None,
    notch_config: NotchFilterConfig | None = None,
    picks: str | list[str] = "eeg",
    verbose: bool | str | int | None = None,
) -> FilteredRaw:
    """Create a filtered copy of an MNE Raw object.

    Filtering is applied to a copy, never in-place on the input object. Optional notch filtering is
    applied before the band-pass filter so line-frequency energy is attenuated before wider-band
    analysis filtering.
    """

    _validate_band(high_pass_hz, low_pass_hz)
    fir_config = fir_config or FirFilterConfig()
    notch_config = notch_config or NotchFilterConfig()

    filtered = raw.copy()
    input_sfreq = float(raw.info["sfreq"])
    freqs = _notch_frequencies(notch_config)
    if freqs:
        filtered.notch_filter(
            freqs=list(freqs),
            picks=picks,
            method=notch_config.method,
            phase=fir_config.phase,
            fir_window=fir_config.fir_window,
            fir_design=fir_config.fir_design,
            filter_length=fir_config.filter_length,
            verbose=verbose,
        )

    filtered.filter(
        l_freq=high_pass_hz,
        h_freq=low_pass_hz,
        picks=picks,
        method=fir_config.method,
        phase=fir_config.phase,
        fir_window=fir_config.fir_window,
        fir_design=fir_config.fir_design,
        filter_length=fir_config.filter_length,
        verbose=verbose,
    )

    provenance = FilterProvenance(
        copy_name=copy_name,
        high_pass_hz=high_pass_hz,
        low_pass_hz=low_pass_hz,
        notch_frequencies_hz=freqs,
        fir_config=fir_config,
        notch_config=notch_config,
        input_sfreq_hz=input_sfreq,
        output_sfreq_hz=float(filtered.info["sfreq"]),
        mne_version=_mne_version(),
    )
    _record_filtering_provenance(filtered, provenance)
    return FilteredRaw(raw=filtered, provenance=provenance)


def make_filtered_copies(
    raw: Any,
    *,
    analysis_high_pass_hz: float = 0.1,
    ica_high_pass_hz: float = 1.0,
    low_pass_hz: float = 40.0,
    fir_config: FirFilterConfig | None = None,
    notch_config: NotchFilterConfig | None = None,
    picks: str | list[str] = "eeg",
    verbose: bool | str | int | None = None,
) -> FilteredCopies:
    """Create the analysis and ICA filtered copies specified by the protocol."""

    analysis = filter_raw(
        raw,
        analysis_high_pass_hz,
        low_pass_hz,
        copy_name="analysis",
        fir_config=fir_config,
        notch_config=notch_config,
        picks=picks,
        verbose=verbose,
    )
    ica = filter_raw(
        raw,
        ica_high_pass_hz,
        low_pass_hz,
        copy_name="ica",
        fir_config=fir_config,
        notch_config=notch_config,
        picks=picks,
        verbose=verbose,
    )
    return FilteredCopies(
        analysis=analysis,
        ica=ica,
        provenance={
            "analysis": analysis.provenance.to_dict(),
            "ica": ica.provenance.to_dict(),
        },
    )
