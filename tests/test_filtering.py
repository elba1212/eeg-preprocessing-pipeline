"""Tests for the filtering stage."""

from __future__ import annotations

import copy
from typing import Any

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.filtering import (
    FirFilterConfig,
    NotchFilterConfig,
    filter_raw,
    make_filtered_copies,
)
from eeg_pipeline.line_noise import notch_frequencies


class RawLike:
    """Tiny Raw-like object that records MNE-style filter calls."""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {"sfreq": 250.0}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)

    def filter(self, **kwargs: Any) -> "RawLike":
        self.calls.append(("filter", kwargs))
        return self

    def notch_filter(self, **kwargs: Any) -> "RawLike":
        self.calls.append(("notch_filter", kwargs))
        return self


def test_make_filtered_copies_uses_analysis_and_ica_bands() -> None:
    raw = RawLike()

    copies = make_filtered_copies(raw, verbose=False)

    assert copies.analysis.raw is not raw
    assert copies.ica.raw is not raw
    assert copies.analysis.provenance.high_pass_hz == 0.1
    assert copies.analysis.provenance.low_pass_hz == 40.0
    assert copies.ica.provenance.high_pass_hz == 1.0
    assert copies.ica.provenance.low_pass_hz == 40.0
    assert copies.analysis.raw.calls[0][1]["l_freq"] == 0.1
    assert copies.ica.raw.calls[0][1]["l_freq"] == 1.0


def test_filter_raw_records_fir_and_notch_provenance() -> None:
    raw = RawLike()

    result = filter_raw(
        raw,
        0.1,
        40.0,
        copy_name="analysis",
        fir_config=FirFilterConfig(fir_window="hann", fir_design="firwin2"),
        notch_config=NotchFilterConfig(enabled=True, line_frequency_hz=50.0, harmonics=2),
        verbose=False,
    )

    assert [call[0] for call in result.raw.calls] == ["notch_filter", "filter"]
    assert result.raw.calls[0][1]["freqs"] == [50.0, 100.0]
    assert result.raw.calls[1][1]["fir_window"] == "hann"
    assert result.raw.calls[1][1]["fir_design"] == "firwin2"
    assert result.provenance.notch_frequencies_hz == (50.0, 100.0)
    assert result.raw.info["temp"]["eeg_pipeline"]["filtering"]["analysis"]["high_pass_hz"] == 0.1


def test_filter_raw_does_not_modify_input_object() -> None:
    raw = RawLike()

    result = filter_raw(raw, 0.1, 40.0, verbose=False)

    assert "temp" not in raw.info
    assert "temp" in result.raw.info


def test_filter_raw_validates_band_edges() -> None:
    raw = RawLike()

    try:
        filter_raw(raw, 40.0, 1.0, verbose=False)
    except ValueError as error:
        assert "high_pass_hz" in str(error)
    else:
        raise AssertionError("invalid band edges should raise ValueError")


def test_notch_frequency_helper_validates_harmonics() -> None:
    assert notch_frequencies(50.0, 3) == [50.0, 100.0, 150.0]


def test_default_config_exposes_filtering_decisions() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.filtering.analysis_high_pass_hz == 0.1
    assert config.filtering.ica_high_pass_hz == 1.0
    assert config.filtering.low_pass_hz == 40.0
    assert config.filtering.method == "fir"
    assert config.line_noise.frequency_hz == 50.0
