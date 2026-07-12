"""Tests for common-average referencing."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.metrics import compute_reference_quality_metrics
from eeg_pipeline.reference import (
    ReferenceConfig,
    ReferenceReport,
    set_eeg_reference,
    validate_bad_channels_interpolated,
)


class RawLike:
    """Synthetic Raw-like object for reference tests."""

    def __init__(
        self, *, bads: list[str] | None = None, interpolated: tuple[str, ...] = ()
    ) -> None:
        self.info: dict[str, Any] = {
            "bads": bads or [],
            "custom_ref_applied": False,
        }
        if interpolated:
            self.info["temp"] = {
                "eeg_pipeline": {
                    "interpolation": {
                        "interpolated_channels": interpolated,
                    }
                }
            }
        self.ch_names = ["E1", "E2", "E3"]
        self.data = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ]
        )
        self.reference_calls: list[dict[str, Any]] = []

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)

    def get_data(self, picks: str | None = None) -> np.ndarray:
        _ = picks
        return self.data

    def set_eeg_reference(self, ref_channels: str, projection: bool = False) -> "RawLike":
        self.reference_calls.append({"ref_channels": ref_channels, "projection": projection})
        if ref_channels == "average":
            self.data = self.data - np.mean(self.data, axis=0, keepdims=True)
        self.info["custom_ref_applied"] = True
        return self


def test_validate_bad_channels_requires_interpolation_provenance() -> None:
    try:
        validate_bad_channels_interpolated(RawLike(bads=["E2"]))
    except ValueError as error:
        assert "E2" in str(error)
    else:
        raise AssertionError("uninterpolated bad channels should block referencing")


def test_validate_bad_channels_accepts_interpolated_marked_bad_channels() -> None:
    validate_bad_channels_interpolated(RawLike(bads=["E2"], interpolated=("E2",)))


def test_set_eeg_reference_returns_report_and_preserves_input() -> None:
    raw = RawLike(bads=["E2"], interpolated=("E2",))

    report = set_eeg_reference(raw)

    assert isinstance(report, ReferenceReport)
    assert report.raw is not raw
    assert raw.info["custom_ref_applied"] is False
    assert report.raw.info["custom_ref_applied"] is True
    assert report.raw.reference_calls == [{"ref_channels": "average", "projection": False}]
    assert report.provenance.method == "average"
    assert report.provenance.processing_time_seconds >= 0.0
    assert report.raw.info["temp"]["eeg_pipeline"]["reference"]["method"] == "average"
    assert abs(report.after["global_mean"]) < 1e-12


def test_set_eeg_reference_accepts_common_average_alias_and_projection() -> None:
    report = set_eeg_reference(
        RawLike(),
        ReferenceConfig(method="common_average", projection=True),
    )

    assert report.raw.reference_calls == [{"ref_channels": "average", "projection": True}]
    assert report.config["method"] == "common_average"


def test_set_eeg_reference_rejects_custom_channels() -> None:
    try:
        set_eeg_reference(RawLike(), channels=["E1"])
    except ValueError as error:
        assert "Custom reference" in str(error)
    else:
        raise AssertionError("custom references should not be implemented yet")


def test_reference_quality_metrics_include_reference_state() -> None:
    metrics = compute_reference_quality_metrics(RawLike())

    assert metrics["n_bad_channels"] == 0
    assert metrics["bad_channels"] == ()
    assert metrics["custom_ref_applied"] is False
    assert "global_mean" in metrics


def test_default_config_exposes_reference_method() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.reference.method == "average"
    assert config.reference.projection is False
