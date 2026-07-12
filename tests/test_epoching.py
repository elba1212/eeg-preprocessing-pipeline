"""Tests for behavioral-task epoch creation."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.epoching import EpochConfig, create_epochs, validate_epoch_timing


class RawLike:
    """Synthetic raw-like object for epoch tests."""

    def __init__(self, sfreq: float = 10.0, n_times: int = 100) -> None:
        self.info: dict[str, Any] = {"sfreq": sfreq}
        self.times = np.arange(n_times) / sfreq


class FakeEpochs:
    """Fake MNE Epochs object that records constructor arguments."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        FakeEpochs.calls.append({"args": args, "kwargs": kwargs})
        self.metadata = kwargs.get("metadata")


def _install_fake_mne(monkeypatch) -> None:
    FakeEpochs.calls = []
    mne = types.ModuleType("mne")
    mne.Epochs = FakeEpochs
    monkeypatch.setitem(sys.modules, "mne", mne)


def _events() -> np.ndarray:
    return np.array([[10, 0, 1], [30, 0, 2], [50, 0, 1]])


def test_validate_epoch_timing_accepts_in_bounds_events() -> None:
    report = validate_epoch_timing(RawLike(), _events(), tmin=-0.2, tmax=0.8)

    assert report.ok
    assert report.n_events == 3
    assert report.invalid_event_indices == ()
    assert report.first_epoch_start_seconds == 0.8
    assert report.last_epoch_stop_seconds == 5.8


def test_validate_epoch_timing_rejects_out_of_bounds_events() -> None:
    events = np.array([[1, 0, 1]])

    report = validate_epoch_timing(RawLike(), events, tmin=-0.2, tmax=0.8)

    assert not report.ok
    assert report.invalid_event_indices == (0,)


def test_create_epochs_preserves_metadata_and_configures_mne(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)
    raw = RawLike()
    metadata = [{"condition": "face"}, {"condition": "object"}, {"condition": "face"}]

    report = create_epochs(
        raw,
        _events(),
        {"face": 1, "object": 2},
        metadata=metadata,
        config=EpochConfig(tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0)),
    )

    kwargs = FakeEpochs.calls[0]["kwargs"]
    assert kwargs["event_id"] == {"face": 1, "object": 2}
    assert kwargs["baseline"] == (-0.1, 0.0)
    assert kwargs["metadata"] == metadata
    assert kwargs["reject"] is None
    assert kwargs["flat"] is None
    assert kwargs["reject_by_annotation"] is True
    assert report.metadata == metadata
    assert report.provenance.event_id == {"face": 1, "object": 2}
    assert raw.info["temp"]["eeg_pipeline"]["epoching"]["metadata_present"] is True


def test_create_epochs_uses_config_event_mapping(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)

    report = create_epochs(
        RawLike(),
        _events(),
        config={"event_id": {"target": 1, "distractor": 2}, "baseline": [None, 0.0]},
    )

    assert report.provenance.event_id == {"target": 1, "distractor": 2}


def test_create_epochs_requires_event_mapping(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)

    try:
        create_epochs(RawLike(), _events())
    except ValueError as error:
        assert "event_id mapping" in str(error)
    else:
        raise AssertionError("missing event mapping should fail")


def test_create_epochs_rejects_invalid_timing(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)

    try:
        create_epochs(RawLike(), np.array([[1, 0, 1]]), {"event": 1})
    except ValueError as error:
        assert "outside raw data bounds" in str(error)
    else:
        raise AssertionError("invalid epoch timing should fail")


def test_default_config_exposes_epoch_settings() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.epoching.enabled
    assert config.epoching.baseline == (None, 0.0)
    assert config.epoching.reject_by_annotation
