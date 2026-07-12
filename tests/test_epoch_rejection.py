"""Tests for epoch-level artifact rejection."""

from __future__ import annotations

import sys
import types
from typing import Any

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.epoch_rejection import (
    EpochRejectionConfig,
    reject_epochs,
    reject_epochs_autoreject,
    reject_epochs_by_peak_to_peak,
    reject_epochs_manual,
    summarize_epoch_rejection,
)
from eeg_pipeline.metrics import compute_epoch_rejection_quality_metrics


class FakeEpochs:
    """Small MNE Epochs-like object for synthetic rejection tests."""

    def __init__(self, amplitudes: list[float]) -> None:
        self.amplitudes = list(amplitudes)
        self.drop_log: tuple[tuple[str, ...], ...] = tuple(() for _ in amplitudes)
        self.info: dict[str, Any] = {}
        self.drop_bad_calls: list[dict[str, Any]] = []

    def copy(self) -> "FakeEpochs":
        copied = FakeEpochs(self.amplitudes)
        copied.drop_log = tuple(tuple(item) for item in self.drop_log)
        copied.info = dict(self.info)
        return copied

    def drop_bad(
        self,
        *,
        reject: dict[str, float] | None = None,
        flat: dict[str, float] | None = None,
    ) -> "FakeEpochs":
        self.drop_bad_calls.append({"reject": reject, "flat": flat})
        reject_threshold = None if reject is None else reject.get("eeg")
        flat_threshold = None if flat is None else flat.get("eeg")
        drop_log: list[tuple[str, ...]] = []
        for amplitude in self.amplitudes:
            reasons: list[str] = []
            if reject_threshold is not None and amplitude > reject_threshold:
                reasons.append("EEG")
            if flat_threshold is not None and amplitude < flat_threshold:
                reasons.append("EEG_FLAT")
            drop_log.append(tuple(reasons))
        self.drop_log = tuple(drop_log)
        return self

    def __len__(self) -> int:
        return sum(1 for item in self.drop_log if not item)


class FakeRejectLog:
    """AutoReject-like rejection log."""

    bad_epochs = [False, True, False, True]


class FakeAutoReject:
    """AutoReject-like estimator that records constructor settings."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        FakeAutoReject.calls.append(kwargs)

    def fit_transform(
        self, epochs: FakeEpochs, *, return_log: bool = False
    ) -> tuple[FakeEpochs, FakeRejectLog]:
        assert return_log
        cleaned = epochs.copy()
        cleaned.drop_log = ((), ("AUTOREJECT",), (), ("AUTOREJECT",))
        return cleaned, FakeRejectLog()


def test_manual_epoch_rejection_returns_detailed_statistics() -> None:
    epochs = FakeEpochs([2.0, 8.0, 1.0, 10.0])

    report = reject_epochs_manual(
        epochs,
        config=EpochRejectionConfig(reject={"eeg": 5.0}, flat={"eeg": 1.5}),
    )

    assert report.stats.n_epochs_before == 4
    assert report.stats.n_epochs_after == 1
    assert report.stats.n_epochs_dropped == 3
    assert report.stats.rejected_indices == (1, 2, 3)
    assert report.stats.reason_counts == {"EEG": 2, "EEG_FLAT": 1}
    assert report.provenance.method == "manual"
    assert report.quality_metrics["dropped_fraction"] == 0.75
    assert "epoch_rejection" in report.epochs.info["temp"]["eeg_pipeline"]


def test_reject_epochs_dispatches_manual_method() -> None:
    report = reject_epochs(FakeEpochs([1.0, 9.0]), config={"reject": {"eeg": 5.0}})

    assert report.stats.rejected_indices == (1,)


def test_peak_to_peak_compatibility_helper_returns_cleaned_epochs() -> None:
    cleaned = reject_epochs_by_peak_to_peak(FakeEpochs([1.0, 9.0]), eeg_threshold_v=5.0)

    assert summarize_epoch_rejection(cleaned) == {"total": 2, "dropped": 1, "retained": 1}


def test_autoreject_epoch_rejection_uses_optional_package(monkeypatch) -> None:
    FakeAutoReject.calls = []
    module = types.ModuleType("autoreject")
    module.AutoReject = FakeAutoReject
    monkeypatch.setitem(sys.modules, "autoreject", module)

    report = reject_epochs_autoreject(
        FakeEpochs([1.0, 2.0, 3.0, 4.0]),
        config={
            "method": "autoreject",
            "autoreject_n_interpolate": [1, 4],
            "autoreject_consensus": [0.5],
            "random_state": 7,
            "cv": 3,
            "n_jobs": 2,
        },
    )

    assert FakeAutoReject.calls[0]["n_interpolate"] == (1, 4)
    assert FakeAutoReject.calls[0]["consensus"] == (0.5,)
    assert report.stats.rejected_indices == (1, 3)
    assert report.stats.reason_counts == {"AUTOREJECT": 2}
    assert report.autoreject_log is not None
    assert report.provenance.method == "autoreject"


def test_epoch_rejection_quality_metrics_accept_stats() -> None:
    report = reject_epochs_manual(FakeEpochs([1.0, 9.0]), config={"reject": {"eeg": 5.0}})

    metrics = compute_epoch_rejection_quality_metrics(report.stats)

    assert metrics["n_epochs_before"] == 2
    assert metrics["n_epochs_after"] == 1
    assert metrics["n_epochs_dropped"] == 1


def test_default_config_exposes_epoch_rejection_settings() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.epoch_rejection.enabled
    assert config.epoch_rejection.method == "manual"
    assert config.epoch_rejection.reject == {}
