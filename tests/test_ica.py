"""Tests for ICA fitting and ICLabel classification."""

from __future__ import annotations

import copy
import sys
import types
from typing import Any

from eeg_pipeline.config import load_pipeline_config
from eeg_pipeline.ica import IcaStageConfig, apply_ica, classify_ica_components, fit_ica


class RawLike:
    """Synthetic annotated 1-40 Hz raw-like object for ICA tests."""

    def __init__(
        self,
        *,
        include_filtering: bool = True,
        include_annotations: bool = True,
    ) -> None:
        eeg_pipeline: dict[str, Any] = {}
        if include_filtering:
            eeg_pipeline["filtering"] = {
                "ica": {
                    "high_pass_hz": 1.0,
                    "low_pass_hz": 40.0,
                }
            }
        if include_annotations:
            eeg_pipeline["noisy_windows"] = {
                "decisions": (),
                "provenance": {"method": "fixed_window_noisy_signal_metrics"},
            }
        self.info: dict[str, Any] = {"temp": {"eeg_pipeline": eeg_pipeline}}
        self.applied = False

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)


class FakeICA:
    """Fake MNE ICA object."""

    instances: list["FakeICA"] = []

    def __init__(
        self,
        n_components: float | int | None,
        method: str,
        random_state: int,
        max_iter: int | str,
    ) -> None:
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter
        self.fit_calls: list[dict[str, Any]] = []
        self.exclude: list[int] = []
        FakeICA.instances.append(self)

    def fit(self, raw: Any, reject_by_annotation: bool = True) -> "FakeICA":
        self.fit_calls.append({"raw": raw, "reject_by_annotation": reject_by_annotation})
        return self

    def apply(self, raw: RawLike) -> RawLike:
        raw.applied = True
        return raw


def _install_fake_mne(monkeypatch) -> None:
    FakeICA.instances = []
    mne = types.ModuleType("mne")
    preprocessing = types.ModuleType("mne.preprocessing")
    preprocessing.ICA = FakeICA
    mne.preprocessing = preprocessing
    monkeypatch.setitem(sys.modules, "mne", mne)
    monkeypatch.setitem(sys.modules, "mne.preprocessing", preprocessing)


def _install_fake_iclabel(monkeypatch) -> None:
    mne_icalabel = types.ModuleType("mne_icalabel")
    mne_icalabel.label_components = lambda raw, ica, method: {
        "labels": ["brain", "eye blink", "muscle artifact"],
        "y_pred_proba": [0.95, 0.91, 0.70],
    }
    monkeypatch.setitem(sys.modules, "mne_icalabel", mne_icalabel)


def test_fit_ica_uses_picard_and_rejects_annotations(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)
    _install_fake_iclabel(monkeypatch)
    raw = RawLike()

    report = fit_ica(raw, IcaStageConfig(method="picard"))

    ica = FakeICA.instances[0]
    assert ica.method == "picard"
    assert ica.fit_calls == [{"raw": raw, "reject_by_annotation": True}]
    assert report.provenance.method == "picard"
    assert report.removed_components == (1,)
    assert ica.exclude == [1]
    assert raw.info["temp"]["eeg_pipeline"]["ica"]["removed_components"] == (1,)


def test_fit_ica_supports_infomax(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)
    _install_fake_iclabel(monkeypatch)

    report = fit_ica(RawLike(), {"method": "infomax", "reject_by_annotation": False})

    ica = FakeICA.instances[0]
    assert ica.method == "infomax"
    assert ica.fit_calls[0]["reject_by_annotation"] is False
    assert report.provenance.method == "infomax"


def test_fit_ica_requires_ica_filtered_copy(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)

    try:
        fit_ica(RawLike(include_filtering=False))
    except ValueError as error:
        assert "1-40 Hz" in str(error)
    else:
        raise AssertionError("missing ICA filtering provenance should fail")


def test_fit_ica_requires_noisy_window_annotation_provenance(monkeypatch) -> None:
    _install_fake_mne(monkeypatch)

    try:
        fit_ica(RawLike(include_annotations=False))
    except ValueError as error:
        assert "clean annotated data" in str(error)
    else:
        raise AssertionError("missing annotation provenance should fail")


def test_classify_ica_components_returns_labels(monkeypatch) -> None:
    _install_fake_iclabel(monkeypatch)

    labels = classify_ica_components(RawLike(), FakeICA(0.99, "picard", 42, "auto"))

    assert [label.label for label in labels] == ["brain", "eye blink", "muscle artifact"]
    assert [label.remove for label in labels] == [False, True, False]


def test_apply_ica_works_on_copy() -> None:
    raw = RawLike()
    ica = FakeICA(0.99, "picard", 42, "auto")

    cleaned = apply_ica(raw, ica, exclude=[1, 2])

    assert cleaned is not raw
    assert cleaned.applied is True
    assert raw.applied is False
    assert ica.exclude == [1, 2]


def test_default_config_exposes_picard_ica() -> None:
    config = load_pipeline_config("config/default_config.yaml")

    assert config.ica.method == "picard"
    assert config.ica.classifier == "iclabel"
    assert config.ica.reject_by_annotation is True
    assert config.ica.label_probability_threshold == 0.8
