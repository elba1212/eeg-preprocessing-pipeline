"""Tests for bad-channel interpolation."""

from __future__ import annotations

import copy
import shutil
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from eeg_pipeline.interpolation import (
    compute_interpolation_metrics,
    interpolate_bad_channels,
    save_interpolation_topomaps,
)


def _workspace_tmp() -> Path:
    path = Path("tests/_tmp") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


class RawLike:
    """Synthetic Raw-like object for interpolation tests."""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {"bads": ["E2"]}
        self.ch_names = ["E1", "E2", "E3"]
        self.data = np.array(
            [
                [1.0, 1.0, 1.0],
                [10.0, 10.0, 10.0],
                [2.0, 2.0, 2.0],
            ]
        )
        self.interpolate_calls: list[dict[str, Any]] = []

    def copy(self) -> "RawLike":
        return copy.deepcopy(self)

    def get_data(self, picks: str | None = None) -> np.ndarray:
        _ = picks
        return self.data

    def interpolate_bads(self, reset_bads: bool = False, mode: str = "accurate") -> "RawLike":
        self.interpolate_calls.append({"reset_bads": reset_bads, "mode": mode})
        self.data[1] = np.mean([self.data[0], self.data[2]], axis=0)
        if reset_bads:
            self.info["bads"] = []
        return self

    def pick(self, picks: str) -> "RawLike":
        _ = picks
        return self


def test_compute_interpolation_metrics_records_bad_channels_and_rms() -> None:
    metrics = compute_interpolation_metrics(RawLike())

    assert metrics.bad_channels == ("E2",)
    assert metrics.n_bad_channels == 1
    assert metrics.n_channels == 3
    assert metrics.channel_rms["E2"] == 10.0


def test_interpolate_bad_channels_returns_report_and_preserves_input() -> None:
    raw = RawLike()

    report = interpolate_bad_channels(raw, reset_bads=False)

    assert raw.info["bads"] == ["E2"]
    assert raw.data[1, 0] == 10.0
    assert report.raw is not raw
    assert report.before.bad_channels == ("E2",)
    assert report.after.bad_channels == ("E2",)
    assert report.after.channel_rms["E2"] == 1.5
    assert report.provenance.interpolated_channels == ("E2",)
    assert report.provenance.method == "mne.io.Raw.interpolate_bads"
    assert report.raw.info["temp"]["eeg_pipeline"]["interpolation"]["interpolated_channels"] == (
        "E2",
    )


def test_interpolate_bad_channels_can_reset_bads() -> None:
    report = interpolate_bad_channels(RawLike(), reset_bads=True, mode="fast")

    assert report.after.bad_channels == ()
    assert report.raw.interpolate_calls == [{"reset_bads": True, "mode": "fast"}]


def test_interpolate_bad_channels_skips_when_no_bad_channels() -> None:
    raw = RawLike()
    raw.info["bads"] = []

    report = interpolate_bad_channels(raw)

    assert report.raw.interpolate_calls == []
    assert report.before.n_bad_channels == 0
    assert report.after.n_bad_channels == 0


def test_save_interpolation_topomaps_uses_mne_topomap(
    monkeypatch,
) -> None:
    root = _workspace_tmp()
    calls: list[str] = []

    class AxisLike:
        def set_title(self, title: str) -> None:
            calls.append(f"title:{title}")

    class FigureLike:
        def suptitle(self, title: str) -> None:
            calls.append(f"suptitle:{title}")

        def savefig(self, output_path: Path, dpi: int) -> None:
            calls.append(f"savefig:{output_path.name}:{dpi}")
            output_path.write_text("figure", encoding="utf-8")

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *args, **kwargs: (FigureLike(), [AxisLike(), AxisLike()])
    pyplot.close = lambda fig: calls.append("close")
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    mne = types.ModuleType("mne")
    mne.viz = types.SimpleNamespace(
        plot_topomap=lambda values, info, axes, show: calls.append(
            f"topomap:{len(values)}:{show}:{bool(info)}"
        )
    )
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "mne", mne)

    try:
        output = save_interpolation_topomaps(RawLike(), RawLike(), root / "interp.png")

        assert output.exists()
        assert calls.count("topomap:3:False:True") == 2
        assert "title:Before" in calls
        assert "title:After" in calls
    finally:
        shutil.rmtree(root, ignore_errors=True)
