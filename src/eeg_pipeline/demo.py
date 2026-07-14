"""Synthetic demo data and end-to-end demo runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def create_synthetic_raw(
    *,
    sfreq: float = 250.0,
    duration_seconds: float = 20.0,
    n_eeg_channels: int = 8,
    random_state: int = 42,
) -> Any:
    """Create a small public synthetic MNE Raw object for demos and tests."""

    import mne

    rng = np.random.default_rng(random_state)
    n_samples = int(round(sfreq * duration_seconds))
    times = np.arange(n_samples) / sfreq
    data = []
    for index in range(n_eeg_channels):
        alpha = np.sin(2 * np.pi * (8.0 + index * 0.3) * times)
        slow = 0.3 * np.sin(2 * np.pi * 1.0 * times)
        noise = 0.08 * rng.standard_normal(n_samples)
        data.append(1e-6 * (alpha + slow + noise))
    array = np.vstack(data)
    ch_names = [f"EEG{index + 1:03d}" for index in range(n_eeg_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(array, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    return raw


def create_demo_events(raw: Any) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    """Create deterministic synthetic behavioral events for the demo raw object."""

    sfreq = float(raw.info["sfreq"])
    onsets_seconds = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
    event_codes = np.array([1, 2, 1, 2, 1])
    events = np.column_stack(
        [
            np.rint(onsets_seconds * sfreq).astype(int),
            np.zeros(onsets_seconds.shape[0], dtype=int),
            event_codes,
        ]
    )
    metadata = [
        {"condition": "face" if code == 1 else "object", "trial": int(index + 1)}
        for index, code in enumerate(event_codes)
    ]
    return events, {"face": 1, "object": 2}, metadata


def save_psd_figure(raw: Any, output_path: str | Path, *, title: str = "Synthetic EEG PSD") -> Path:
    """Save a simple PSD figure without embedding private data."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    psd = np.mean(np.abs(np.fft.rfft(data, axis=1)) ** 2, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(freqs, 10 * np.log10(psd + np.finfo(float).eps), color="#0f766e")
    ax.set_xlim(0, min(60, sfreq / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def demo_pipeline_overrides() -> dict[str, Any]:
    """Return safe demo overrides for optional stages that need extra dependencies."""

    return {
        "bad_channels": {"enabled": False},
        "ica": {"enabled": False},
        "epoch_rejection": {"enabled": True, "method": "manual", "reject": {}, "flat": {}},
        "reports": {"use_mne_report": False},
    }
