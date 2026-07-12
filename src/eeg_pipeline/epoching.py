"""Task epoch creation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def create_epochs(
    raw: Any,
    events: np.ndarray,
    event_id: dict[str, int],
    *,
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple[float | None, float | None] = (None, 0.0),
    preload: bool = True,
) -> Any:
    """Create MNE epochs from a preprocessed continuous recording."""

    import mne

    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
    )
