"""Independent component analysis helpers for artifact correction."""

from __future__ import annotations

from typing import Any


def fit_ica(raw: Any, config: dict[str, Any] | None = None) -> Any:
    """Fit an ICA model for artifact detection and correction."""

    from mne.preprocessing import ICA

    config = config or {}
    ica = ICA(
        n_components=config.get("n_components", 0.99),
        method=str(config.get("method", "fastica")),
        random_state=int(config.get("random_state", 42)),
        max_iter=config.get("max_iterations", "auto"),
    )
    return ica.fit(raw)


def classify_ica_components(raw: Any, ica: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify ICA components with mne-icalabel when available."""

    config = config or {}
    try:
        from mne_icalabel import label_components
    except ImportError:
        return {"available": False, "labels": [], "exclude": []}

    labels = label_components(raw, ica, method=str(config.get("classifier", "iclabel")))
    artifact_labels = set(config.get("artifact_labels", ("eye blink", "heart beat", "muscle artifact")))
    exclude = [
        index
        for index, label in enumerate(labels.get("labels", []))
        if str(label).lower() in artifact_labels
    ]
    return {"available": True, "labels": labels.get("labels", []), "exclude": exclude}


def apply_ica(raw: Any, ica: Any, exclude: list[int] | None = None) -> Any:
    """Apply an ICA solution to raw EEG data."""

    cleaned = raw.copy()
    ica.exclude = list(exclude or [])
    return ica.apply(cleaned)
