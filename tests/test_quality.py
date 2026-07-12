"""Tests for EEG quality metric stubs."""

from eeg_pipeline.quality_metrics import compute_quality_metrics


def test_compute_quality_metrics_none_returns_empty_dict() -> None:
    """The compatibility wrapper should tolerate missing data."""

    assert compute_quality_metrics(None) == {}
