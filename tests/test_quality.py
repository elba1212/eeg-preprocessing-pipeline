"""Tests for EEG quality metric stubs."""

from eeg_pipeline.quality_metrics import compute_quality_metrics


def test_compute_quality_metrics_is_placeholder() -> None:
    """The compatibility wrapper should expose the placeholder state."""

    try:
        compute_quality_metrics(None)
    except NotImplementedError:
        return
    raise AssertionError("compute_quality_metrics should not be implemented yet")
