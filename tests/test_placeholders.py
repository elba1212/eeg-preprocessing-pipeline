"""Tests for placeholder preprocessing modules."""

import pytest

from eeg_pipeline.metrics import compute_quality_metrics
from eeg_pipeline.pipeline import Pipeline
from eeg_pipeline.resampling import resample_raw


@pytest.mark.parametrize(
    "call",
    [
        lambda: compute_quality_metrics(None),
        lambda: resample_raw(None, 250.0),
    ],
)
def test_preprocessing_modules_are_placeholders(call) -> None:
    """Preprocessing algorithms should not be implemented in this structural step."""

    with pytest.raises(NotImplementedError):
        call()


def test_pipeline_run_is_placeholder() -> None:
    """The high-level preprocessing run should remain unimplemented."""

    with pytest.raises(NotImplementedError):
        Pipeline().run("data/raw/example.mff")
