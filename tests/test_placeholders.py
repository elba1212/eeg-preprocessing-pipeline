"""Tests for placeholder preprocessing modules."""

import pytest

from eeg_pipeline.metrics import compute_quality_metrics
from eeg_pipeline.pipeline import Pipeline


@pytest.mark.parametrize(
    "call",
    [
        lambda: compute_quality_metrics(None),
    ],
)
def test_preprocessing_modules_are_placeholders(call) -> None:
    """Preprocessing algorithms should not be implemented in this structural step."""

    with pytest.raises(NotImplementedError):
        call()


def test_pipeline_run_is_placeholder() -> None:
    """The high-level preprocessing run should require an existing recording."""

    with pytest.raises(FileNotFoundError):
        Pipeline().run("data/raw/example.mff")
