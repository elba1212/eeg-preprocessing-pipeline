"""Automated EEG preprocessing and quality-control pipeline using MNE-Python."""

from eeg_pipeline.config import PipelineConfig, load_pipeline_config
from eeg_pipeline.pipeline import Pipeline

__all__ = ["Pipeline", "PipelineConfig", "load_pipeline_config"]
__version__ = "0.1.0"
