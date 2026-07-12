"""Automated EEG preprocessing and quality-control pipeline using MNE-Python."""

from eeg_pipeline.batch import BatchConfig, BatchReport, process_dataset, summarize_batch
from eeg_pipeline.config import PipelineConfig, load_pipeline_config
from eeg_pipeline.pipeline import Pipeline
from eeg_pipeline.reports import generate_participant_dashboard, generate_recording_dashboard

__version__ = "0.1.0"

__all__ = [
    "BatchConfig",
    "BatchReport",
    "Pipeline",
    "PipelineConfig",
    "__version__",
    "generate_participant_dashboard",
    "generate_recording_dashboard",
    "load_pipeline_config",
    "process_dataset",
    "summarize_batch",
]
