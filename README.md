# eeg-preprocessing-pipeline

An open-source Python package for automated EEG preprocessing and quality control using
[MNE-Python](https://mne.tools/).

This repository is a reusable Python package scaffold for incremental EEG preprocessing
development. The package is organized around transformations rather than notebook sections, with
path-only inspection and safety checks available before any private EEG recording is loaded.

## Data Privacy

Raw EEG recordings, processed EEG files, reports, figures, logs, and subject-level files are not
included in this repository. They are excluded for participant privacy and file-size reasons.

Keep study data in ignored local directories such as `data/raw/` and write generated files to
ignored locations such as `data/processed/`, `reports/`, and `figures/`. Before publishing changes,
confirm that no participant information, subject IDs, raw recordings, derived datasets, notebooks
with embedded outputs, or generated artifacts are staged for commit.

## Pipeline Architecture

```text
Raw EEG files
    |
    v
Data loading and metadata validation
    |
    v
Filtering, notch filtering, and resampling
    |
    v
Bad channel detection
    |
    v
Bad channel interpolation
    |
    v
Re-referencing
    |
    v
ICA artifact correction
    |
    v
Spectral analysis and quality metrics
    |
    v
Reports, figures, and processed EEG outputs
```

Core modules live under `src/eeg_pipeline/`:

```text
config.py            Typed configuration and defaults.
io.py                Path discovery and explicit MNE loading.
records.py           Recording and participant metadata helpers.
montage.py           EGI channel, montage, and sampling validation.
events.py            Event extraction and timing checks.
resampling.py        Early 1000 Hz to 250 Hz resampling.
filtering.py         Separate analysis and ICA filtered copies.
line_noise.py        Optional 50 Hz notch handling.
bad_channels.py      pyPREP bad-channel detection with fallback heuristics.
interpolation.py     Bad-channel interpolation.
reference.py         Common-average and custom references.
annotations.py       Noisy continuous-window annotations.
ica.py               ICA fitting, classification, and application.
epoching.py          Task epoch creation.
epoch_rejection.py   Epoch rejection summaries.
metrics.py           Quality metrics.
reports.py           HTML QC reports.
batch.py             Participant/session discovery.
pipeline.py          High-level orchestration.
cli.py               Command-line interface.
```

Compatibility modules such as `loader.py`, `rereference.py`, `quality_metrics.py`, and `report.py`
remain for older imports, but new code should use the transformation-specific modules above.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-org/eeg-preprocessing-pipeline.git
cd eeg-preprocessing-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Command-Line Usage

Run safety checks before doing anything with private data:

```bash
PYTHONPATH=src python -m eeg_pipeline.cli preflight .
```

Inspect available recording paths without loading EEG signals:

```bash
PYTHONPATH=src python -m eeg_pipeline.cli inspect data/raw
```

Run one recording only when you intentionally want to load private EEG data:

```bash
PYTHONPATH=src python -m eeg_pipeline.cli preprocess-recording "data/raw/path/to/recording.mff"
```

## Repository Structure

```text
config/                 Default YAML configuration.
data/raw/               Local raw EEG files, ignored by Git.
data/processed/         Generated preprocessing outputs, ignored by Git.
data/demo/              Small demo datasets or download instructions.
figures/                Generated figures, ignored by Git.
notebooks/              Exploratory and demonstration notebooks.
reports/                Generated quality-control reports, ignored by Git.
src/eeg_pipeline/       Package source code organized by transformation.
tests/                  Unit tests for package components.
examples/               Example scripts for common workflows.
docs/                   Project notes, including missing notebook reference status.
```

## Roadmap

- Restore or document the original scientific preprocessing notebook.
- Validate EGI/MFF loading, montage, event streams, and channel metadata.
- Harden pyPREP, noisy-window, ICA, epoching, and rejection settings against reviewed notebook logic.
- Add slow integration tests using public or generated data only.
- Add continuous integration, documentation, and release automation.

## License

This project is released under the MIT License.
