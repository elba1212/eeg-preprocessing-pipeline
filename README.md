# eeg-preprocessing-pipeline

An open-source Python package for automated EEG preprocessing and quality control using
[MNE-Python](https://mne.tools/).

This repository is a reusable Python package scaffold for incremental EEG preprocessing
development. The package is organized around transformations rather than notebook sections. Most
preprocessing modules are intentionally TODO placeholders until the scientific reference workflow
has been reviewed and translated into tested package code.

## Data Privacy

Raw EEG recordings, processed EEG files, reports, figures, logs, and subject-level files are not
included in this repository. They are excluded for participant privacy and file-size reasons.

Keep study data in ignored local directories such as `data/raw/` and write generated files to
ignored locations such as `data/processed/`, `reports/`, and `figures/`. Before publishing changes,
confirm that no participant information, subject IDs, raw recordings, derived datasets, notebooks
with embedded outputs, or generated artifacts are staged for commit.

Users must provide their own local EEG data or use a future public demo dataset once one is added.
Private research recordings, participant-level metadata, cleaned EEG files, cached arrays, reports,
figures, and logs should never be committed to this repository.

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
filtering.py         Analysis and ICA filtered copies with provenance.
line_noise.py        Optional configurable line-noise notch helpers.
bad_channels.py      pyPREP bad-channel detection reports.
interpolation.py     Bad-channel interpolation reports and topomap QC.
reference.py         Common-average reference reports.
annotations.py       Continuous noisy-window annotation reports.
ica.py               ICA fitting and ICLabel component classification.
epoching.py          Behavioral-task epoch creation.
epoch_rejection.py   Manual and AutoReject epoch rejection reports.
metrics.py           TODO placeholder for quality metrics.
reports.py           Recording and participant HTML QC dashboards.
batch.py             Resumable dataset-level batch orchestration.
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

Summarize a dataset by path inspection only:

```bash
PYTHONPATH=src python -m eeg_pipeline.cli batch-summary data/raw
```

Full preprocessing commands are intentionally not implemented yet.

## Filtering Decisions

The implemented filtering stage creates two independent copies of an MNE `Raw` object:

- Analysis copy: `0.1-40 Hz`.
- ICA copy: `1-40 Hz`.

Both copies use configurable MNE FIR filtering defaults from `config/default_config.yaml`
(`method`, `phase`, `fir_window`, `fir_design`, and `filter_length`). Optional line-noise notch
filtering can be enabled separately with a configurable line frequency, defaulting to `50 Hz`.
Notch filtering, when enabled, is applied before the band-pass filter. Filtering functions return
provenance metadata and also store it under `raw.info["temp"]["eeg_pipeline"]["filtering"]` on the
returned copies.

## Bad-Channel Detection Decisions

The implemented bad-channel stage wraps pyPREP and returns a structured `BadChannelReport`.

- Supported detectors: deviation, correlation, high-frequency noise, and RANSAC.
- Each detector can be enabled or disabled in `config/default_config.yaml`.
- The report records detector-specific channel lists, the final union of bad channels, the
  detector name, configuration, and processing time.
- Detection does not interpolate channels and does not mark channels on the input object.

## Interpolation Decisions

The implemented interpolation stage operates only on channels already marked in
`raw.info["bads"]`.

- The input `Raw` object is copied before interpolation.
- Interpolation uses MNE `Raw.interpolate_bads`.
- The stage returns an `InterpolationReport` containing before/after metrics and provenance.
- Metrics include bad-channel lists, bad-channel counts, channel count, and per-channel RMS.
- Topomap QC figures can be saved with `save_interpolation_topomaps`.
- Average reference is intentionally not applied in this stage.

## Reference Decisions

The implemented referencing stage applies common-average EEG reference only.

- The reference method is configurable as `average` or `common_average`.
- Bad channels must either be absent from `raw.info["bads"]` or already listed in interpolation
  provenance.
- The input `Raw` object is copied before referencing.
- The stage returns a `ReferenceReport` with before/after reference quality metrics and provenance.
- Custom channel references are intentionally not implemented yet.

## Noisy-Window Annotation Decisions

The implemented noisy-window stage annotates continuous data without rejecting samples.

- Fixed window length defaults to `2 seconds`.
- Each window records peak-to-peak, variance, flat-signal status, high-frequency noise,
  clipped-sample fraction, abrupt jumps, and NaN fraction.
- Noisy windows are annotated with `BAD_noisy_window`.
- Every window decision is stored in provenance, including clean windows.
- The input `Raw` object is copied before annotations are added.

## ICA Decisions

The implemented ICA stage fits ICA only on the annotated `1-40 Hz` ICA copy.

- Supported fitting algorithms: Picard and Infomax.
- The configured algorithm is selected in `config/default_config.yaml`.
- ICA fitting uses `reject_by_annotation=True` by default so annotated noisy windows are excluded.
- ICLabel is used when available to record component labels and probabilities.
- Components whose labels match configured artifact labels and exceed the probability threshold are
  stored as removal decisions.
- The stage records component labels, probabilities, removed components, and provenance.
- Dashboards are not created in this stage.

## Epoch Creation Decisions

The implemented epoching stage creates behavioral-task epochs without rejecting them.

- Event mappings are configurable and required.
- Baseline, `tmin`, `tmax`, preload, and annotation handling are configurable.
- Epoch timing is validated against raw recording bounds before calling MNE.
- Metadata is passed through to `mne.Epochs`.
- `reject` and `flat` are explicitly set to `None`; epoch rejection is a later stage.
- Epoch creation provenance is stored under `raw.info["temp"]["eeg_pipeline"]["epoching"]`.

## Epoch Rejection Decisions

The implemented epoch-rejection stage operates only on already-created epochs.

- Supported methods: manual MNE thresholds and optional AutoReject.
- Manual rejection uses explicit configurable `reject` and `flat` threshold mappings.
- AutoReject is optional and imported only when the `autoreject` method is selected.
- The default configuration uses manual mode with empty thresholds because reviewed
  study-specific cutoffs are not yet available.
- The stage returns detailed statistics: input epoch count, retained/dropped counts, dropped
  fraction, rejected indices, reason counts, and the full drop log.
- Rejection provenance and focused quality metrics are stored on the returned epochs when an
  MNE-style `info` mapping is available.

## Dashboard Decisions

The implemented reporting stage creates modular HTML quality-control dashboards inspired by
SleepEEGpy-style transparent preprocessing summaries.

- Recording dashboards include metadata, preprocessing summary, bad channels, interpolation,
  noisy windows, ICA, PSD, and quality-metric sections.
- Participant dashboards summarize all provided recording summaries.
- MNE `Report` is used when available and enabled; otherwise the package writes standalone HTML.
- Report generation is data-driven and does not load raw EEG recordings.
- Path-like fields and obvious participant/subject identifier fields are redacted by default.
- Callers should pass anonymized recording and participant labels for public or shared reports.

## Batch Processing Decisions

The implemented batch stage processes all discovered recordings through a caller-provided recording
processor function.

- Batch discovery uses path inspection only and does not load raw EEG payloads.
- Output directories are configurable and default to `outputs/batch`.
- Each recording gets a status marker so interrupted processing can resume.
- Progress display uses `tqdm` when available and falls back to normal iteration otherwise.
- Parallelization is configurable with serial or thread execution.
- A dataset summary CSV and batch log are written to the configured output directory.
- Participant and recording identifiers are anonymized in batch keys by default.
- The full scientific preprocessing pipeline is still intentionally separate from this batch
  orchestration layer.

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

## Release Checklist

- Run `pytest`.
- Run `ruff check .`.
- Run `black --check .`.
- Run `mypy src tests`.
- Run `python -m compileall src tests`.
- Run `PYTHONPATH=src python -m eeg_pipeline.cli preflight .`.
- Confirm `git ls-files data/raw` lists only `data/raw/.gitkeep`.
- Confirm generated outputs, reports, figures, logs, caches, and private data are unstaged.
- Review README, license, package metadata, and default configuration.
- Tag a release only after CI passes on a clean clone.

## License

This project is released under the MIT License.
