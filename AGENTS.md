# AGENTS.md

## Project

This repository is an open-source Python package for automated preprocessing and quality control of wake EEG recordings collected during behavioral tasks.

The long-term goal is to create a production-quality preprocessing framework that is scientifically rigorous, modular, and reusable.

This project is **not** a research notebook.

Every design decision should prioritize:

* scientific correctness
* readability
* modularity
* reproducibility
* testability
* maintainability

---

# Data Safety

The repository is public.

Never:

* commit EEG recordings
* commit processed EEG
* commit participant identifiers
* commit reports
* commit figures
* commit notebook outputs
* commit local paths
* commit temporary files

The private recordings remain outside Git.

All unit tests must use synthetic MNE data.

---

# Coding Style

Use:

* Python 3.12+
* pathlib
* logging
* dataclasses
* type hints
* black formatting
* ruff-compatible style

Avoid:

* global variables
* large functions
* duplicated logic
* hidden side effects

Functions should generally perform one task.

---

# Scientific Principles

The package should preserve the preprocessing logic from the original notebook while making it reusable.

Scientific choices must never be changed without explanation.

Whenever there is uncertainty, explain the alternatives before implementing.

Prefer MNE best practices.

Do not silently modify EEG data.

Every transformation should be recorded in provenance metadata.

---

# Development Philosophy

Implement one processing stage at a time.

Do not begin implementing later stages before the current stage is stable.

Every completed stage must include:

* documentation
* unit tests
* logging
* configuration
* type hints

---

# Package Design

The pipeline should eventually support:

Load EEG

↓

Validate recording

↓

Filtering

↓

Resampling

↓

Bad channel detection

↓

Interpolation

↓

Average reference

↓

Noisy window annotation

↓

ICA

↓

Epoching

↓

Epoch rejection

↓

Quality metrics

↓

Dashboards

↓

Batch processing

Each stage should be implemented independently.

---

# Reports

The package should generate interactive quality-control reports similar in spirit to SleepEEGpy.

Every preprocessing decision should be transparent.

Never silently remove channels, ICA components, or epochs.

---

# Commits

Never commit automatically.

Never push automatically.

Always summarize the proposed changes before modifying multiple files.
