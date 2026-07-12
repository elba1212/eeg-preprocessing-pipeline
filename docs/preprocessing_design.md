# Notebook Reference Status

The repository currently contains only `notebooks/demo.ipynb`, a one-cell placeholder notebook.
The original scientific preprocessing notebook is not present in this checkout.

Implementation should therefore treat the package modules as infrastructure and avoid encoding
study-specific scientific decisions until the reference notebook is added or its intended logic is
summarized in a reviewed document.

Open assumptions to validate from the missing notebook:

- MFF reader options, preload policy, channel naming, and EGI montage handling.
- Event streams, task event IDs, and how four recordings map to behavioral runs.
- Line-noise policy: diagnostic only, notch filtering, or conditional handling.
- pyPREP settings for 256-channel HydroCel recordings.
- Noisy-window thresholds and ICA fitting/classification decisions.
- Epoch timing, baseline, rejection thresholds, and output formats.

## Implemented Filtering Specification

This repository implements only the filtering stage so far.

- The filtering stage must create two independent filtered MNE `Raw` copies.
- The analysis copy uses a `0.1-40 Hz` band-pass.
- The ICA fitting copy uses a `1-40 Hz` band-pass.
- Filters use MNE FIR filtering by default.
- FIR options are configurable: method, phase, FIR window, FIR design, and filter length.
- Optional line-noise notch filtering is configurable and disabled by default.
- The default line frequency is `50 Hz`; harmonics are configurable.
- If notch filtering is enabled, it is applied before band-pass filtering.
- Filtering must not modify the input raw object in place.
- Each returned copy must include provenance describing the filtering decisions.

No ICA, epoching, or rejection logic is implemented in this stage.

## Implemented Bad-Channel Detection Specification

This repository implements automatic bad-channel detection using pyPREP.

- The detection stage must use pyPREP `NoisyChannels`.
- The supported detector families are deviation, correlation, high-frequency noise, and RANSAC.
- Each detector family must be independently configurable.
- The detection stage returns a structured `BadChannelReport`.
- The report records detector-specific channel lists, the final union of bad channels, the
  configuration used, and processing time.
- Detection must not interpolate channels.
- Detection must not silently mark channels as bad on the input object; that decision belongs to a
  later stage.

## Implemented Interpolation Specification

This repository implements channel interpolation only for channels already marked as bad.

- Interpolation must use MNE `Raw.interpolate_bads`.
- Interpolation must operate on a copy and must not modify the input raw object in place.
- The stage must preserve provenance including method, `reset_bads`, interpolated channels, and
  processing time.
- The stage must return before/after metrics including bad-channel counts and channel-level RMS.
- The stage may optionally reset `raw.info["bads"]` according to the explicit `reset_bads` argument.
- The stage must produce before/after topomap QC figures when requested.
- Average reference must not be applied during interpolation.

## Implemented Reference Specification

This repository implements common-average EEG referencing.

- Referencing must use MNE `Raw.set_eeg_reference(ref_channels="average")`.
- The reference method is configurable, but only common-average reference is implemented.
- Referencing must operate on a copy and must not modify the input raw object in place.
- Referencing must validate that bad channels have already been interpolated.
- The stage must preserve provenance including method, projection flag, and processing time.
- The stage must return before/after reference quality metrics.
- Custom references are not implemented in this stage.

## Implemented Noisy-Window Annotation Specification

This repository implements continuous noisy-window detection as annotations only.

- The window length is `2 seconds`.
- Windows must not be rejected or removed.
- The stage computes peak-to-peak, variance, flat signal, high-frequency noise, clipped samples,
  abrupt jumps, and NaNs for every window.
- Each window must receive a stored decision, including clean windows.
- Noisy windows must be annotated rather than dropped.
- The stage must copy the input raw object before adding annotations.
- The stage must preserve provenance including configuration, processing time, and all decisions.

## Implemented ICA Specification

This repository implements ICA fitting and automatic component labeling.

- ICA must be fit only on the annotated `1-40 Hz` ICA copy.
- ICA fitting must use annotated data with `reject_by_annotation=True` by default.
- Supported fitting algorithms are Picard and Infomax.
- The algorithm must be selected through configuration.
- ICLabel support must record component labels and probabilities.
- Components matching configured artifact labels above the probability threshold must be stored as
  removal decisions.
- The stage must preserve provenance including method, component settings, classifier, and
  processing time.
- Dashboards are not implemented in this stage.

## Implemented Epoch Creation Specification

This repository implements behavioral-task epoch creation without epoch rejection.

- Event mappings must be configurable and explicit.
- Baseline, `tmin`, `tmax`, preload, and annotation handling must be configurable.
- Epoch timing must be validated against the raw recording duration before epoch creation.
- Metadata must be preserved by passing it to MNE `Epochs`.
- Epoch rejection must not be applied in this stage.
- The stage must preserve provenance including epoch timing, baseline, event mapping, and
  processing time.

## Implemented Epoch Rejection Specification

This repository implements epoch-level rejection after behavioral-task epochs have already been
created.

- Supported methods are manual MNE thresholds and optional AutoReject.
- Manual rejection must use explicit configurable `reject` and `flat` mappings.
- AutoReject must be selected through configuration and imported lazily as an optional dependency.
- Default manual thresholds are empty because study-specific rejection cutoffs still require
  scientific validation.
- The stage must return detailed statistics including epoch counts, dropped fraction, rejected
  indices, reason counts, and the full drop log.
- The stage must preserve provenance including method, threshold settings, AutoReject settings, and
  processing time.
- The stage must update focused epoch-rejection quality metrics without treating global QC metrics
  as complete.

## Implemented Dashboard Specification

This repository implements modular HTML quality-control dashboards inspired by SleepEEGpy's
transparent reporting style.

- Recording dashboards must include metadata, preprocessing summary, bad channels, interpolation,
  noisy windows, ICA, PSD, and quality metrics.
- Participant dashboards must summarize all supplied recording-level summaries.
- Dashboard generation must not load or inspect private EEG recordings.
- MNE `Report` should be used when available and enabled, with standalone HTML as a fallback.
- Path-like fields and obvious participant/subject identifier fields must be redacted by default.
- Public dashboards should use anonymized recording and participant labels supplied by the caller.

## Implemented Batch Processing Specification

This repository implements dataset-level batch orchestration around recording-level processors.

- Batch processing must discover all supported recordings in an input directory without loading EEG
  payloads.
- Output directory, resume behavior, progress display, and parallelization must be configurable.
- Each recording must write a status marker so interrupted processing can resume safely.
- Batch logging must write to the configured output directory.
- A dataset summary CSV must be produced for every batch run.
- Participant and recording identifiers must be anonymized in batch keys by default.
- The batch layer must not hardcode local paths or scientific preprocessing assumptions.
