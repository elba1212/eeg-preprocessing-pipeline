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
