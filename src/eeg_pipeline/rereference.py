"""Backward-compatible re-reference imports.

New code should import from :mod:`eeg_pipeline.reference`.
"""

from __future__ import annotations

from eeg_pipeline.reference import set_eeg_reference


__all__ = ["set_eeg_reference"]
