"""Automatic bad-channel detection using pyPREP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class BadChannelDetectionConfig:
    """Configuration for pyPREP bad-channel detection."""

    enabled: bool = True
    method: str = "pyprep"
    random_state: int = 42
    deviation: bool = True
    correlation: bool = True
    high_frequency_noise: bool = True
    ransac: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        return asdict(self)


@dataclass(frozen=True)
class BadChannelReport:
    """Structured output from automatic bad-channel detection."""

    detector: str
    detector_results: dict[str, tuple[str, ...]]
    final_bad_channels: tuple[str, ...]
    processing_time_seconds: float
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping."""

        return asdict(self)


_DETECTOR_STEPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "deviation": ("find_bad_by_deviation", ("bad_by_deviation",)),
    "correlation": ("find_bad_by_correlation", ("bad_by_correlation",)),
    "high_frequency_noise": (
        "find_bad_by_hfnoise",
        ("bad_by_hf_noise", "bad_by_hfnoise", "bad_by_high_frequency_noise"),
    ),
    "ransac": ("find_bad_by_ransac", ("bad_by_ransac",)),
}


def _config_from_mapping(
    config: dict[str, Any] | BadChannelDetectionConfig | None,
) -> BadChannelDetectionConfig:
    if isinstance(config, BadChannelDetectionConfig):
        return config
    data = config or {}
    return BadChannelDetectionConfig(
        enabled=bool(data.get("enabled", BadChannelDetectionConfig.enabled)),
        method=str(data.get("method", BadChannelDetectionConfig.method)),
        random_state=int(data.get("random_state", BadChannelDetectionConfig.random_state)),
        deviation=bool(data.get("deviation", BadChannelDetectionConfig.deviation)),
        correlation=bool(data.get("correlation", BadChannelDetectionConfig.correlation)),
        high_frequency_noise=bool(
            data.get("high_frequency_noise", BadChannelDetectionConfig.high_frequency_noise)
        ),
        ransac=bool(data.get("ransac", BadChannelDetectionConfig.ransac)),
    )


def _make_noisy_channels(raw: Any, random_state: int) -> Any:
    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except ImportError as error:
        raise ImportError(
            "pyPREP is required for automatic bad-channel detection. "
            "Install the optional pipeline dependencies with `pip install -e .[pipeline]`."
        ) from error

    try:
        return NoisyChannels(raw, random_state=random_state)
    except TypeError:
        detector = NoisyChannels(raw)
        if hasattr(detector, "random_state"):
            detector.random_state = random_state
        return detector


def _channels_from_detector(detector: Any, attribute_names: tuple[str, ...]) -> tuple[str, ...]:
    for attribute_name in attribute_names:
        value = getattr(detector, attribute_name, None)
        if value is not None:
            return tuple(str(channel) for channel in value)
    return ()


def _run_detector_step(detector: Any, step_name: str) -> None:
    method_name = _DETECTOR_STEPS[step_name][0]
    method = getattr(detector, method_name, None)
    if method is None:
        raise AttributeError(f"pyPREP NoisyChannels does not provide {method_name!r}.")
    method()


def _enabled_steps(config: BadChannelDetectionConfig) -> tuple[str, ...]:
    steps: list[str] = []
    if config.deviation:
        steps.append("deviation")
    if config.correlation:
        steps.append("correlation")
    if config.high_frequency_noise:
        steps.append("high_frequency_noise")
    if config.ransac:
        steps.append("ransac")
    return tuple(steps)


def detect_bad_channels(
    raw: Any,
    config: dict[str, Any] | BadChannelDetectionConfig | None = None,
) -> BadChannelReport:
    """Detect globally bad channels using the selected pyPREP detectors.

    The input object is not modified. The returned report records each detector-specific channel
    list, the union of final bad channels, and the elapsed processing time.
    """

    detection_config = _config_from_mapping(config)
    start = perf_counter()
    if not detection_config.enabled:
        return BadChannelReport(
            detector=detection_config.method,
            detector_results={},
            final_bad_channels=(),
            processing_time_seconds=perf_counter() - start,
            config=detection_config.to_dict(),
        )
    if detection_config.method.lower() != "pyprep":
        raise ValueError("Only pyPREP bad-channel detection is supported.")

    detector = _make_noisy_channels(raw, detection_config.random_state)
    detector_results: dict[str, tuple[str, ...]] = {}
    for step_name in _enabled_steps(detection_config):
        _run_detector_step(detector, step_name)
        attribute_names = _DETECTOR_STEPS[step_name][1]
        detector_results[step_name] = _channels_from_detector(detector, attribute_names)

    final_bad_channels = tuple(
        sorted({channel for values in detector_results.values() for channel in values})
    )
    return BadChannelReport(
        detector="pyprep",
        detector_results=detector_results,
        final_bad_channels=final_bad_channels,
        processing_time_seconds=perf_counter() - start,
        config=detection_config.to_dict(),
    )


def mark_bad_channels(raw: Any, bad_channels: list[str]) -> Any:
    """Mark channels as bad in a future stage."""

    _ = (raw, bad_channels)
    raise NotImplementedError("Bad-channel marking is not implemented yet.")
