"""ICA fitting and ICLabel component classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class IcaStageConfig:
    """Configuration for ICA fitting and automatic component labeling."""

    enabled: bool = True
    method: str = "picard"
    n_components: float | int | None = 0.99
    random_state: int = 42
    max_iterations: int | str = "auto"
    reject_by_annotation: bool = True
    classifier: str = "iclabel"
    artifact_labels: tuple[str, ...] = ("eye blink", "heart beat", "muscle artifact")
    label_probability_threshold: float = 0.8
    require_ica_filter: bool = True
    require_annotations: bool = True

    def normalized_method(self) -> str:
        """Return the MNE ICA method name."""

        method = self.method.lower()
        if method not in {"picard", "infomax"}:
            raise ValueError("ICA method must be either 'picard' or 'infomax'.")
        return method

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable config mapping."""

        return asdict(self)


@dataclass(frozen=True)
class IcaComponentLabel:
    """ICLabel result for one ICA component."""

    component: int
    label: str
    probability: float
    remove: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable label mapping."""

        return asdict(self)


@dataclass(frozen=True)
class IcaProvenance:
    """Provenance for one ICA fit/classification run."""

    method: str
    n_components: float | int | None
    random_state: int
    max_iterations: int | str
    reject_by_annotation: bool
    classifier: str
    processing_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable provenance mapping."""

        return asdict(self)


@dataclass(frozen=True)
class IcaReport:
    """Structured output from ICA fitting and automatic classification."""

    ica: Any
    component_labels: tuple[IcaComponentLabel, ...]
    removed_components: tuple[int, ...]
    provenance: IcaProvenance
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report mapping without embedding the ICA object."""

        return {
            "component_labels": tuple(label.to_dict() for label in self.component_labels),
            "removed_components": self.removed_components,
            "provenance": self.provenance.to_dict(),
            "config": self.config,
        }


def _config_from_mapping(config: dict[str, Any] | IcaStageConfig | None) -> IcaStageConfig:
    if isinstance(config, IcaStageConfig):
        return config
    data = config or {}
    return IcaStageConfig(
        enabled=bool(data.get("enabled", IcaStageConfig.enabled)),
        method=str(data.get("method", IcaStageConfig.method)),
        n_components=data.get("n_components", IcaStageConfig.n_components),
        random_state=int(data.get("random_state", IcaStageConfig.random_state)),
        max_iterations=data.get("max_iterations", IcaStageConfig.max_iterations),
        reject_by_annotation=bool(
            data.get("reject_by_annotation", IcaStageConfig.reject_by_annotation)
        ),
        classifier=str(data.get("classifier", IcaStageConfig.classifier)),
        artifact_labels=tuple(data.get("artifact_labels", IcaStageConfig.artifact_labels)),
        label_probability_threshold=float(
            data.get(
                "label_probability_threshold",
                IcaStageConfig.label_probability_threshold,
            )
        ),
        require_ica_filter=bool(data.get("require_ica_filter", IcaStageConfig.require_ica_filter)),
        require_annotations=bool(
            data.get("require_annotations", IcaStageConfig.require_annotations)
        ),
    )


def _pipeline_metadata(raw: Any) -> dict[str, Any]:
    return raw.info.get("temp", {}).get("eeg_pipeline", {})


def validate_ica_input(raw: Any, config: IcaStageConfig) -> None:
    """Validate that ICA is fit only on the annotated 1-40 Hz ICA copy."""

    metadata = _pipeline_metadata(raw)
    if config.require_ica_filter:
        filtering = metadata.get("filtering", {}).get("ica")
        if not filtering:
            raise ValueError("ICA fitting requires the 1-40 Hz ICA filtered copy.")
        high_pass = filtering.get("high_pass_hz")
        low_pass = filtering.get("low_pass_hz")
        if high_pass != 1.0 or low_pass != 40.0:
            raise ValueError("ICA fitting requires filtering provenance with 1-40 Hz settings.")
    if config.require_annotations and "noisy_windows" not in metadata:
        raise ValueError("ICA fitting requires clean annotated data from the noisy-window stage.")


def _make_ica(config: IcaStageConfig) -> Any:
    from mne.preprocessing import ICA

    return ICA(
        n_components=config.n_components,
        method=config.normalized_method(),
        random_state=config.random_state,
        max_iter=config.max_iterations,
    )


def _fit_ica(raw: Any, config: IcaStageConfig) -> Any:
    ica = _make_ica(config)
    return ica.fit(raw, reject_by_annotation=config.reject_by_annotation)


def _label_components(raw: Any, ica: Any, config: IcaStageConfig) -> dict[str, Any]:
    try:
        from mne_icalabel import label_components
    except ImportError:
        return {"labels": (), "probabilities": ()}
    return label_components(raw, ica, method=config.classifier)


def _probability_for_component(probabilities: Any, index: int) -> float:
    if probabilities is None:
        return 0.0
    try:
        value = probabilities[index]
    except (IndexError, TypeError, KeyError):
        return 0.0
    if isinstance(value, (list, tuple)):
        return float(max(value)) if value else 0.0
    try:
        import numpy as np

        array = np.asarray(value)
        if array.ndim:
            return float(np.max(array))
    except Exception:
        pass
    return float(value)


def _component_labels(
    label_result: dict[str, Any], config: IcaStageConfig
) -> tuple[IcaComponentLabel, ...]:
    labels = tuple(str(label) for label in label_result.get("labels", ()))
    probabilities = label_result.get("y_pred_proba", label_result.get("probabilities", ()))
    artifact_labels = {label.lower() for label in config.artifact_labels}
    component_labels: list[IcaComponentLabel] = []
    for index, label in enumerate(labels):
        probability = _probability_for_component(probabilities, index)
        remove = (
            label.lower() in artifact_labels and probability >= config.label_probability_threshold
        )
        component_labels.append(
            IcaComponentLabel(
                component=index,
                label=label,
                probability=probability,
                remove=remove,
            )
        )
    return tuple(component_labels)


def _record_ica_provenance(raw: Any, report: IcaReport) -> None:
    temp = raw.info.setdefault("temp", {})
    eeg_pipeline = temp.setdefault("eeg_pipeline", {})
    eeg_pipeline["ica"] = report.to_dict()


def fit_ica(raw: Any, config: dict[str, Any] | IcaStageConfig | None = None) -> IcaReport:
    """Fit ICA on the annotated 1-40 Hz copy and classify components with ICLabel."""

    stage_config = _config_from_mapping(config)
    start = perf_counter()
    if not stage_config.enabled:
        provenance = IcaProvenance(
            method=stage_config.normalized_method(),
            n_components=stage_config.n_components,
            random_state=stage_config.random_state,
            max_iterations=stage_config.max_iterations,
            reject_by_annotation=stage_config.reject_by_annotation,
            classifier=stage_config.classifier,
            processing_time_seconds=perf_counter() - start,
        )
        return IcaReport(
            ica=None,
            component_labels=(),
            removed_components=(),
            provenance=provenance,
            config=stage_config.to_dict(),
        )

    validate_ica_input(raw, stage_config)
    ica = _fit_ica(raw, stage_config)
    label_result = _label_components(raw, ica, stage_config)
    labels = _component_labels(label_result, stage_config)
    removed_components = tuple(label.component for label in labels if label.remove)
    if hasattr(ica, "exclude"):
        ica.exclude = list(removed_components)
    provenance = IcaProvenance(
        method=stage_config.normalized_method(),
        n_components=stage_config.n_components,
        random_state=stage_config.random_state,
        max_iterations=stage_config.max_iterations,
        reject_by_annotation=stage_config.reject_by_annotation,
        classifier=stage_config.classifier,
        processing_time_seconds=perf_counter() - start,
    )
    report = IcaReport(
        ica=ica,
        component_labels=labels,
        removed_components=removed_components,
        provenance=provenance,
        config=stage_config.to_dict(),
    )
    _record_ica_provenance(raw, report)
    return report


def classify_ica_components(
    raw: Any,
    ica: Any,
    config: dict[str, Any] | IcaStageConfig | None = None,
) -> tuple[IcaComponentLabel, ...]:
    """Classify an already fitted ICA object with ICLabel."""

    stage_config = _config_from_mapping(config)
    return _component_labels(_label_components(raw, ica, stage_config), stage_config)


def apply_ica(raw: Any, ica: Any, exclude: list[int] | None = None) -> Any:
    """Apply an already fitted ICA object to a copy of raw data."""

    cleaned = raw.copy()
    if hasattr(ica, "exclude"):
        ica.exclude = list(exclude or getattr(ica, "exclude", []))
    return ica.apply(cleaned)
