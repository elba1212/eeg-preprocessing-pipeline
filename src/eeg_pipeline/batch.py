"""Batch discovery, resumable orchestration, and dataset summaries."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

from eeg_pipeline.io import discover_recordings
from eeg_pipeline.records import Recording


BatchProcessor = Callable[[Recording, Path], Optional[dict[str, Any]]]


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for dataset-level batch processing."""

    output_dir: Path = Path("outputs/batch")
    resume: bool = True
    n_jobs: int = 1
    parallel_backend: str = "thread"
    show_progress: bool = True
    summary_csv_name: str = "dataset_summary.csv"
    log_file_name: str = "batch.log"
    status_file_name: str = "status.json"
    anonymize_identifiers: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(frozen=True)
class BatchRecordingResult:
    """Result for one recording in a batch run."""

    recording_key: str
    participant_key: str
    run_label: str | None
    status: str
    output_dir: Path
    started_at: str | None
    finished_at: str | None
    processing_time_seconds: float
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result mapping."""

        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(frozen=True)
class BatchReport:
    """Structured result from processing a dataset."""

    input_dir: Path
    output_dir: Path
    summary_csv_path: Path
    log_path: Path
    total_recordings: int
    completed: int
    failed: int
    skipped: int
    results: tuple[BatchRecordingResult, ...]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable batch report."""

        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "summary_csv_path": str(self.summary_csv_path),
            "log_path": str(self.log_path),
            "total_recordings": self.total_recordings,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": [result.to_dict() for result in self.results],
            "config": self.config,
        }


def _config_from_mapping(config: dict[str, Any] | BatchConfig | None) -> BatchConfig:
    if isinstance(config, BatchConfig):
        return config
    data = config or {}
    return BatchConfig(
        output_dir=Path(data.get("output_dir", BatchConfig.output_dir)),
        resume=bool(data.get("resume", BatchConfig.resume)),
        n_jobs=max(1, int(data.get("n_jobs", BatchConfig.n_jobs))),
        parallel_backend=str(data.get("parallel_backend", BatchConfig.parallel_backend)),
        show_progress=bool(data.get("show_progress", BatchConfig.show_progress)),
        summary_csv_name=str(data.get("summary_csv_name", BatchConfig.summary_csv_name)),
        log_file_name=str(data.get("log_file_name", BatchConfig.log_file_name)),
        status_file_name=str(data.get("status_file_name", BatchConfig.status_file_name)),
        anonymize_identifiers=bool(
            data.get("anonymize_identifiers", BatchConfig.anonymize_identifiers)
        ),
    )


def group_recordings_by_participant(input_dir: str | Path) -> dict[str, list[Recording]]:
    """Discover recordings and group them by participant identifier."""

    grouped: dict[str, list[Recording]] = {}
    for recording in discover_recordings(input_dir):
        grouped.setdefault(recording.participant_id, []).append(recording)
    return dict(sorted(grouped.items()))


def summarize_batch(input_dir: str | Path) -> dict[str, int]:
    """Return path-level batch counts without loading raw EEG signals."""

    grouped = group_recordings_by_participant(input_dir)
    return {
        "participants": len(grouped),
        "recordings": sum(len(recordings) for recordings in grouped.values()),
    }


def _recording_digest(recording: Recording) -> str:
    normalized = str(recording.path).replace("\\", "/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _recording_key(recording: Recording, index: int, *, anonymize: bool) -> str:
    if anonymize:
        return f"recording_{index + 1:04d}_{_recording_digest(recording)}"
    return f"{recording.participant_id}_{recording.recording_id}_{_recording_digest(recording)}"


def _participant_keys(recordings: list[Recording], *, anonymize: bool) -> dict[str, str]:
    participant_ids = sorted({recording.participant_id for recording in recordings})
    if anonymize:
        return {
            participant_id: f"participant_{index + 1:03d}"
            for index, participant_id in enumerate(participant_ids)
        }
    return {participant_id: participant_id for participant_id in participant_ids}


def _status_path(recording_output_dir: Path, config: BatchConfig) -> Path:
    return recording_output_dir / config.status_file_name


def _load_completed_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("status") == "completed":
        return data
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_message(message: str, recording: Recording) -> str:
    sanitized = message.replace(str(recording.path), "[recording-path]")
    sanitized = sanitized.replace(recording.participant_id, "[participant]")
    sanitized = sanitized.replace(recording.recording_id, "[recording]")
    return sanitized


def _write_status(path: Path, result: BatchRecordingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)


def _logger(output_dir: Path, log_file_name: str) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_file_name
    logger = logging.getLogger(f"eeg_pipeline.batch.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, log_path


def _progress(items: list[Any], *, enabled: bool, total: int) -> Any:
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, total=total, desc="EEG batch", unit="recording")


def _result_from_completed_status(
    data: dict[str, Any],
    *,
    output_dir: Path,
) -> BatchRecordingResult:
    return BatchRecordingResult(
        recording_key=str(data.get("recording_key", output_dir.name)),
        participant_key=str(data.get("participant_key", "unknown")),
        run_label=data.get("run_label"),
        status="skipped",
        output_dir=output_dir,
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        processing_time_seconds=float(data.get("processing_time_seconds", 0.0)),
        message="Skipped because a completed status marker exists.",
        metrics=dict(data.get("metrics", {})),
        resumed=True,
    )


def _process_one(
    recording: Recording,
    *,
    recording_key: str,
    participant_key: str,
    output_dir: Path,
    config: BatchConfig,
    processor: BatchProcessor,
) -> BatchRecordingResult:
    recording_output_dir = output_dir / "recordings" / recording_key
    status_path = _status_path(recording_output_dir, config)
    if config.resume:
        completed_status = _load_completed_status(status_path)
        if completed_status is not None:
            return _result_from_completed_status(completed_status, output_dir=recording_output_dir)

    recording_output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    start = perf_counter()
    try:
        metrics = processor(recording, recording_output_dir) or {}
        processing_time = perf_counter() - start
        result = BatchRecordingResult(
            recording_key=recording_key,
            participant_key=participant_key,
            run_label=recording.run_label,
            status="completed",
            output_dir=recording_output_dir,
            started_at=started_at,
            finished_at=_utc_now(),
            processing_time_seconds=processing_time,
            metrics=dict(metrics),
        )
        logging.getLogger(__name__).debug("Completed recording %s", recording_key)
    except Exception as error:  # noqa: BLE001 - batch runs must keep processing later recordings.
        processing_time = perf_counter() - start
        result = BatchRecordingResult(
            recording_key=recording_key,
            participant_key=participant_key,
            run_label=recording.run_label,
            status="failed",
            output_dir=recording_output_dir,
            started_at=started_at,
            finished_at=_utc_now(),
            processing_time_seconds=processing_time,
            message=_safe_message(str(error), recording),
        )
        logging.getLogger(__name__).warning(
            "Recording %s failed during batch processing: %s",
            recording_key,
            result.message,
        )
    _write_status(status_path, result)
    return result


def _write_summary_csv(path: Path, results: tuple[BatchRecordingResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for result in results for key in result.metrics})
    fieldnames = [
        "recording_key",
        "participant_key",
        "run_label",
        "status",
        "resumed",
        "processing_time_seconds",
        "message",
        "output_dir",
        *[f"metric_{key}" for key in metric_keys],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "recording_key": result.recording_key,
                "participant_key": result.participant_key,
                "run_label": result.run_label or "",
                "status": result.status,
                "resumed": result.resumed,
                "processing_time_seconds": f"{result.processing_time_seconds:.6f}",
                "message": result.message,
                "output_dir": str(result.output_dir),
            }
            for key in metric_keys:
                row[f"metric_{key}"] = result.metrics.get(key, "")
            writer.writerow(row)


def _run_serial(
    jobs: list[dict[str, Any]],
    *,
    show_progress: bool,
) -> tuple[BatchRecordingResult, ...]:
    results: list[BatchRecordingResult] = []
    for job in _progress(jobs, enabled=show_progress, total=len(jobs)):
        results.append(_process_one(**job))
    return tuple(results)


def _run_threaded(
    jobs: list[dict[str, Any]],
    *,
    n_jobs: int,
    show_progress: bool,
) -> tuple[BatchRecordingResult, ...]:
    results: list[BatchRecordingResult] = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_process_one, **job) for job in jobs]
        for future in _progress(futures, enabled=show_progress, total=len(futures)):
            results.append(future.result())
    return tuple(sorted(results, key=lambda result: result.recording_key))


def process_dataset(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    processor: BatchProcessor,
    config: dict[str, Any] | BatchConfig | None = None,
) -> BatchReport:
    """Process all discovered recordings in a dataset with resume and summary outputs."""

    batch_config = _config_from_mapping(config)
    if output_dir is not None:
        batch_config = BatchConfig(
            output_dir=Path(output_dir),
            resume=batch_config.resume,
            n_jobs=batch_config.n_jobs,
            parallel_backend=batch_config.parallel_backend,
            show_progress=batch_config.show_progress,
            summary_csv_name=batch_config.summary_csv_name,
            log_file_name=batch_config.log_file_name,
            status_file_name=batch_config.status_file_name,
            anonymize_identifiers=batch_config.anonymize_identifiers,
        )
    if batch_config.parallel_backend not in {"serial", "thread"}:
        raise ValueError("parallel_backend must be 'serial' or 'thread'.")

    root = Path(input_dir)
    recordings = discover_recordings(root)
    participant_keys = _participant_keys(
        recordings,
        anonymize=batch_config.anonymize_identifiers,
    )
    logger, log_path = _logger(batch_config.output_dir, batch_config.log_file_name)
    logger.info("Starting batch processing: recordings=%s", len(recordings))

    jobs = [
        {
            "recording": recording,
            "recording_key": _recording_key(
                recording,
                index,
                anonymize=batch_config.anonymize_identifiers,
            ),
            "participant_key": participant_keys[recording.participant_id],
            "output_dir": batch_config.output_dir,
            "config": batch_config,
            "processor": processor,
        }
        for index, recording in enumerate(recordings)
    ]
    if batch_config.parallel_backend == "thread" and batch_config.n_jobs > 1:
        results = _run_threaded(
            jobs,
            n_jobs=batch_config.n_jobs,
            show_progress=batch_config.show_progress,
        )
    else:
        results = _run_serial(jobs, show_progress=batch_config.show_progress)

    for result in results:
        logger.info(
            "Recording %s finished with status=%s resumed=%s",
            result.recording_key,
            result.status,
            result.resumed,
        )
    completed = sum(1 for result in results if result.status == "completed")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    summary_csv_path = batch_config.output_dir / batch_config.summary_csv_name
    _write_summary_csv(summary_csv_path, results)
    logger.info(
        "Finished batch processing: completed=%s failed=%s skipped=%s",
        completed,
        failed,
        skipped,
    )
    return BatchReport(
        input_dir=root,
        output_dir=batch_config.output_dir,
        summary_csv_path=summary_csv_path,
        log_path=log_path,
        total_recordings=len(recordings),
        completed=completed,
        failed=failed,
        skipped=skipped,
        results=results,
        config=batch_config.to_dict(),
    )
