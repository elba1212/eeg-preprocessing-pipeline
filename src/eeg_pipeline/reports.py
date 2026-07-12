"""HTML quality-control dashboards for EEG preprocessing reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from html import escape
from pathlib import Path
from typing import Any


PRIVATE_KEY_FRAGMENTS = (
    "participant",
    "subject",
    "patient",
    "person",
    "path",
    "directory",
    "folder",
    "file",
    "filename",
)


@dataclass(frozen=True)
class DashboardSection:
    """A rendered dashboard section."""

    title: str
    content: Any
    description: str = ""


@dataclass(frozen=True)
class RecordingDashboard:
    """Result from creating a recording-level dashboard."""

    output_path: Path
    title: str
    sections: tuple[str, ...]
    used_mne_report: bool


@dataclass(frozen=True)
class ParticipantDashboard:
    """Result from creating a participant-level dashboard."""

    output_path: Path
    title: str
    n_recordings: int
    sections: tuple[str, ...]
    used_mne_report: bool


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for HTML dashboard rendering."""

    title: str = "EEG Quality-Control Dashboard"
    use_mne_report: bool = True
    redact_private_fields: bool = True
    include_empty_sections: bool = True
    overwrite: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration mapping."""

        return asdict(self)


def _config_from_mapping(config: dict[str, Any] | ReportConfig | None) -> ReportConfig:
    if isinstance(config, ReportConfig):
        return config
    data = config or {}
    return ReportConfig(
        title=str(data.get("title", ReportConfig.title)),
        use_mne_report=bool(data.get("use_mne_report", ReportConfig.use_mne_report)),
        redact_private_fields=bool(
            data.get("redact_private_fields", ReportConfig.redact_private_fields)
        ),
        include_empty_sections=bool(
            data.get("include_empty_sections", ReportConfig.include_empty_sections)
        ),
        overwrite=bool(data.get("overwrite", ReportConfig.overwrite)),
    )


def _is_private_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in PRIVATE_KEY_FRAGMENTS)


def _safe_value(value: Any, *, key: str = "", redact_private_fields: bool = True) -> Any:
    if redact_private_fields and _is_private_key(key):
        return "[redacted]"
    if isinstance(value, Path):
        return "[redacted-path]" if redact_private_fields else str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_value(asdict(value), key=key, redact_private_fields=redact_private_fields)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _safe_value(value.to_dict(), key=key, redact_private_fields=redact_private_fields)
    if isinstance(value, dict):
        return {
            str(item_key): _safe_value(
                item_value,
                key=str(item_key),
                redact_private_fields=redact_private_fields,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, redact_private_fields=redact_private_fields) for item in value]
    return value


def _format_scalar(value: Any) -> str:
    if value is None:
        return '<span class="empty">Not available</span>'
    if isinstance(value, float):
        return escape(f"{value:.6g}")
    if isinstance(value, bool):
        return "yes" if value else "no"
    return escape(str(value))


def _render_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return '<span class="empty">Not available</span>'
        rows = "\n".join(
            "<tr>" f"<th>{escape(str(key))}</th>" f"<td>{_render_value(item)}</td>" "</tr>"
            for key, item in value.items()
        )
        return f"<table>{rows}</table>"
    if isinstance(value, list):
        if not value:
            return '<span class="empty">None</span>'
        items = "\n".join(f"<li>{_render_value(item)}</li>" for item in value)
        return f"<ul>{items}</ul>"
    return _format_scalar(value)


def _render_section(section: DashboardSection) -> str:
    description = (
        f'<p class="section-description">{escape(section.description)}</p>'
        if section.description
        else ""
    )
    return (
        "<section>"
        f"<h2>{escape(section.title)}</h2>"
        f"{description}"
        f"{_render_value(section.content)}"
        "</section>"
    )


def _document(title: str, sections: tuple[DashboardSection, ...]) -> str:
    section_html = "\n".join(_render_section(section) for section in sections)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #172026;
      --muted: #5b6871;
      --border: #d9e0e5;
      --surface: #f7f9fb;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      background: white;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      color: var(--accent);
    }}
    section {{
      padding: 18px 0 22px;
      border-bottom: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 10px;
      border: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 30%;
      background: var(--surface);
      font-weight: 600;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .section-description, .empty, .generated {{
      color: var(--muted);
    }}
    .generated {{
      margin-top: 8px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p class="generated">Generated by eeg-preprocessing-pipeline. Private path-like fields are redacted by default.</p>
  </header>
  <main>
    {section_html}
  </main>
</body>
</html>
"""


def _write_html(output_path: str | Path, html: str, *, overwrite: bool) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _mne_report_from_html(title: str, html: str) -> Any | None:
    try:
        import mne
    except ImportError:
        return None
    report_class = getattr(mne, "Report", None)
    if report_class is None:
        return None
    report = report_class(title=title)
    add_html = getattr(report, "add_html", None)
    if callable(add_html):
        add_html(html=html, title="Dashboard")
        return report
    add_htmls_to_section = getattr(report, "add_htmls_to_section", None)
    if callable(add_htmls_to_section):
        add_htmls_to_section([html], captions=["Dashboard"], section="Dashboard")
        return report
    return None


def _save_mne_report(report: Any, output_path: Path, *, overwrite: bool) -> bool:
    try:
        report.save(output_path, overwrite=overwrite, open_browser=False)
    except TypeError:
        report.save(output_path, overwrite=overwrite)
    return True


def _normalize_sections(
    sections: list[DashboardSection],
    *,
    include_empty_sections: bool,
    redact_private_fields: bool,
) -> tuple[DashboardSection, ...]:
    normalized: list[DashboardSection] = []
    for section in sections:
        content = _safe_value(
            section.content,
            key=section.title,
            redact_private_fields=redact_private_fields,
        )
        if not include_empty_sections and content in ({}, [], None):
            continue
        normalized.append(
            DashboardSection(
                title=section.title,
                content=content,
                description=section.description,
            )
        )
    return tuple(normalized)


def build_recording_sections(
    *,
    metadata: dict[str, Any] | None = None,
    preprocessing_summary: dict[str, Any] | None = None,
    bad_channels: Any = None,
    interpolation: Any = None,
    noisy_windows: Any = None,
    ica: Any = None,
    psd: Any = None,
    quality_metrics: dict[str, Any] | None = None,
) -> tuple[DashboardSection, ...]:
    """Build modular recording dashboard sections from stage reports."""

    return (
        DashboardSection("Metadata", metadata or {}),
        DashboardSection("Preprocessing Summary", preprocessing_summary or {}),
        DashboardSection("Bad Channels", bad_channels or {}),
        DashboardSection("Interpolation", interpolation or {}),
        DashboardSection("Noisy Windows", noisy_windows or {}),
        DashboardSection("ICA", ica or {}),
        DashboardSection("PSD", psd or {}),
        DashboardSection("Quality Metrics", quality_metrics or {}),
    )


def generate_recording_dashboard(
    output_path: str | Path,
    *,
    recording_label: str = "Recording",
    metadata: dict[str, Any] | None = None,
    preprocessing_summary: dict[str, Any] | None = None,
    bad_channels: Any = None,
    interpolation: Any = None,
    noisy_windows: Any = None,
    ica: Any = None,
    psd: Any = None,
    quality_metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | ReportConfig | None = None,
) -> RecordingDashboard:
    """Generate a recording-level HTML quality-control dashboard."""

    report_config = _config_from_mapping(config)
    title = f"{recording_label} Quality-Control Dashboard"
    sections = _normalize_sections(
        list(
            build_recording_sections(
                metadata=metadata,
                preprocessing_summary=preprocessing_summary,
                bad_channels=bad_channels,
                interpolation=interpolation,
                noisy_windows=noisy_windows,
                ica=ica,
                psd=psd,
                quality_metrics=quality_metrics,
            )
        ),
        include_empty_sections=report_config.include_empty_sections,
        redact_private_fields=report_config.redact_private_fields,
    )
    html = _document(title, sections)
    path = Path(output_path)
    used_mne_report = False
    if report_config.use_mne_report:
        mne_report = _mne_report_from_html(title, html)
        if mne_report is not None:
            used_mne_report = _save_mne_report(
                mne_report,
                path,
                overwrite=report_config.overwrite,
            )
    if not used_mne_report:
        path = _write_html(path, html, overwrite=report_config.overwrite)
    return RecordingDashboard(
        output_path=path,
        title=title,
        sections=tuple(section.title for section in sections),
        used_mne_report=used_mne_report,
    )


def build_participant_sections(
    recording_summaries: list[dict[str, Any]],
    *,
    participant_summary: dict[str, Any] | None = None,
) -> tuple[DashboardSection, ...]:
    """Build modular participant dashboard sections."""

    return (
        DashboardSection("Participant Summary", participant_summary or {}),
        DashboardSection("Recordings", recording_summaries),
    )


def generate_participant_dashboard(
    output_path: str | Path,
    recording_summaries: list[dict[str, Any]],
    *,
    participant_label: str = "Participant",
    participant_summary: dict[str, Any] | None = None,
    config: dict[str, Any] | ReportConfig | None = None,
) -> ParticipantDashboard:
    """Generate a participant-level HTML dashboard summarizing recordings."""

    report_config = _config_from_mapping(config)
    title = f"{participant_label} Quality-Control Summary"
    sections = _normalize_sections(
        list(
            build_participant_sections(
                recording_summaries,
                participant_summary=participant_summary,
            )
        ),
        include_empty_sections=report_config.include_empty_sections,
        redact_private_fields=report_config.redact_private_fields,
    )
    html = _document(title, sections)
    path = Path(output_path)
    used_mne_report = False
    if report_config.use_mne_report:
        mne_report = _mne_report_from_html(title, html)
        if mne_report is not None:
            used_mne_report = _save_mne_report(
                mne_report,
                path,
                overwrite=report_config.overwrite,
            )
    if not used_mne_report:
        path = _write_html(path, html, overwrite=report_config.overwrite)
    return ParticipantDashboard(
        output_path=path,
        title=title,
        n_recordings=len(recording_summaries),
        sections=tuple(section.title for section in sections),
        used_mne_report=used_mne_report,
    )


def generate_report(
    metrics: dict[str, Any],
    output_path: str | Path,
    figures: list[str | Path] | None = None,
    *,
    title: str = "EEG Quality-Control Report",
) -> Path:
    """Backward-compatible recording report helper."""

    figure_entries = [
        {"figure": f"figure_{index + 1:03d}"} for index, _figure in enumerate(figures or [])
    ]
    report = generate_recording_dashboard(
        output_path,
        recording_label=title,
        quality_metrics=metrics,
        psd={"figures": figure_entries},
        config=ReportConfig(use_mne_report=False),
    )
    return report.output_path


def generate_participant_summary(
    participant_id: str,
    recording_metrics: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Backward-compatible participant summary helper."""

    report = generate_participant_dashboard(
        output_path,
        recording_metrics,
        participant_label="Participant",
        participant_summary={"participant_label": participant_id},
    )
    return report.output_path
