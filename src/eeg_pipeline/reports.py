"""HTML quality-control report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _render_mapping(mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in sorted(mapping.items()):
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>")
    return "\n".join(rows)


def generate_report(
    metrics: dict[str, Any],
    output_path: str | Path,
    figures: list[str | Path] | None = None,
    *,
    title: str = "EEG Quality-Control Report",
) -> Path:
    """Generate a small standalone HTML QC report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_items = ""
    for figure in figures or []:
        figure_items += f'<li><a href="{escape(str(figure))}">{escape(str(figure))}</a></li>\n'
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; min-width: 24rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>
    {_render_mapping(metrics)}
  </table>
  <h2>Figures</h2>
  <ul>{figure_items}</ul>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def generate_participant_summary(
    participant_id: str,
    recording_metrics: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Generate a participant-level HTML summary."""

    metrics = {
        "participant_id": participant_id,
        "n_recordings": len(recording_metrics),
        "recordings": "; ".join(str(item.get("recording_id", "unknown")) for item in recording_metrics),
    }
    return generate_report(metrics, output_path, title=f"{participant_id} EEG QC Summary")
