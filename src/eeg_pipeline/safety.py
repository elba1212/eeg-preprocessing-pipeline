"""Repository and data-safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


PRIVATE_DATA_PATTERNS = ("*.mff", "*.bin", "*.mov", "*.mp4", "*.fif", "*.edf", "*.bdf", "*.set")


@dataclass(frozen=True)
class SafetyReport:
    """Result of repository safety checks."""

    git_ok: bool
    private_data_present: bool
    tracked_private_paths: tuple[str, ...]
    warnings: tuple[str, ...]


def find_private_data(root: str | Path = ".") -> list[Path]:
    """Find private or heavy EEG artifacts by path pattern."""

    root = Path(root)
    matches: list[Path] = []
    for pattern in PRIVATE_DATA_PATTERNS:
        matches.extend(root.glob(f"data/raw/**/{pattern}"))
    return sorted(set(matches))


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def run_safety_preflight(root: str | Path = ".") -> SafetyReport:
    """Check Git validity and whether private EEG paths are tracked."""

    root = Path(root)
    warnings: list[str] = []
    status = _git(["status", "--short"], root)
    git_ok = status.returncode == 0
    if not git_ok:
        warnings.append(status.stderr.strip() or "Git status failed; repository metadata may be invalid.")

    tracked_private_paths: tuple[str, ...] = ()
    if git_ok:
        listed = _git(["ls-files", "data/raw"], root)
        tracked = [
            line
            for line in listed.stdout.splitlines()
            if any(line.lower().endswith(pattern.replace("*", "").lower()) for pattern in PRIVATE_DATA_PATTERNS)
        ]
        tracked_private_paths = tuple(sorted(tracked))
        if tracked_private_paths:
            warnings.append("Private EEG artifacts appear to be tracked by Git.")

    private_data_present = bool(find_private_data(root))
    if private_data_present:
        warnings.append("Private EEG data are present locally under data/raw; keep them ignored.")

    return SafetyReport(
        git_ok=git_ok,
        private_data_present=private_data_present,
        tracked_private_paths=tracked_private_paths,
        warnings=tuple(warnings),
    )
