"""Command-line interface for eeg-preprocessing-pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_pipeline.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="eeg-pipeline",
        description="Run the EEG preprocessing and quality-control pipeline.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/default_config.yaml"),
        help="Path to a YAML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect recording paths without loading EEG.")
    inspect_parser.add_argument("input_path", type=Path)

    preflight_parser = subparsers.add_parser("preflight", help="Run Git and data-safety checks.")
    preflight_parser.add_argument("root", nargs="?", type=Path, default=Path("."))

    run_parser = subparsers.add_parser("preprocess-recording", help="Run one recording.")
    run_parser.add_argument("input_path", type=Path)
    run_parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/processed"))

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    pipeline = Pipeline(config_path=args.config)
    if args.command == "inspect":
        result = pipeline.inspect(args.input_path)
    elif args.command == "preflight":
        result = pipeline.preflight(args.root)
    elif args.command == "preprocess-recording":
        result = pipeline.run(input_path=args.input_path, output_dir=args.output_dir)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
