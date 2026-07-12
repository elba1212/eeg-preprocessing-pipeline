"""Example script for running the EEG preprocessing pipeline."""

from pathlib import Path

from eeg_pipeline import Pipeline


def main() -> None:
    """Run the scaffolded pipeline on a placeholder input path."""
    # TODO: Replace this path with a demo dataset once one is available.
    input_path = Path("data/raw/example.edf")
    output_dir = Path("data/processed")

    pipeline = Pipeline(config_path=Path("config/default_config.yaml"))
    pipeline.run(input_path=input_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
