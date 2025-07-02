"""File validation utilities for training."""

from pathlib import Path

from loguru import logger

from ..exceptions import DatasetValidationError


class TrainingFileValidator:
    """Validates files for training purposes."""

    @staticmethod
    def validate_dataset_files(dataset_paths: list[str]) -> None:
        """Validate dataset files for training.

        Args:
            dataset_paths: List of dataset file paths to validate

        Raises:
            DatasetValidationError: If any dataset file is invalid
        """
        for path in dataset_paths:
            TrainingFileValidator._validate_single_file(path)
            logger.debug("Validated dataset file", dataset_file=path)

    @staticmethod
    def _validate_single_file(path: str) -> None:
        """Validate a single dataset file."""
        file_path = Path(path)

        if not file_path.is_file():
            raise DatasetValidationError(f"Dataset file not found: {path}")

        if file_path.stat().st_size == 0:
            raise DatasetValidationError(f"Dataset file is empty: {path}")

        if file_path.suffix not in {".csv", ".tsv", ".jsonl"}:
            raise DatasetValidationError(
                f"Invalid file type for dataset: {path}. "
                "Supported formats: .csv, .tsv, .jsonl"
            )
