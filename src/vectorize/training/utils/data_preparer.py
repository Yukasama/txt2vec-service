"""Training data preparation utilities."""

from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from vectorize.evaluation.utils.dataset_validator import DatasetValidator

from .input_examples import InputExampleDataset, prepare_input_examples


class TrainingDataPreparer:
    """Prepares training data from datasets."""

    @staticmethod
    def prepare_training_data(
        dataset_paths: list[str], batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from multiple dataset paths.

        Args:
            dataset_paths: List of dataset file paths
                (training datasets + optional validation)
            batch_size: Training batch size

        Returns:
            Tuple of (DataLoader for training, validation dataset path)
        """
        if len(dataset_paths) > 1:
            return TrainingDataPreparer._prepare_multi_dataset_training(
                dataset_paths, batch_size
            )
        return TrainingDataPreparer._prepare_single_dataset_training(
            dataset_paths[0], batch_size
        )

    @staticmethod
    def _prepare_multi_dataset_training(
        dataset_paths: list[str], batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from multiple datasets with explicit validation."""
        training_paths = dataset_paths[:-1]
        validation_path = dataset_paths[-1]

        logger.debug(
            "Multi-dataset training setup",
            num_training_datasets=len(training_paths),
            has_validation_dataset=True,
            training_datasets=training_paths,
            validation_dataset=validation_path,
        )

        all_train_examples = []
        dataset_stats = []

        for i, path in enumerate(training_paths):
            df = DatasetValidator.validate_dataset(Path(path))
            examples = prepare_input_examples(df)
            all_train_examples.extend(examples)

            dataset_stats.append({
                "dataset_name": Path(path).name,
                "type": "training",
                "samples": len(df),
                "examples": len(examples),
            })

            logger.debug(
                "Loaded training dataset",
                dataset_index=i + 1,
                dataset_name=Path(path).name,
                samples=len(df),
                examples_generated=len(examples),
            )

        val_df = DatasetValidator.validate_dataset(Path(validation_path))
        val_examples_count = len(prepare_input_examples(val_df))
        dataset_stats.append({
            "dataset_name": Path(validation_path).name,
            "type": "validation",
            "samples": len(val_df),
            "examples": val_examples_count,
        })

        TrainingDataPreparer._log_training_summary(
            dataset_stats, len(all_train_examples), batch_size, validation_path
        )

        train_dataset = InputExampleDataset(all_train_examples)
        return (
            DataLoader(train_dataset, batch_size=batch_size, num_workers=0),
            validation_path,
        )

    @staticmethod
    def _prepare_single_dataset_training(
        dataset_path: str, batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from single dataset with auto-split."""
        validation_dataset_path = f"{dataset_path}#auto-split"

        df = DatasetValidator.validate_dataset(Path(dataset_path))
        all_examples = prepare_input_examples(df)

        val_split = int(0.1 * len(all_examples))
        train_examples = all_examples[val_split:]
        val_examples = all_examples[:val_split]

        dataset_stats = [
            {
                "dataset_name": Path(dataset_path).name,
                "type": "single_with_split",
                "total_samples": len(df),
                "total_examples": len(all_examples),
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "validation_split": "10%",
            }
        ]

        logger.debug(
            "Single dataset training with auto-split",
            dataset_name=Path(dataset_path).name,
            total_samples=len(df),
            total_examples=len(all_examples),
            train_examples=len(train_examples),
            validation_examples=len(val_examples),
            validation_split_percent=10,
        )

        TrainingDataPreparer._log_training_summary(
            dataset_stats, len(train_examples), batch_size, validation_dataset_path
        )

        train_dataset = InputExampleDataset(train_examples)
        return (
            DataLoader(train_dataset, batch_size=batch_size, num_workers=0),
            validation_dataset_path,
        )

    @staticmethod
    def _log_training_summary(
        dataset_stats: list[dict],
        total_train_examples: int,
        batch_size: int,
        validation_dataset_path: str | None,
    ) -> None:
        """Log final training data preparation summary."""
        dataset_summary = []
        for stat in dataset_stats:
            summary = (
                f"{stat.get('dataset_name', 'unknown')} "
                f"({stat.get('type', 'unknown')}): "
                f"{stat.get('examples', 0)} examples"
            )
            dataset_summary.append(summary)

        logger.debug(
            "Training data preparation complete",
            total_datasets_used=len(dataset_stats),
            total_training_examples=total_train_examples,
            batch_size=batch_size,
            datasets=dataset_summary,
            validation_dataset_path=validation_dataset_path,
        )
