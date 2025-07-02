"""Orchestrates the end-to-end SBERT training process."""


import asyncio
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from .exceptions import DatasetValidationError
from .schemas import TrainRequest
from .utils import (
    AsyncSBERTTrainingEngine,
    TrainingDataPreparer,
    TrainingDatabaseManager,
    TrainingDatasetValidator,
    TrainingFileValidator,
    cleanup_resources,
    load_and_prepare_model,
)


class TrainingOrchestrator:
    """Handles the orchestration of SBERT training."""

    def __init__(self, db: AsyncSession, task_id: UUID) -> None:
        """Initialize the training orchestrator.

        Args:
            db: Database session
            task_id: Training task ID
        """
        self.db = db
        self.task_id = task_id
        self.model = None
        self.db_manager = TrainingDatabaseManager(db, task_id)

    @staticmethod
    async def validate_datasets(
        db: AsyncSession,
        train_dataset_ids: list[str],
        val_dataset_id: str | None = None,
    ) -> list[str]:
        """Validate and prepare dataset paths for training.

        Args:
            db: Database session
            train_dataset_ids: List of training dataset IDs
            val_dataset_id: Optional validation dataset ID

        Returns:
            List of validated dataset paths

        Raises:
            InvalidDatasetIdError: If dataset ID is invalid
            TrainingDatasetNotFoundError: If dataset file is not found or invalid
        """
        return await TrainingDatasetValidator.validate_datasets(
            db, train_dataset_ids, val_dataset_id
        )

    async def run_training_svc(
        self,
        model_path: str,
        train_request: TrainRequest,
        dataset_paths: list[str],
        output_dir: str,
    ) -> None:
        """Main orchestration function for SBERT training.

        Args:
            model_path: Path to the base model
            train_request: Training configuration
            dataset_paths: List of dataset file paths
            output_dir: Output directory for the trained model
        """
        logger.debug(
            "Starting SBERT training",
            model_path=model_path,
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            task_id=str(self.task_id),
        )

        try:
            loop = asyncio.get_running_loop()

            # Model loading in executor
            self.model = await loop.run_in_executor(None, load_and_prepare_model, model_path)

            # Data preparation in executor
            def prepare_data():
                return TrainingDataPreparer.prepare_training_data(
                    dataset_paths, train_request.per_device_train_batch_size
                )
            train_dataloader, _ = await loop.run_in_executor(None, prepare_data)

            # Dataset ID extraction and updating
            await self._update_dataset_ids(dataset_paths)

            # Training in executor with periodic yielding
            training_metrics = await self._train_with_yielding(
                train_dataloader, train_request, output_dir
            )

            await self.db_manager.save_training_metrics(training_metrics)
            await self.db_manager.save_trained_model(train_request, output_dir)
            await self.db_manager.mark_training_complete()

            logger.debug(
                "Training finished successfully",
                model_path=model_path,
                task_id=str(self.task_id),
            )

        except (OSError, RuntimeError, DatasetValidationError) as exc:
            await self.db_manager.handle_training_error(exc)
        finally:
            self._cleanup_resources()

    async def _update_dataset_ids(self, dataset_paths: list[str]) -> None:
        """Extract and update dataset IDs from file paths."""
        import os

        JSONL_SUFFIX = '.jsonl'

        def extract_id(filename: str) -> str | None:
            if '_' in filename and filename.endswith(JSONL_SUFFIX):
                return filename.rsplit('_', 1)[-1].replace(JSONL_SUFFIX, '')
            return None

        if len(dataset_paths) == 1:
            train_file = os.path.basename(str(dataset_paths[0]))
            train_id = extract_id(train_file)
            await self.db_manager.update_dataset_ids([train_id] if train_id else [], val_dataset_id=None)
        else:
            train_ids = []
            for path in dataset_paths[:-1]:
                file = os.path.basename(str(path))
                tid = extract_id(file)
                if tid:
                    train_ids.append(tid)
            val_file = os.path.basename(str(dataset_paths[-1]))
            val_id = extract_id(val_file)
            await self.db_manager.update_dataset_ids(train_ids, val_dataset_id=val_id)

    async def _train_with_yielding(
        self, train_dataloader, train_request, output_dir: str
    ):
        """Train model with async yielding to allow other tasks."""
        if self.model is None:
            raise RuntimeError("Model was not loaded correctly.")

        # Use the new async training engine
        async_training_engine = AsyncSBERTTrainingEngine(self.model)
        return await async_training_engine.train_model_async(
            train_dataloader, train_request, output_dir, yield_interval_steps=50
        )

    def _cleanup_resources(self) -> None:
        """Clean up resources after training."""
        if self.model is not None:
            cleanup_resources(self.model)
            self.model = None

    @staticmethod
    def validate_dataset_files(dataset_paths: list[str]) -> None:
        """Validate dataset files for training.

        Args:
            dataset_paths: List of dataset file paths to validate

        Raises:
            DatasetValidationError: If any dataset file is invalid
        """
        TrainingFileValidator.validate_dataset_files(dataset_paths)
