"""Orchestrates the end-to-end SBERT training process."""

import asyncio
import concurrent.futures
import time
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from .exceptions import DatasetValidationError
from .schemas import TrainRequest
from .utils import (
    SBERTTrainingEngine,
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
            self.model = await loop.run_in_executor(
                None, load_and_prepare_model, model_path
            )

            # Data preparation in executor
            def prepare_data() -> tuple:
                return TrainingDataPreparer.prepare_training_data(
                    dataset_paths, train_request.per_device_train_batch_size
                )
            train_dataloader, _ = await loop.run_in_executor(None, prepare_data)

            # Update dataset IDs with the correct values from train_request
            await self._update_dataset_ids_from_request(train_request)

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

    async def _update_dataset_ids_from_request(
        self, train_request: TrainRequest
    ) -> None:
        """Update dataset IDs using the correct values from the training request.

        Args:
            train_request: Training request containing the actual dataset IDs
        """
        await self.db_manager.update_dataset_ids(
            train_request.train_dataset_ids,
            val_dataset_id=train_request.val_dataset_id
        )

        logger.debug(
            "Updated training task with correct dataset IDs from request",
            task_id=str(self.task_id),
            train_dataset_ids=train_request.train_dataset_ids,
            val_dataset_id=train_request.val_dataset_id,
        )

    async def _train_with_yielding(
        self,
        train_dataloader: "DataLoader",
        train_request: TrainRequest,
        output_dir: str,
    ) -> dict:
        """Train model with yielding to prevent blocking other workers."""
        if self.model is None:
            raise RuntimeError("Model was not loaded correctly.")

        # Use ThreadPoolExecutor with yielding - simpler and more reliable

        def train_in_thread() -> dict:
            """Run training in thread with periodic status updates."""
            try:
                if self.model is None:
                    raise RuntimeError("Model was not loaded correctly.")
                training_engine = SBERTTrainingEngine(self.model)
                return training_engine.train_model(
                    train_dataloader, train_request, output_dir
                )
            except Exception as e:
                logger.error(
                    "Training failed in thread",
                    error=str(e),
                    task_id=str(self.task_id)
                )
                raise

        # Submit to thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_in_thread)

            start_time = time.time()
            # Yield control every 3 seconds while training runs
            while not future.done():
                await asyncio.sleep(3)
                elapsed = time.time() - start_time
                logger.debug(
                    "Training in progress...",
                    task_id=str(self.task_id),
                    elapsed_minutes=round(elapsed / 60, 1)
                )

            # Get result or raise exception
            return future.result()

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
