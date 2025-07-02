"""Dataset resolution utilities for evaluation tasks."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.config import settings
from vectorize.dataset.repository import get_dataset_db
from vectorize.training.exceptions import (
    InvalidDatasetIdError,
    TrainingDatasetNotFoundError,
)
from vectorize.training.repository import get_train_task_by_id_db

from ..schemas import EvaluationRequest


class EvaluationDatasetResolver:
    """Handles dataset resolution for evaluation tasks."""

    @staticmethod
    def validate_dataset_resolution_input(
        evaluation_request: EvaluationRequest,
    ) -> None:
        """Validate dataset resolution input parameters.

        Args:
            evaluation_request: Evaluation request to validate

        Raises:
            ValueError: If validation fails
        """
        if evaluation_request.dataset_id and evaluation_request.training_task_id:
            raise ValueError(
                "Cannot specify both dataset_id and training_task_id. "
                "Use dataset_id for explicit dataset or training_task_id "
                "to use the validation dataset from that training."
            )

        if (
            not evaluation_request.dataset_id
            and not evaluation_request.training_task_id
        ):
            raise ValueError("Must specify either dataset_id or training_task_id.")

    @staticmethod
    async def resolve_evaluation_dataset(
        db: AsyncSession, evaluation_request: EvaluationRequest
    ) -> Path:
        """Resolve the dataset path for evaluation.

        Uses either the explicit dataset_id or gets the validation dataset
        from the training_task_id.

        Args:
            db: Database session
            evaluation_request: Evaluation request with dataset_id and/or
                training_task_id

        Returns:
            Path to the dataset file

        Raises:
            ValueError: If neither dataset_id nor training_task_id is provided,
                       or if both are provided.
            InvalidDatasetIdError: If dataset ID is invalid.
            TrainingDatasetNotFoundError: If dataset or training task not found.
        """
        EvaluationDatasetResolver.validate_dataset_resolution_input(evaluation_request)

        if evaluation_request.dataset_id:
            return await EvaluationDatasetResolver._resolve_explicit_dataset(
                db, evaluation_request.dataset_id
            )

        if evaluation_request.training_task_id is None:
            raise ValueError("training_task_id must be set at this point")
        return await EvaluationDatasetResolver._resolve_training_validation_dataset(
            db, evaluation_request.training_task_id
        )

    @staticmethod
    async def _resolve_explicit_dataset(db: AsyncSession, dataset_id: str) -> Path:
        """Resolve explicit dataset by ID.

        Args:
            db: Database session
            dataset_id: Dataset ID to resolve

        Returns:
            Path to the dataset file

        Raises:
            InvalidDatasetIdError: If dataset ID is invalid
            TrainingDatasetNotFoundError: If dataset not found
        """
        try:
            dataset_uuid = UUID(dataset_id)
        except ValueError as exc:
            raise InvalidDatasetIdError(dataset_id) from exc

        dataset = await get_dataset_db(db, dataset_uuid)
        if not dataset:
            raise TrainingDatasetNotFoundError(f"Dataset not found: {dataset_id}")

        dataset_path = settings.dataset_upload_dir / dataset.file_name
        if not dataset_path.exists():
            raise TrainingDatasetNotFoundError(
                f"Dataset file not found: {dataset_path}"
            )

        logger.info(
            "Using explicit dataset for evaluation",
            dataset_id=dataset_id,
            dataset_path=str(dataset_path),
        )
        return dataset_path

    @staticmethod
    async def _resolve_training_validation_dataset(
        db: AsyncSession, training_task_id: str
    ) -> Path:
        """Resolve validation dataset from training task.

        Args:
            db: Database session
            training_task_id: Training task ID to get validation dataset from

        Returns:
            Path to the validation dataset file

        Raises:
            InvalidDatasetIdError: If training task ID is invalid
            TrainingDatasetNotFoundError: If training task or dataset not found
        """
        try:
            task_uuid = UUID(training_task_id)
        except ValueError as exc:
            raise InvalidDatasetIdError(training_task_id) from exc

        training_task = await get_train_task_by_id_db(db, task_uuid)
        if not training_task:
            raise TrainingDatasetNotFoundError(
                f"Training task not found: {training_task_id}"
            )

        if not training_task.val_dataset_id:
            raise TrainingDatasetNotFoundError(
                f"Training task {training_task_id} has no validation dataset ID"
            )

        try:
            val_dataset_uuid = UUID(training_task.val_dataset_id)
        except ValueError as exc:
            raise InvalidDatasetIdError(training_task.val_dataset_id) from exc

        val_dataset = await get_dataset_db(db, val_dataset_uuid)
        if not val_dataset:
            raise TrainingDatasetNotFoundError(
                f"Validation dataset not found: {training_task.val_dataset_id}"
            )

        dataset_path = settings.dataset_upload_dir / val_dataset.file_name
        if not dataset_path.exists():
            raise TrainingDatasetNotFoundError(
                f"Validation dataset file not found: {dataset_path}"
            )

        logger.info(
            "Using validation dataset from training task",
            training_task_id=training_task_id,
            val_dataset_id=training_task.val_dataset_id,
            validation_dataset_path=str(dataset_path),
        )
        return dataset_path

    @staticmethod
    async def get_dataset_info(
        db: AsyncSession, evaluation_request: EvaluationRequest
    ) -> str | None:
        """Get dataset info string for display purposes.

        Args:
            db: Database session
            evaluation_request: Evaluation request

        Returns:
            Dataset info string or None
        """
        if evaluation_request.dataset_id:
            dataset = await get_dataset_db(db, UUID(evaluation_request.dataset_id))
            if dataset:
                return f"Dataset: {dataset.file_name}"
            return f"Dataset ID: {evaluation_request.dataset_id}"
        if evaluation_request.training_task_id:
            return f"Training validation set: {evaluation_request.training_task_id}"
        return None
