"""Database operations manager for training tasks."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.task.task_status import TaskStatus

from ..repository import (
    get_train_task_by_id_db,
    update_training_task_metrics_db,
    update_training_task_status_db,
)
from ..schemas import TrainRequest


class TrainingDatabaseManager:

    async def update_dataset_ids(
        self,
        train_dataset_ids: list[str],
        val_dataset_id: str | None = None
    ) -> None:
        """Update the training task with dataset IDs.

        Args:
            train_dataset_ids: List of training dataset IDs
            val_dataset_id: Optional validation dataset ID
        """
        from ..repository import update_training_task_dataset_ids_db
        await update_training_task_dataset_ids_db(
            self.db, self.task_id, train_dataset_ids, val_dataset_id
        )

        logger.debug(
            "Updated training task with dataset IDs",
            task_id=str(self.task_id),
            train_dataset_ids=train_dataset_ids,
            val_dataset_id=val_dataset_id,
        )
    """Handles database operations for training tasks."""

    def __init__(self, db: AsyncSession, task_id: UUID) -> None:
        """Initialize the database manager.

        Args:
            db: Database session
            task_id: Training task ID
        """
        self.db = db
        self.task_id = task_id

    async def save_training_metrics(self, training_metrics: dict) -> None:
        """Save training metrics to the database.

        Args:
            training_metrics: Dictionary containing training metrics
        """
        await update_training_task_metrics_db(
            self.db,
            self.task_id,
            training_metrics,
        )

        logger.debug(
            "Training metrics saved to database",
            task_id=str(self.task_id),
            **training_metrics,
        )

    async def save_trained_model(
        self, train_request: TrainRequest, output_dir: str
    ) -> None:
        """Save the trained model to the database.

        Args:
            train_request: Training configuration
            output_dir: Output directory where model was saved
        """
        parent_model = await get_ai_model_db(self.db, train_request.model_tag)
        tag_time = Path(output_dir).name
        new_model_tag = Path(output_dir).name

        new_model = AIModel(
            name=f"Fine-tuned: {parent_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=parent_model.id,
        )

        new_model_id = await save_ai_model_db(self.db, new_model)

        task = await get_train_task_by_id_db(self.db, self.task_id)
        if task:
            task.trained_model_id = new_model_id
            await self.db.commit()
            await self.db.refresh(task)

    async def handle_training_error(self, exc: Exception) -> None:
        """Handle training errors.

        Args:
            exc: The exception that occurred
        """
        logger.error(
            "Training failed",
            task_id=str(self.task_id),
            exc=str(exc),
        )
        await update_training_task_status_db(
            self.db,
            self.task_id,
            TaskStatus.FAILED,
            error_msg=str(exc),
        )

    async def mark_training_complete(self) -> None:
        """Mark the training task as completed."""
        await update_training_task_status_db(self.db, self.task_id, TaskStatus.DONE)
        logger.debug(
            "Training task marked as complete",
            task_id=str(self.task_id),
        )
