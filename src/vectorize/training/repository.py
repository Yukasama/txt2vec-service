"""Repository functions for TrainingTask persistence."""

from datetime import UTC, datetime
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.task.task_status import TaskStatus

from .models import TrainingTask

__all__ = [
    "get_train_task_by_id_db",
    "save_training_task_db",
    "update_training_task_dataset_ids_db",
    "update_training_task_metrics_db",
    "update_training_task_status_db",
    "update_training_task_validation_dataset_db",
]


async def update_training_task_dataset_ids_db(
    db: AsyncSession,
    task_id: UUID,
    train_dataset_ids: list[str],
    val_dataset_id: str | None = None
) -> None:
    """Update the dataset IDs of a TrainingTask.

    Args:
        db: The database session.
        task_id: The ID of the training task.
        train_dataset_ids: List of training dataset IDs.
        val_dataset_id: Optional validation dataset ID.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.train_dataset_ids = train_dataset_ids
        task.val_dataset_id = val_dataset_id
        task.updated_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(task)


async def save_training_task_db(db: AsyncSession, task: TrainingTask) -> None:
    """Persist a new TrainingTask to the database.

    Args:
        db: The database session.
        task: The training task to save.
    """
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_training_task_status_db(
    db: AsyncSession, task_id: UUID, status: TaskStatus, error_msg: str | None = None
) -> None:
    """Update the status and error message of a TrainingTask.

    Args:
        db: The database session.
        task_id (UUID): The ID of the training task.
        status (TaskStatus): The new status.
        error_msg (str | None): Optional error message.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.task_status = status
        if status in {TaskStatus.DONE, TaskStatus.FAILED}:
            task.end_date = datetime.now(UTC)
        if error_msg:
            task.error_msg = error_msg
        await db.commit()
        await db.refresh(task)


async def update_training_task_metrics_db(
    db: AsyncSession,
    task_id: UUID,
    metrics: dict[str, float | None],
) -> None:
    """Update the training metrics of a TrainingTask.

    Args:
        db: The database session.
        task_id: The ID of the training task.
        metrics: Dictionary of training metrics.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        if metrics.get("train_runtime") is not None:
            task.train_runtime = metrics["train_runtime"]
        if metrics.get("train_samples_per_second") is not None:
            task.train_samples_per_second = metrics["train_samples_per_second"]
        if metrics.get("train_steps_per_second") is not None:
            task.train_steps_per_second = metrics["train_steps_per_second"]
        if metrics.get("train_loss") is not None:
            task.train_loss = metrics["train_loss"]
        if metrics.get("epoch") is not None:
            task.epoch = metrics["epoch"]
        await db.commit()
        await db.refresh(task)


async def update_training_task_validation_dataset_db(
    db: AsyncSession, task_id: UUID, val_dataset_id: str
) -> None:
    """Update the validation dataset ID of a TrainingTask.

    Args:
        db (AsyncSession): The database session.
        task_id (UUID): The ID of the training task.
        val_dataset_id (str): ID of the validation dataset.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    task = result.first()
    if task:
        task.val_dataset_id = val_dataset_id
        task.updated_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(task)


async def get_train_task_by_id_db(
        db: AsyncSession, task_id: UUID) -> TrainingTask | None:
    """Fetch a TrainingTask by its ID.

    Args:
        db (AsyncSession): The database session.
        task_id (UUID): The ID of the training task.

    Returns:
        TrainingTask | None: The training task if found, else None.
    """
    result = await db.exec(select(TrainingTask).where(TrainingTask.id == task_id))
    return result.first()
