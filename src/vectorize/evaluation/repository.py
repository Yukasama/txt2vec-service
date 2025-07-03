"""Repository for evaluation task operations."""

from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.task.task_status import TaskStatus

from .models import EvaluationTask

__all__ = [
    "get_evaluation_task_by_id_db",
    "save_evaluation_task_db",
    "update_evaluation_task_metadata_db",
    "update_evaluation_task_results_db",
    "update_evaluation_task_status_db",
]


async def save_evaluation_task_db(
    db: AsyncSession, evaluation_task: EvaluationTask
) -> EvaluationTask:
    """Save evaluation task to database.

    Args:
        db: Database session
        evaluation_task: EvaluationTask instance to save

    Returns:
        Saved EvaluationTask instance
    """
    db.add(evaluation_task)
    await db.commit()
    await db.refresh(evaluation_task)
    return evaluation_task


async def get_evaluation_task_by_id_db(
    db: AsyncSession, task_id: UUID
) -> EvaluationTask | None:
    """Get evaluation task by ID.

    Args:
        db: Database session
        task_id: UUID of the evaluation task

    Returns:
        EvaluationTask instance or None if not found
    """
    query = select(EvaluationTask).where(EvaluationTask.id == task_id)
    result = await db.exec(query)
    return result.first()


async def update_evaluation_task_status_db(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
    progress: float | None = None,
) -> None:
    """Update evaluation task status.

    Args:
        db: Database session
        task_id: UUID of the evaluation task
        status: New task status
        error_msg: Optional error message
        progress: Optional progress value (0.0-1.0)
    """
    task = await get_evaluation_task_by_id_db(db, task_id)
    if task:
        task.task_status = status
        if error_msg is not None:
            task.error_msg = error_msg
        if progress is not None:
            task.progress = progress
        db.add(task)
        await db.commit()


async def update_evaluation_task_results_db(
    db: AsyncSession,
    task_id: UUID,
    evaluation_metrics: str | None = None,
    baseline_metrics: str | None = None,
    evaluation_summary: str | None = None,
) -> None:
    """Update evaluation task results.

    Args:
        db: Database session
        task_id: UUID of the evaluation task
        evaluation_metrics: JSON string of evaluation metrics
        baseline_metrics: JSON string of baseline metrics
        evaluation_summary: Human-readable evaluation summary
    """
    await _update_evaluation_task_fields_db(
        db,
        task_id,
        evaluation_metrics=evaluation_metrics,
        baseline_metrics=baseline_metrics,
        evaluation_summary=evaluation_summary,
    )


async def update_evaluation_task_metadata_db(
    db: AsyncSession,
    task_id: UUID,
    model_tag: str | None = None,
    baseline_model_tag: str | None = None,
) -> None:
    """Update evaluation task metadata (model, dataset, baseline info).

    Args:
        db: Database session
        task_id: UUID of the evaluation task
        model_tag: Tag of the evaluated model
        baseline_model_tag: Tag of the baseline model (if comparison performed)
    """
    await _update_evaluation_task_fields_db(
        db,
        task_id,
        model_tag=model_tag,
        baseline_model_tag=baseline_model_tag,
    )


async def _update_evaluation_task_fields_db(
    db: AsyncSession,
    task_id: UUID,
    **field_updates: str | None,
) -> None:
    """Generic function to update evaluation task fields.

    Args:
        db: Database session
        task_id: UUID of the evaluation task
        **field_updates: Field values to update
    """
    task = await get_evaluation_task_by_id_db(db, task_id)
    if task:
        for field, value in field_updates.items():
            if value is not None and hasattr(task, field):
                setattr(task, field, value)
        db.add(task)
        await db.commit()
