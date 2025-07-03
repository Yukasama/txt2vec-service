"""Tasks repository."""

from collections.abc import Sequence
from typing import Any

from loguru import logger
from sqlalchemy import func, select, union_all
from sqlmodel import text
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.evaluation.models import EvaluationTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.training.models import TrainingTask
from vectorize.upload.models import UploadTask

from .query_builder import build_query
from .schemas import TaskFilters
from .task_type import TaskType

__all__ = ["get_tasks_db"]


async def get_tasks_db(db: AsyncSession, params: TaskFilters) -> tuple[Sequence, int]:
    """Retrieve tasks from database with filtering and pagination.

    Aggregates tasks from multiple types (upload, synthesis, dataset) with
    comprehensive filtering and pagination support.

    Args:
        db: Database session for executing queries.
        params: Filter parameters containing limit, offset, completed status,
                specific statuses, and time window criteria.

    Returns:
        Tuple containing:
        - Sequence of task action rows ordered by creation date (newest first).
          Each row contains id, task_status, created_at, end_date, and task_type.
        - Total count of tasks matching the filters (without pagination).
    """
    if params.baseline_id:
        available_types = [TaskType.TRAINING]
    elif params.dataset_id:
        available_types = [TaskType.TRAINING, TaskType.EVALUATION]
    else:
        available_types = [
            TaskType.MODEL_UPLOAD,
            TaskType.SYNTHESIS,
            TaskType.DATASET_UPLOAD,
            TaskType.TRAINING,
            TaskType.EVALUATION,
        ]

    if params.task_types:
        task_types = [tt for tt in params.task_types if tt in available_types]
    else:
        task_types = available_types

    queries = _build_task_queries(task_types, params)

    if not queries:
        return [], 0

    combined = queries[0] if len(queries) == 1 else union_all(*queries)
    tasks_sq = combined.subquery("tasks_sq")

    stmt = select(tasks_sq)

    bind_params: dict[str, Any] = {}
    if params.tag:
        stmt = stmt.where(tasks_sq.c.tag == text(":tag"))
        bind_params["tag"] = params.tag

    count_stmt = select(func.count()).select_from(tasks_sq)
    if params.tag:
        count_stmt = count_stmt.where(tasks_sq.c.tag == text(":tag"))

    if bind_params:
        total_result = await db.exec(count_stmt, params=bind_params)  # type: ignore[arg-type]
    else:
        total_result = await db.exec(count_stmt)  # type: ignore[arg-type]

    total = total_result.scalar() or 0

    stmt = (
        stmt.order_by(tasks_sq.c.created_at.desc())
        .limit(params.limit)
        .offset(params.offset or 0)
    )

    if bind_params:
        result = await db.exec(stmt, params=bind_params)  # type: ignore[arg-type]
    else:
        result = await db.exec(stmt)  # type: ignore[arg-type]

    rows = result.all()
    logger.debug(
        "Tasks fetched from DB", count=len(rows), total=total, params=str(params)
    )
    return rows, total


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _build_task_queries(task_types: list[TaskType], params: TaskFilters) -> list:
    """Build queries for specified task types with given parameters.

    Args:
        task_types: List of task types to build queries for.
        params: Filter parameters for the queries.

    Returns:
        List of SQLAlchemy query objects.
    """
    status_set = set(params.statuses or [])
    queries = []

    for tt in task_types:
        if tt == TaskType.MODEL_UPLOAD:
            queries.append(
                build_query(
                    UploadTask,
                    "model_upload",
                    statuses=status_set,
                    hours=params.within_hours,
                )
            )
        elif tt == TaskType.SYNTHESIS:
            queries.append(
                build_query(
                    SynthesisTask,
                    "synthesis",
                    statuses=status_set,
                    hours=params.within_hours,
                )
            )
        elif tt == TaskType.DATASET_UPLOAD:
            queries.append(
                build_query(
                    UploadDatasetTask,
                    "dataset_upload",
                    statuses=status_set,
                    hours=params.within_hours,
                )
            )
        elif tt == TaskType.TRAINING:
            queries.append(
                build_query(
                    TrainingTask,
                    "training",
                    statuses=status_set,
                    hours=params.within_hours,
                    baseline_id=params.baseline_id,
                    dataset_id=params.dataset_id,
                )
            )
        elif tt == TaskType.EVALUATION:
            queries.append(
                build_query(
                    EvaluationTask,
                    "evaluation",
                    statuses=status_set,
                    hours=params.within_hours,
                    dataset_id=params.dataset_id,
                )
            )

    return queries
