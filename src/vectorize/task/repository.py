"""Tasks repository."""

from collections.abc import Sequence
from typing import Any

from loguru import logger
from sqlalchemy import select, union_all
from sqlmodel import text
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.dataset.task_model import UploadDatasetTask
from vectorize.evaluation.models import EvaluationTask
from vectorize.synthesis.models import SynthesisTask
from vectorize.task.task_type import TaskType
from vectorize.training.models import TrainingTask
from vectorize.upload.models import UploadTask

from .query_builder import build_query
from .schemas import TaskFilters

__all__ = ["get_tasks_db"]


async def get_tasks_db(db: AsyncSession, params: TaskFilters) -> Sequence:
    """Retrieve tasks from database with filtering and pagination.

    Aggregates tasks from multiple types (upload, synthesis, dataset) with
    comprehensive filtering and pagination support.

    Args:
        db: Database session for executing queries.
        params: Filter parameters containing limit, offset, completed status,
                specific statuses, and time window criteria.

    Returns:
        Sequence of task action rows ordered by creation date (newest first).
        Each row contains id, task_status, created_at, end_date, and task_type.
    """
    status_set = set(params.statuses or [])

    task_types: list[TaskType] = params.task_types or [
        TaskType.MODEL_UPLOAD,
        TaskType.SYNTHESIS,
        TaskType.DATASET_UPLOAD,
        TaskType.TRAINING,
        TaskType.EVALUATION,
    ]

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
                )
            )
        elif tt == TaskType.EVALUATION:
            queries.append(
                build_query(
                    EvaluationTask,
                    "evaluation",
                    statuses=status_set,
                    hours=params.within_hours,
                )
            )

    if not queries:
        return []

    combined = queries[0] if len(queries) == 1 else union_all(*queries)
    tasks_sq = combined.subquery("tasks_sq")
    stmt = select(tasks_sq)

    bind_params: dict[str, Any] = {}
    if params.tag:
        stmt = stmt.where(tasks_sq.c.tag == text(":tag"))
        bind_params["tag"] = params.tag

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
    logger.debug("Tasks fetched from DB", count=len(rows), params=str(params))
    return rows
