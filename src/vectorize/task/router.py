"""Tasks router."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session
from vectorize.dataset.pagination import Page

from .models import TaskModel
from .schemas import TaskFilters
from .service import get_tasks_svc
from .task_status import TaskStatus
from .task_type import TaskType

__all__ = ["router"]


router = APIRouter(tags=["Tasks"])


@router.get("", summary="Get filterable tasks")
async def get_tasks(  # noqa: PLR0913, PLR0917
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
    tag: Annotated[str | None, Query(max_length=100)] = None,
    baseline_id: Annotated[UUID | None, Query()] = None,
    dataset_id: Annotated[UUID | None, Query()] = None,
    task_type: Annotated[list[TaskType] | None, Query()] = None,
    status: Annotated[list[TaskStatus] | None, Query()] = None,
    within_hours: Annotated[int, Query(ge=1)] = 1,
) -> Page[TaskModel]:
    """Get tasks with filtering and pagination.

    Args:
        db: Database session for queries.
        limit: Maximum number of records to return (default 100).
        offset: Number of records to skip (default 0).
        tag: Filter tasks by specific tag.
        baseline_id: Filter tasks by specific baseline ID.
        dataset_id: Filter tasks by specific dataset ID.
        task_type: Filter tasks by specific type (e.g., upload, synthesis).
        status: Filter by specific task statuses.
        within_hours: Time window in hours to filter tasks (default 1).

    Returns:
        Paginated list of task action models with metadata.
    """
    task_filters = TaskFilters(
        limit=limit,
        offset=offset,
        tag=tag,
        baseline_id=baseline_id,
        dataset_id=dataset_id,
        task_types=task_type,
        statuses=status,
        within_hours=within_hours,
    )

    logger.debug("Fetching tasks with parameters", filters=str(task_filters))
    tasks, total = await get_tasks_svc(db, task_filters)

    logger.debug("Tasks retrieved", length=len(tasks), total=total)
    return Page[TaskModel](items=tasks, total=total, limit=limit, offset=offset)
