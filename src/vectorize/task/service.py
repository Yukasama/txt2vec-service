"""Taks service."""

from sqlmodel.ext.asyncio.session import AsyncSession

from .models import TaskModel
from .repository import get_tasks_db
from .schemas import TaskFilters

__all__ = ["get_tasks_svc"]


async def get_tasks_svc(
    db: AsyncSession, params: TaskFilters
) -> tuple[list[TaskModel], int]:
    """Retrieve tasks from the database with filtering.

    Acts as a bridge between the API router and the database repository,
    handling data validation and transformation. It queries the database
    for tasks based on the provided filters and converts the raw database
    rows into validated Pydantic models.

    Args:
        db: Database session for executing queries.
        params: Filter parameters for querying tasks.

    Returns:
        Tuple containing:
        - List of validated Pydantic models representing tasks.
          Each model contains task metadata including ID, status,
          creation date, and type information.
        - Total count of tasks matching the filters (without pagination).
    """
    rows, total = await get_tasks_db(db, params)
    tasks = [TaskModel.model_validate(row._mapping) for row in rows]  # noqa: SLF001
    return tasks, total
