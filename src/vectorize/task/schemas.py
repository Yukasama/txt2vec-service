"""Query and filter schemas for tasks."""

from uuid import UUID

from pydantic import BaseModel, Field

from .task_status import TaskStatus
from .task_type import TaskType

__all__ = ["TaskFilters"]


class TaskFilters(BaseModel):
    """Parameters for filtering tasks."""

    limit: int | None = Field(
        None, ge=1, le=100, description="Maximum number of records to return"
    )
    offset: int | None = Field(None, ge=0, description="Number of records to skip")
    tag: str | None = Field(None, description="Filter tasks by specific tag")
    baseline_id: UUID | None = Field(
        None, description="Filter tasks by specific baseline model ID"
    )
    dataset_id: UUID | None = Field(
        None, description="Filter tasks by specific dataset ID"
    )
    task_types: list[TaskType] | None = Field(
        None, description="Filter tasks by specific type (e.g., upload, synthesis)"
    )
    statuses: list[TaskStatus] | None = Field(
        None, description="Filter by specific task statuses"
    )
    within_hours: int = Field(
        1, ge=1, description="Time window in hours for task filtering"
    )
