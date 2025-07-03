"""Model for action responses."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from .task_status import TaskStatus

__all__ = ["TaskModel"]


class TaskModel(BaseModel):
    """Pydantic model for action responses."""

    id: UUID
    tag: str | None = None
    task_status: TaskStatus
    baseline_id: UUID | None = None
    created_at: datetime
    end_date: datetime | None
    error_msg: str | None = None
    task_type: Literal[
        "model_upload", "synthesis", "dataset_upload", "training", "evaluation"
    ]
