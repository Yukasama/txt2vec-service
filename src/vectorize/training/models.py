"""Training Task model."""

# pyright: reportAssignmentType=false

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

from vectorize.common.task_status import TaskStatus

if TYPE_CHECKING:
    from vectorize.ai_model.models import AIModel

__all__ = ["TrainingTask"]


class TrainingTask(SQLModel, table=True):
    """Model training task."""

    __tablename__ = "training_task"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        description="Unique identifier for the training task.",
    )

    task_status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        index=True,
        description="Current status of the model training.",
    )

    end_date: datetime | None = Field(
        default=None, description="Optional end time of the training task."
    )

    error_msg: str | None = Field(
        default=None,
        description="Optional error message encountered during training.",
    )

    trained_model: Optional["AIModel"] = Relationship(back_populates="training_task")

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), insert_default=func.now()),
        description="Timestamp when the training task was created.",
    )

    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), onupdate=func.now(), insert_default=func.now()
        ),
        description="Timestamp when the training task was last updated.",
    )
