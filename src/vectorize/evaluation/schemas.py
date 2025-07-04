"""Schemas for evaluation endpoints."""

import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from vectorize.config import settings
from vectorize.task.task_status import TaskStatus

from .models import EvaluationTask

__all__ = ["EvaluationRequest", "EvaluationResponse", "EvaluationStatusResponse"]


class EvaluationRequest(BaseModel):
    """Request for evaluating a trained model."""

    model_tag: str = Field(description="Tag of the trained model from the database")
    dataset_id: str | None = Field(
        default=None, description="ID of the dataset to use for evaluation"
    )
    max_samples: int | None = Field(
        default=settings.evaluation_default_max_samples,
        description="Maximum number of samples to evaluate (default: 1000)",
        gt=0,
    )
    baseline_model_tag: str | None = Field(
        default=None, description="Optional tag of baseline model for comparison"
    )
    training_task_id: str | None = Field(
        default=None,
        description="Optional training task ID to use its validation dataset",
    )


class EvaluationResponse(BaseModel):
    """Response for model evaluation."""

    model_tag: str
    dataset_used: str
    metrics: dict
    baseline_metrics: dict | None = None
    evaluation_summary: str


class EvaluationStatusResponse(BaseModel):
    """Response for evaluation task status."""

    task_id: UUID
    status: TaskStatus
    progress: float
    created_at: datetime
    updated_at: datetime
    end_date: datetime | None = None
    error_msg: str | None = None
    evaluation_metrics: dict | None = None
    baseline_metrics: dict | None = None
    evaluation_summary: str | None = None
    model_tag: str | None = None
    baseline_model_tag: str | None = None
    evaluation_dataset_ids: list[str] = []

    @classmethod
    def from_task(cls, task: EvaluationTask) -> "EvaluationStatusResponse":
        """Create response from EvaluationTask model.

        Args:
            task: EvaluationTask instance

        Returns:
            EvaluationStatusResponse instance
        """
        evaluation_metrics = None
        if task.evaluation_metrics:
            try:
                evaluation_metrics = json.loads(task.evaluation_metrics)
            except json.JSONDecodeError:
                evaluation_metrics = None

        baseline_metrics = None
        if task.baseline_metrics:
            try:
                baseline_metrics = json.loads(task.baseline_metrics)
            except json.JSONDecodeError:
                baseline_metrics = None

        return cls(
            task_id=task.id,
            status=task.task_status,
            progress=task.progress,
            created_at=task.created_at,
            updated_at=task.updated_at,
            end_date=task.end_date,
            error_msg=task.error_msg,
            evaluation_metrics=evaluation_metrics,
            baseline_metrics=baseline_metrics,
            evaluation_summary=task.evaluation_summary,
            model_tag=task.model_tag,
            baseline_model_tag=task.baseline_model_tag,
            evaluation_dataset_ids=getattr(task, "evaluation_dataset_ids", []) or [],
        )
