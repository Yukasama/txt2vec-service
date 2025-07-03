"""Helpers for tasks repository."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, String, cast, func, literal, or_, select

from vectorize.ai_model.models import AIModel

from .task_status import TaskStatus

__all__ = ["build_query"]


_UNFINISHED = {TaskStatus.QUEUED, TaskStatus.RUNNING}


def build_query(  # noqa: ANN201, PLR0913
    model,  # noqa: ANN001
    tag: str,
    *,
    statuses: set[TaskStatus],
    hours: int,
    tag_filter: str | None = None,
    baseline_id: UUID | None = None,
    dataset_id: UUID | None = None,
):
    """Build SQL query for a task model with common filters applied.

    Args:
        model: SQLModel table class to query.
        tag: String identifier for task type.
        statuses: Set of task statuses to include.
        hours: Time-window in hours for filtering.
        tag_filter: Optional tag value to filter by.
        baseline_id: Optional baseline model ID to filter by (TrainingTask only).
        dataset_id: Optional dataset ID to filter by (TrainingTask and EvaluationTask).

    Returns:
        SQLAlchemy *Select* query.
    """
    model_table = model.__table__
    ai_table = AIModel.__table__  # type: ignore

    if baseline_id and not hasattr(model, "baseline_model_id"):
        return select(*_get_base_columns(model, tag)).where(literal(False))

    if dataset_id and not (
        hasattr(model, "train_dataset_ids") or hasattr(model, "evaluation_dataset_ids")
    ):
        return select(*_get_base_columns(model, tag)).where(literal(False))

    if hasattr(model, "trained_model_id"):
        join_expr = model_table.outerjoin(
            ai_table,
            func.coalesce(
                model_table.c.trained_model_id, model_table.c.baseline_model_id
            )
            == ai_table.c.id,
        )
        tag_col = ai_table.c.model_tag
    elif hasattr(model, "model_tag"):
        join_expr = model_table
        tag_col = model_table.c.model_tag
    elif hasattr(model, "tag"):
        join_expr = model_table
        tag_col = model_table.c.tag
    else:
        join_expr = model_table
        tag_col = literal(None)

    if hasattr(model, "baseline_model_id"):
        baseline_col = model_table.c.baseline_model_id.label("baseline_id")
    else:
        baseline_col = literal(None).label("baseline_id")

    base_columns = [
        model_table.c.id,
        tag_col.label("tag"),
        model_table.c.task_status,
        model_table.c.created_at,
        model_table.c.end_date,
        model_table.c.error_msg,
        cast(literal(tag), String).label("task_type"),
        baseline_col,
    ]

    query = (
        select(*base_columns)
        .select_from(join_expr)
        .where(_time_filter(model, hours=hours))
    )

    if statuses:
        query = query.where(model_table.c.task_status.in_(statuses))

    if tag_filter and tag_col is not literal(None):
        query = query.where(tag_col == tag_filter)

    if baseline_id and hasattr(model, "baseline_model_id"):
        query = query.where(model_table.c.baseline_model_id == str(baseline_id))

    if dataset_id:
        query = _apply_dataset_filter(query, model, dataset_id)

    return query


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _apply_dataset_filter(query, model, dataset_id: UUID):  # noqa: ANN001, ANN202
    """Apply dataset ID filter to query based on model capabilities.

    Args:
        query: SQLAlchemy query to filter.
        model: SQLModel table class being queried.
        dataset_id: Dataset ID to filter by.

    Returns:
        Filtered SQLAlchemy query.
    """
    model_table = model.__table__
    dataset_id_str = str(dataset_id)

    if hasattr(model, "train_dataset_ids"):
        return query.where(
            func.json_extract(model_table.c.train_dataset_ids, "$").like(
                f'%"{dataset_id_str}"%'
            )
        )
    if hasattr(model, "evaluation_dataset_ids"):
        return query.where(
            func.json_extract(model_table.c.evaluation_dataset_ids, "$").like(
                f'%"{dataset_id_str}"%'
            )
        )

    return query


def _get_base_columns(model, tag: str) -> list[Any]:  # noqa: ANN001
    """Get base columns for consistent empty query structure."""
    model_table = model.__table__

    if hasattr(model, "baseline_model_id"):
        baseline_col = model_table.c.baseline_model_id.label("baseline_id")
    else:
        baseline_col = literal(None).label("baseline_id")

    return [
        model_table.c.id,
        literal(None).label("tag"),
        model_table.c.task_status,
        model_table.c.created_at,
        model_table.c.end_date,
        model_table.c.error_msg,
        cast(literal(tag), String).label("task_type"),
        baseline_col,
    ]


def _time_filter(model, *, hours: int) -> ColumnElement[bool]:  # noqa: ANN001
    """Build SQL filter to limit results to recent tasks.

    Args:
        model: SQLModel table class to filter.
        hours: Time window in hours for recent tasks.

    Returns:
        SQLAlchemy filter condition for unfinished or recently completed tasks.
    """
    threshold = datetime.now(tz=UTC) - timedelta(hours=hours)
    return or_(
        model.task_status.in_(_UNFINISHED),
        func.coalesce(model.end_date, func.now()) >= threshold,
    )
