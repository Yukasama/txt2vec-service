"""Helpers for tasks repository."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import ColumnElement, String, cast, func, literal, or_, select

from vectorize.ai_model.models import AIModel

from .task_status import TaskStatus

__all__ = ["build_query"]


_UNFINISHED = {TaskStatus.QUEUED, TaskStatus.RUNNING}


def build_query(  # noqa: ANN201
    model,  # noqa: ANN001
    tag: str,
    *,
    statuses: set[TaskStatus],
    hours: int,
    tag_filter: str | None = None,
):
    """Build SQL query for a task model with common filters applied.

    Args:
        model: SQLModel table class to query.
        tag: String identifier for task type.
        statuses: Set of task statuses to include.
        hours: Time-window in hours for filtering.
        tag_filter: Optional tag value to filter by.

    Returns:
        SQLAlchemy *Select* query.
    """
    model_table = model.__table__
    ai_table = AIModel.__table__  # type: ignore

    if hasattr(model, "trained_model_id"):
        if hasattr(model, "baseline_model_id"):
            join_expr = model_table.outerjoin(
                ai_table,
                func.coalesce(
                    model_table.c.trained_model_id, model_table.c.baseline_model_id
                )
                == ai_table.c.id,
            )
        else:
            join_expr = model_table.outerjoin(
                ai_table, model_table.c.trained_model_id == ai_table.c.id
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

    query = (
        select(
            model_table.c.id,
            tag_col.label("tag"),
            model_table.c.task_status,
            model_table.c.created_at,
            model_table.c.end_date,
            model_table.c.error_msg,
            cast(literal(tag), String).label("task_type"),
        )
        .select_from(join_expr)
        .where(_time_filter(model, hours=hours))
    )

    if statuses:
        query = query.where(model_table.c.task_status.in_(statuses))

    if tag_filter and tag_col is not literal(None):
        query = query.where(tag_col == tag_filter)

    return query


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


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
