"""AIModel repository."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

from loguru import logger
from sqlmodel import and_, func, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import VersionMismatchError
from vectorize.config import settings

from .exceptions import ModelNotFoundError
from .models import AIModel, AIModelUpdate
from .utils.model_deletion import remove_model_from_memory

__all__ = [
    "delete_model_db",
    "get_ai_model_db",
    "get_models_paged_db",
    "save_ai_model_db",
    "update_ai_model_db",
]


async def get_models_paged_db(
    db: AsyncSession,
    page: int = 1,
    size: int = 5,
) -> tuple[Sequence[AIModel], int]:
    """Fetches a all ai models paged and sorted by recent inference (30 days).

    Args:
        db (AsyncSession): The database session.
        page (int, optional): Page number, starts at 1. Defaults to 1.
        size (int, optional): Number of items per page. Defaults to 5.

    Returns:
        Tuple[Sequence[AIModel], int]: A tuple containing the list of AIModel
        objects for the requested page and the total number of models in the database.

    Raises:
        NoModelFoundError: If there are no models in the database.
    """
    from vectorize.inference.models import InferenceCounter  # noqa: PLC0415
    page = max(1, page)
    size = max(1, size)
    cutoff_date = datetime.now(UTC) - timedelta(days=settings.model_inference_limiter)
    offset = (page - 1) * size

    statement = (
            select(
                AIModel,
                func.count(InferenceCounter.id).label("inference_count")  # type: ignore[reportArgumentType]
            )
            .join(
                InferenceCounter,
                onclause=and_(  # type: ignore[reportArgumentType]
                    AIModel.id == InferenceCounter.ai_model_id,
                    or_(
                        InferenceCounter.created_at.is_(None),  # type: ignore[reportArgumentType]
                        InferenceCounter.created_at >= cutoff_date
                    )
                ),
                isouter=True
            )
            .group_by(AIModel.id)  # type: ignore[reportArgumentType]
            .order_by(func.count(InferenceCounter.id).desc())  # type: ignore[reportArgumentType]
            .offset(offset)
            .limit(size)
        )

    results = await db.exec(statement)
    models_with_counts = results.all()

    models = [ai_model for ai_model, _ in models_with_counts]

    total_count_stmt = (
        select(func.count(func.distinct(AIModel.id)))
        .select_from(AIModel)
        .join(
            InferenceCounter,
            onclause=and_(
                AIModel.id == InferenceCounter.ai_model_id,
                or_(
                    InferenceCounter.created_at.is_(None),  # type: ignore[reportArgumentType]
                    InferenceCounter.created_at >= cutoff_date
                )
            ),
            isouter=True
        )
    )
    total_count = await db.scalar(total_count_stmt) or 0  # type: ignore[reportArgumentType]

    if total_count == 0:
        raise ModelNotFoundError()

    return models, total_count


async def get_ai_model_db(db: AsyncSession, model_tag: str) -> AIModel:
    """Retrieve an AI model by its ID.

    Args:
        db: Database session instance.
        model_tag: The Tag of the model to retrieve.

    Returns:
        AIModel: The AI model object corresponding to the given Model Tag.

    Raises:
        ModelNotFoundError: If the model is not found.
    """
    statement = select(AIModel).where(AIModel.model_tag == model_tag)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(model_tag)

    logger.debug("AI Model loaded from DB", ai_model=model)
    return model


async def save_ai_model_db(db: AsyncSession, model: AIModel) -> UUID:
    """Persist model to database with a unique tag.

    Args:
        db: Database session instance.
        model: The AI model object to save.

    Returns:
        UUID: The ID of the saved AI model.
    """
    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.debug("AI Model saved to DB", ai_model=model)
    return model.id


async def update_ai_model_db(
    db: AsyncSession, model_id: UUID, update_data: AIModelUpdate, expected_version: int
) -> AIModel:
    """Update an AIModel using optimistic locking.

    Args:
        db: The database session instance.
        model_id: The unique identifier of the AI model to update.
        update_data: The Pydantic model containing the fields to update.
        expected_version: The version expected by the client (for optimistic locking).

    Returns:
        AIModel: The updated AI model object.

    Raises:
        ModelNotFoundError: If the AI model is not found.
        VersionMismatchError: If the version does not match (lost update).
    """
    result = await db.exec(select(AIModel).where(AIModel.id == model_id))
    model = result.first()
    if model is None:
        raise ModelNotFoundError(model_id)

    if model.version != expected_version:
        raise VersionMismatchError(model_id, model.version)

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(model, field, value)

    model.version += 1

    await db.commit()
    await db.refresh(model)

    logger.debug("AIModel updated", ai_model=model)
    return model


async def delete_model_db(db: AsyncSession, model_id: UUID) -> None:
    """Delete an AI model from the database by ID.

    Args:
        db: Database session instance.
        model_id: The UUID of the AI model to delete.

    Raises:
        ModelNotFoundError: If the model with the given ID does not exist.
    """
    statement = select(AIModel).where(AIModel.id == model_id)
    result = await db.exec(statement)
    model = result.first()

    if model is None:
        raise ModelNotFoundError(str(model_id))

    await db.delete(model)
    await db.commit()
    logger.debug("Model deleted", model=model)
    await remove_model_from_memory(model.model_tag)
