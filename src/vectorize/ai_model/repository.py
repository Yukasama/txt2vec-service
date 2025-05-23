"""AIModel repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import VersionMismatchError

from .exceptions import ModelNotFoundError
from .models import AIModel, AIModelUpdate

__all__ = ["get_ai_model_db", "save_ai_model_db", "update_ai_model_db"]


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
