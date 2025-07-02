"""AIModel service."""

from uuid import UUID

from fastapi import Request
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.utils.etag_parser import parse_etag

from .models import AIModelPublic, AIModelUpdate
from .repository import delete_model_db, get_ai_model_db, update_ai_model_db

__all__ = ["delete_model_svc", "get_ai_model_svc", "update_ai_model_svc"]


async def get_ai_model_svc(
    db: AsyncSession, ai_model_identifier: str
) -> tuple[AIModelPublic, int]:
    """Read a single AI model from the database.

    This function retrieves an AI model by its ID from the database and returns it
    as a dictionary. The dictionary contains all fields of the AI model.

    Args:
        db: Database session for persistence operations
        ai_model_identifier: The tag or ID of the AI model to retrieve

    Returns:
        Dictionary representing the AI model with all fields.
    """
    ai_model = await get_ai_model_db(db, ai_model_identifier)
    return AIModelPublic.model_validate(ai_model), ai_model.version


async def update_ai_model_svc(
    db: AsyncSession, request: Request, ai_model_id: UUID, update_data: AIModelUpdate
) -> int:
    """Update an AI model in the database.

    This function updates an existing AI model in the database with the provided
    data. It returns the updated version.

    Args:
        db: Database session for persistence operations
        request: The HTTP request object
        ai_model_id: The UUID of the AI model to update
        update_data: The updated AI model data

    Returns:
        The updated version of the AI model.
    """
    expected_version = parse_etag(str(ai_model_id), request)

    updated_model = await update_ai_model_db(
        db, ai_model_id, update_data, expected_version
    )

    logger.debug("AIModel updated", ai_model_id=ai_model_id)
    return updated_model.version


async def delete_model_svc(db: AsyncSession, model_id: UUID) -> None:
    """Delete an AI model by its ID from the database.

    Args:
        db: Database session.
        model_id: UUID of the model to delete.
    """
    await delete_model_db(db, model_id)
    logger.debug("Model deleted from service", model_id=model_id)
