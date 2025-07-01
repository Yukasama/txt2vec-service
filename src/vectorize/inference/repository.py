"""Inference repository."""

from uuid import UUID

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.models import AIModel
from vectorize.inference.models import InferenceCounter

__all__ = ["create_inference_counter", "get_model_count"]


async def create_inference_counter(db: AsyncSession, ai_model_id: UUID) -> UUID:
    """Save a new inference counter for the specified AI model.

    Records an inference usage event for the given AI model.

    Args:
        db: Database session instance.
        ai_model_id: The UUID of the AI model used for inference.

    Returns:
        UUID: The UUID of the created inference counter record.
    """
    counter = InferenceCounter(ai_model_id=ai_model_id)

    db.add(counter)
    await db.commit()
    await db.refresh(counter)

    logger.debug("Inference counter saved", model_id=ai_model_id, counter_id=counter.id)
    return counter.id


async def get_model_count(
    db: AsyncSession, ai_model_tag: str
) -> list[InferenceCounter]:
    """Get all inference counters for a specific AI model.

    Args:
        db: Database session instance.
        ai_model_tag: The tag of the AI model.

    Returns:
        list[InferenceCounter]: List of inference counter records for the model.
    """
    statement = select(AIModel).where(AIModel.model_tag == ai_model_tag)
    result = await db.exec(statement)
    ai_model = result.first()

    if ai_model is None:
        logger.debug("AI Model not found", model_tag=ai_model_tag)
        raise ModelNotFoundError(ai_model_tag)

    counter_statement = select(InferenceCounter).where(
        InferenceCounter.ai_model_id == ai_model.id
    )
    results = await db.exec(counter_statement)
    results_list = list(results)
    logger.debug("Model count of AI Model", length=results_list)
    return results_list
