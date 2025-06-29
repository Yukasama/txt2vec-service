"""Inference router."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Response,
    status,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session

from .embedding_model import Embeddings
from .schemas import EmbeddingRequest
from .service import get_embeddings_srv, get_model_stats_srv
from .utils.model_cache_wrapper import (
    clear_model_cache,
    get_cache_status,
)

__all__ = ["router"]


router = APIRouter(tags=["Inference"])


@router.post("", summary="Get embeddings by model")
async def get_embeddings(
    data: EmbeddingRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Embeddings:
    """Get embeddings for an input text.

    Args:
        data: The embedding request containing input text and model specifications.
        db: Database session for retrieving model information.

    Returns:
        Embeddings: OpenAI-compatible response containing the generated
            embeddings and usage statistics.

    Raises:
        ModelNotFoundError: If the requested model cannot be found.
        ModelLoadError: If there's an error loading the AI model.
    """
    return await get_embeddings_srv(db, data)


@router.get("/counter/{model_tag}", summary="Get model inference statistics")
async def get_model_stats(
    model_tag: str,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, int]:
    """Get daily inference statistics for an AI model.

    Returns a dictionary with dates as keys and the number of inferences
    performed on that date as values. Includes all dates from model creation
    to the present, with zero values for dates with no inferences.

    Args:
        model_tag: Tag of the AI model.
        db: Database session.

    Returns:
        Dictionary with dates (format: YYYY-MM-DD) and inference counts.
    """
    return await get_model_stats_srv(db, model_tag)


@router.get("/cache/status", summary="Get model cache status")
async def cache_status() -> dict:
    """Get current model cache status including preload candidates."""
    return get_cache_status()


@router.delete("/cache/clear", summary="Clear model cache")
async def clear_cache() -> Response:
    """Clear all cached models from memory."""
    clear_model_cache()
    return Response(
        status_code=status.HTTP_204_NO_CONTENT, headers={"X-Cache-Status": "cleared"}
    )
