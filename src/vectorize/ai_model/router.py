"""AIModel router."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    Response,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import get_session

from .models import AIModelPublic, AIModelUpdate
from .repository import get_models_paged_db
from .schemas import PagedResponse
from .service import delete_model_svc, get_ai_model_svc, update_ai_model_svc

__all__ = ["router"]


router = APIRouter(tags=["AIModel"])


@router.get("", summary="Get all AIModels")
async def get_ai_models(
    db: Annotated[AsyncSession, Depends(get_session)],
    page: Annotated[int, Query(ge=1, description="Page number, starts at 1")] = 1,
    size: Annotated[int, Query(ge=5, le=100, description="Items per page")] = 5,
) -> PagedResponse:
    """Returns a paged response of AI models sorted by recent inference (30 days).

    Args:
        page (int, optional): The page number. Defaults to 1.
        size (int, optional): The page size. Defaults to 5.
        db: Database session.

    Returns:
        PagedResponse: Paginated list of AI models for the requested page.

    Raises:
        NoModelFoundError: If no models exist in the database.
    """
    items, total = await get_models_paged_db(db, page, size)
    totalpages = (total + size - 1) // size

    response = PagedResponse(page=page, size=size, totalpages=totalpages, items=items)
    logger.debug(
        "AIModels retrieved",
        page=page,
        size=size,
        totalpages=totalpages,
        total=total,
        items_count=len(response.items),
    )
    return response


@router.get("/{ai_model_tag}", response_model=None, summary="Get AIModel by tag")
async def get_ai_model(
    ai_model_tag: str,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> AIModelPublic | Response:
    """Retrieve a single AI model by its ID.

    Args:
        ai_model_tag: The tag of the AI model to retrieve
        request: The HTTP request object
        response: FastAPI response object for setting headers
        db: Database session for persistence operations

    Returns:
        The AI model object, or 304 Not Modified if the ETag matches the current version

    Raises:
        ModelNotFoundError: If the AI model with the specified ID doesn't exist
    """
    ai_model, version = await get_ai_model_svc(db, ai_model_tag)
    response.headers["ETag"] = f'"{version}"'
    etag = f'"{version}"'

    client_match = request.headers.get("If-None-Match")
    if client_match and client_match.strip('"') == str(version):
        logger.debug("AIModel not modified", ai_model_tag=ai_model_tag, version=version)
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag}
        )

    logger.debug("AIModel retrieved", ai_model_tag=ai_model_tag, version=version)
    return ai_model


@router.put("/{ai_model_id}", summary="Update AIModel by ID")
async def update_ai_model(
    ai_model_id: UUID,
    request: Request,
    ai_model: AIModelUpdate,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Update an AI model with version control using ETags.

    Updates an AI model by its ID, requiring an If-Match header with the current
    version to prevent concurrent modification issues.

    Args:
        ai_model_id: The UUID of the AIModel to update
        request: The HTTP request object containing If-Match header
        ai_model: The updated AIModel object
        db: Database session for persistence operations

    Returns:
        204 No Content response with Location header

    Raises:
        VersionMismatchError: If the ETag doesn't match current version
        VersionMissingError: If the If-Match header is missing
        ModelNotFoundError: If the AIModel doesn't exist
    """
    new_version = await update_ai_model_svc(db, request, ai_model_id, ai_model)
    logger.debug("AIModel updated", ai_model_id=ai_model_id)

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Location": f"{request.url.path}", "ETag": f'"{new_version}"'},
    )


@router.delete("/{model_id}", summary="Delete AIModel by ID")
async def delete_model(
    model_id: UUID, db: Annotated[AsyncSession, Depends(get_session)]
) -> Response:
    """Delete an AI model by its ID.

    Args:
        model_id: The UUID of the model to delete.
        db: Database session.

    Returns:
        204 No Content if deletion is successful.

    Raises:
        ModelNotFoundError: If no model with that ID exists.
    """
    await delete_model_svc(db, model_id)
    logger.debug("Model deleted", modelId=model_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
