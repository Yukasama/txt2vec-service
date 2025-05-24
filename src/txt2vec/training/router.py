"""Training router."""

from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.config.db import get_session

from .schemas import TrainRequest
from .tasks import train_model_task
from .exceptions import TrainingDatasetNotFoundError

__all__ = ["router"]

router = APIRouter(tags=["Training"])


@router.post("/train")
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Start model training as a background task."""
    # Check if dataset file exists before starting background task
    dataset_path = Path(train_request.dataset_path)
    if not dataset_path.is_file():
        logger.error(
            "Training request failed: Dataset file not found: %s", train_request.dataset_path
        )
        raise TrainingDatasetNotFoundError(train_request.dataset_path)
    logger.info(
        "Training requested for model_tag=%s, dataset_path=%s",
        train_request.model_tag,
        train_request.dataset_path,
    )
    background_tasks.add_task(train_model_task, train_request)
    logger.info(
        "Training started in background for model_tag=%s", train_request.model_tag
    )
    return Response(
        content=(
            f'{{"message": "Training started", "model_tag": "{train_request.model_tag}"}}'
        ),
        status_code=status.HTTP_202_ACCEPTED,
        media_type="application/json",
    )
