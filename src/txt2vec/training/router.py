from fastapi import APIRouter, BackgroundTasks, Depends
from typing import Annotated
from sqlmodel.ext.asyncio.session import AsyncSession
from txt2vec.config.db import get_session
from .schemas import TrainRequest
from .tasks import train_model_task

router = APIRouter(tags=["Training"])

@router.post("/train", status_code=202)
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """Starte das Training eines Modells mit einem Datensatz im Hintergrund."""
    background_tasks.add_task(train_model_task, train_request)
    return {"message": "Training gestartet", "model_tag": train_request.model_tag}
