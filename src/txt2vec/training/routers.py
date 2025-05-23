"""Router Initializer für Training."""

from fastapi import APIRouter, FastAPI
from txt2vec.training.router import router as training_router

def register_training_routers(app: FastAPI) -> None:
    """Register only the training API routers with the FastAPI application."""
    base_router = APIRouter()
    base_router.include_router(training_router, prefix="/training")
    app.include_router(base_router)
