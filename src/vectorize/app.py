"""Main application module for the Vectorize service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import config_logger, engine, seed_db, settings
from vectorize.utils.banner import create_banner
from vectorize.utils.cache import initialize_cache
from vectorize.utils.error_handler import register_exception_handlers
from vectorize.utils.prometheus import add_prometheus_metrics
from vectorize.utils.routers import register_routers

config_logger()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Initialize resources on startup."""
    await makedirs(settings.dataset_upload_dir, exist_ok=True)
    await makedirs(settings.model_upload_dir, exist_ok=True)

    create_banner(settings)

    async with engine.begin() as conn:
        if settings.clear_db_on_restart:
            await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    if settings.seed_db_on_start:
        async with AsyncSession(engine) as session:
            await seed_db(session)

    await initialize_cache(max_preload=3)

    yield
    await engine.dispose()


app: Final = FastAPI(
    title="Vectorize",
    description="Service for text embedding and vector operations",
    root_path=settings.root_path,
    version=settings.version,
    lifespan=lifespan,
)


# --------------------------------------------------------
# P R O M E T H E U S
# --------------------------------------------------------
Instrumentator().instrument(app).expose(app, include_in_schema=False)
add_prometheus_metrics(app)


# --------------------------------------------------------
# C O R S
# --------------------------------------------------------
if settings.app_env != "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origin_in_dev,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --------------------------------------------------------
# R O U T E R S
# --------------------------------------------------------
register_routers(app)


# --------------------------------------------------------
# E X C E P T I O N S
# --------------------------------------------------------
register_exception_handlers(app)
