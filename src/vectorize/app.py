"""Main application module for the Vectorize service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Final

from aiofiles.os import makedirs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import config_logger, engine, seed_db, settings
from vectorize.inference.cache.preloader import CachePreloader
from vectorize.inference.utils.model_loader import load_model
from vectorize.utils.banner import create_banner
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
    try:
        logger.info("Starting model preloading...")

        from vectorize.inference.utils.model_cache_wrapper import _cache  # noqa

        preloader = CachePreloader(_cache.usage_tracker)

        candidates = preloader.get_preload_candidates(max_preload=3)

        if candidates:
            logger.info(f"Preloading {len(candidates)} models: {candidates}")

            def cache_store_func(model_tag: str, model_data) -> None:  # noqa
                """Store model in the global cache."""
                with _cache.lock:
                    _cache.cache[model_tag] = model_data

                    from vectorize.inference.cache.vram_model_cache import (  # noqa
                        VRAMModelCache,
                    )

                    if isinstance(_cache, VRAMModelCache):
                        _cache.eviction.track_model_vram(model_tag, model_data[0])

                    _cache.usage_tracker.track_access(model_tag)

                    _cache.usage_tracker.save_stats()

            loaded_count = await preloader.preload_models_async(
                candidates,
                load_model,
                cache_store_func,
            )

            logger.info(
                f"Successfully preloaded {loaded_count}/{len(candidates)} models"
            )

            cache_info = _cache.get_info()

            logger.info(
                f"Global cache status: {cache_info.get('cache_size', 0)} models cached"
            )

        else:
            logger.info("No models to preload (no usage statistics available)")

    except Exception as e:
        logger.warning(f"Model preloading failed: {e}")

        logger.debug("Preloader error details", exc_info=True)

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
