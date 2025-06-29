"""Factory module for creating model caches based on configuration settings."""

import os
from pathlib import Path

from loguru import logger

from vectorize.config import settings

from .model_cache import ModelCache
from .vram_model_cache import VRAMModelCache

__all__ = ["create_model_cache"]


def _get_cache_file_path() -> str:
    """Get cache file path from environment or use default."""
    cache_dir = os.getenv("CACHE_DIR", "data/cache")
    cache_path = Path(cache_dir) / "model_usage_stats.json"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache directory created/verified", path=str(cache_path.parent))
    except OSError as e:
        logger.warning("Could not create cache directory", error=str(e))

    return str(cache_path)


def create_model_cache(
    cache_file: str | None = None,
) -> ModelCache | VRAMModelCache:
    """Factory-Funktion für Cache basierend auf Konfiguration."""
    if cache_file is None:
        cache_file = _get_cache_file_path()

    if settings.cache_strategy == "fixed_size":
        logger.info(
            "Creating fixed-size model cache",
            max_models=settings.cache_max_models,
            cache_file=cache_file,
        )

        return ModelCache(max_models=settings.cache_max_models, cache_file=cache_file)

    if settings.cache_strategy == "vram_aware":
        logger.info(
            "Creating VRAM-aware model cache",
            safety_margin_gb=settings.cache_vram_safety_margin_gb,
            cache_file=cache_file,
        )

        return VRAMModelCache(
            safety_margin_gb=settings.cache_vram_safety_margin_gb, cache_file=cache_file
        )

    logger.error("Unknown cache strategy", strategy=settings.cache_strategy)
    logger.warning("Falling back to fixed_size strategy")
    return ModelCache(max_models=5, cache_file=cache_file)
