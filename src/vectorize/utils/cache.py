"""Cache initialization and preloading utilities."""

from typing import TYPE_CHECKING

from loguru import logger

from vectorize.inference.cache.preloader import CachePreloader
from vectorize.inference.cache.vram_model_cache import VRAMModelCache
from vectorize.inference.utils.model_cache_wrapper import get_cache
from vectorize.inference.utils.model_loader import load_model

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer

__all__ = ["initialize_cache"]


def _cache_store_func(
    model_tag: str, model_data: tuple["torch.nn.Module", "AutoTokenizer | None"]
) -> None:
    """Store model in the global cache.

    Args:
        model_tag: Identifier for the model
        model_data: Tuple of (model, tokenizer) to store in cache
    """
    cache = get_cache()

    with cache.lock:
        cache.cache[model_tag] = model_data

        if isinstance(cache, VRAMModelCache):
            cache.eviction.track_model_vram(model_tag, model_data[0])

        cache.usage_tracker.track_access(model_tag)
        cache.usage_tracker.save_stats()


async def initialize_cache(max_preload: int = 3) -> None:
    """Initialize and preload models into cache based on usage statistics.

    Args:
        max_preload: Maximum number of models to preload

    Raises:
        Exception: If preloading fails (logged as warning)
    """
    try:
        logger.info("Starting model preloading...")

        cache = get_cache()
        preloader = CachePreloader(cache.usage_tracker)
        candidates = preloader.get_preload_candidates(max_preload=max_preload)

        if candidates:
            logger.info("Preloading models", count=len(candidates), models=candidates)

            loaded_count = await preloader.preload_models_async(
                candidates,
                load_model,
                _cache_store_func,
            )

            logger.info(
                "Model preloading completed",
                loaded=loaded_count,
                total=len(candidates),
            )

            cache_info = cache.get_info()
            logger.info(
                "Cache status after preload",
                cached_models=cache_info.get("cache_size", 0),
            )

        else:
            logger.info("No models to preload (no usage statistics available)")

    except Exception as e:
        logger.warning("Model preloading failed", error=str(e))
        logger.debug("Preloader error details", exc_info=True)
