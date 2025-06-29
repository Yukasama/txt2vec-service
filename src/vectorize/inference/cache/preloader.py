"""Module for preloading models into cache based on usage statistics."""

import asyncio
import operator
from collections.abc import Callable
from typing import Any

from loguru import logger

from .usage_tracker import UsageTracker

__all__ = ["CachePreloader"]


class CachePreloader:
    """Loads models at server startup based on usage statistics."""

    def __init__(self, usage_tracker: UsageTracker) -> None:
        """Initialize the CachePreloader.

        Args:
            usage_tracker: Tracker for model usage statistics
        """
        self.usage_tracker = usage_tracker

    def get_preload_candidates(self, max_preload: int = 3) -> list[str]:
        """Determine which models should be loaded at startup.

        Args:
            max_preload: Maximum number of models to preload

        Returns:
            List of model_tags sorted by priority
        """
        stats = self.usage_tracker.get_stats()

        if not stats:
            logger.info("No usage stats found - skipping preload")
            return []

        model_scores = []
        for model_tag, data in stats.items():
            score = data["count"]
            model_scores.append((model_tag, score))
        model_scores.sort(key=operator.itemgetter(1), reverse=True)

        candidates = [model_tag for model_tag, _ in model_scores[:max_preload]]

        logger.info(
            "Preload candidates selected",
            candidates=candidates,
            total_models_in_stats=len(stats),
        )

        return candidates

    @staticmethod
    async def preload_models_async(
        candidates: list[str],
        loader_func: Callable[[str], Any],
        cache_store_func: Callable[[str, Any], None],
    ) -> int:
        """Load models asynchronously at server startup.

        Args:
            candidates: List of model_tags to load
            loader_func: Function to load models
            cache_store_func: Function to store in cache

        Returns:
            Number of successfully loaded models
        """
        if not candidates:
            return 0

        logger.info("Starting model preload", models=candidates)
        loaded_count = 0

        for model_tag in candidates:
            try:
                logger.info("Preloading model", model=model_tag)

                model_data = await asyncio.get_event_loop().run_in_executor(
                    None, loader_func, model_tag
                )

                cache_store_func(model_tag, model_data)
                loaded_count += 1

                logger.info(
                    "Model preloaded successfully",
                    model=model_tag,
                    loaded_count=loaded_count,
                )

            except Exception as e:
                logger.warning("Failed to preload model", model=model_tag, error=str(e))
                continue

        logger.info(
            "Model preload completed", loaded=loaded_count, total=len(candidates)
        )

        return loaded_count
