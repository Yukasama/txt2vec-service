"""Cache eviction strategies for model management."""

from collections import OrderedDict

from loguru import logger

from .usage_tracker import UsageTracker

__all__ = ["CacheEviction"]


class CacheEviction:
    """Manages cache eviction based on usage patterns."""

    def __init__(self, usage_tracker: UsageTracker, max_models: int) -> None:
        """Initialize the cache eviction manager.

        Args:
            usage_tracker: Tracker for model usage patterns
            max_models: Maximum number of models to keep in cache
        """
        self.usage_tracker = usage_tracker
        self.max_models = max_models

    def should_evict(self, cache_size: int) -> bool:
        """Check if eviction is necessary."""
        return cache_size >= self.max_models

    def select_eviction_candidate(self, cache: OrderedDict) -> str | None:
        """Select model for eviction based on usage score."""
        if not cache:
            return None

        scores = {}
        for model_tag in cache:
            scores[model_tag] = self.usage_tracker.calculate_score(model_tag)

        return min(scores.keys(), key=lambda x: scores[x])

    def evict_model(self, cache: OrderedDict, model_tag: str) -> None:
        """Remove model from cache and free memory."""
        if model_tag not in cache:
            return

        evicted_model = cache.pop(model_tag)

        try:
            if hasattr(evicted_model[0], "cpu"):
                evicted_model[0].cpu()
            del evicted_model
        except Exception as e:
            logger.warning("Error during model cleanup", error=str(e))

        score = self.usage_tracker.calculate_score(model_tag)
        logger.info(
            "Model evicted", model=model_tag, score=score, cache_size=len(cache)
        )
