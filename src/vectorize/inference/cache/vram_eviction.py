"""VRAM-aware eviction strategy for model cache management."""

import operator
from collections import OrderedDict

import torch
from loguru import logger

from .usage_tracker import UsageTracker
from .vram_monitor import VRAMMonitor

__all__ = ["VRAMEviction"]


class VRAMEviction:
    """VRAM-aware eviction strategy."""

    def __init__(self, usage_tracker: UsageTracker, vram_monitor: VRAMMonitor) -> None:
        """Initialize VRAM eviction strategy.

        Args:
            usage_tracker: Tracker for model usage statistics
            vram_monitor: Monitor for VRAM usage and estimation
        """
        self.usage_tracker = usage_tracker
        self.vram_monitor = vram_monitor
        self.model_vram_sizes: dict[str, int] = {}

    def track_model_vram(self, model_tag: str, model: torch.nn.Module) -> int:
        """Store VRAM size of a model."""
        vram_size = self.vram_monitor.estimate_model_vram(model)
        self.model_vram_sizes[model_tag] = vram_size

        logger.debug(
            "Model VRAM tracked", model=model_tag, vram_gb=vram_size / (1024**3)
        )

        return vram_size

    def needs_eviction_for_new_model(self, estimated_vram: int) -> bool:
        """Check if eviction is needed for new model."""
        return not self.vram_monitor.can_fit_model(estimated_vram)

    def find_models_to_evict(self, cache: OrderedDict, required_vram: int) -> list[str]:
        """Find models that need to be evicted to make space.

        Args:
            cache: Current model cache
            required_vram: Required VRAM in bytes for new model

        Returns:
            List of model_tags to evict (sorted by priority)
        """
        if not self.needs_eviction_for_new_model(required_vram):
            return []

        candidates = []
        for model_tag in cache:
            score = self.usage_tracker.calculate_score(model_tag)
            vram_size = self.model_vram_sizes.get(model_tag, 0)
            candidates.append((model_tag, score, vram_size))
        candidates.sort(key=operator.itemgetter(1))

        to_evict = []
        freed_vram = 0

        for model_tag, score, vram_size in candidates:
            to_evict.append(model_tag)
            freed_vram += vram_size

            logger.debug(
                "Eviction candidate",
                model=model_tag,
                score=score,
                vram_gb=vram_size / (1024**3),
                total_freed_gb=freed_vram / (1024**3),
            )

            current_available = self.vram_monitor.get_available_vram()
            projected_available = current_available + freed_vram

            if projected_available >= required_vram:
                break

        logger.info(
            "Eviction plan",
            models_to_evict=len(to_evict),
            total_freed_gb=freed_vram / (1024**3),
            required_gb=required_vram / (1024**3),
        )

        return to_evict

    def evict_model(self, cache: OrderedDict, model_tag: str) -> int:
        """Evict a model and return freed VRAM."""
        if model_tag not in cache:
            return 0

        vram_size = self.model_vram_sizes.get(model_tag, 0)

        evicted_model = cache.pop(model_tag)

        try:
            model, tokenizer = evicted_model
            if hasattr(model, "cpu"):
                model.cpu()
            del model, tokenizer, evicted_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(
                "Error during model eviction cleanup", model=model_tag, error=str(e)
            )

        self.model_vram_sizes.pop(model_tag, None)

        score = self.usage_tracker.calculate_score(model_tag)
        logger.info(
            "Model evicted",
            model=model_tag,
            score=score,
            freed_vram_gb=vram_size / (1024**3),
            cache_size=len(cache),
        )

        return vram_size
