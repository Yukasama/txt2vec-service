"""VRAM-aware model cache with dynamic capacity based on GPU memory."""

import operator
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import torch
from loguru import logger
from transformers import AutoTokenizer

from .usage_tracker import UsageTracker
from .vram_eviction import VRAMEviction
from .vram_monitor import VRAMMonitor

__all__ = ["VRAMModelCache"]


def _timestamp_to_iso(timestamp: int) -> str:
    """Convert Unix timestamp to ISO format string."""
    if timestamp == 0:
        return "Never"
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


class ModelLoader(Protocol):
    """Protocol for model loader functions."""

    def __call__(
        self, model_tag: str
    ) -> tuple[torch.nn.Module, AutoTokenizer | None]: ...


class VRAMModelCache:
    """Model cache with VRAM awareness - dynamic number of models."""

    def __init__(
        self,
        safety_margin_gb: float = 1.0,
        cache_file: str = "data/cache/model_usage_stats.json",
    ) -> None:
        """Initialize VRAM-aware model cache.

        Args:
            safety_margin_gb: Safety margin in GB to keep free
            cache_file: Path to file for storing usage statistics
        """
        self.cache: OrderedDict[str, tuple[torch.nn.Module, AutoTokenizer | None]] = (
            OrderedDict()
        )
        self.lock = threading.Lock()

        self.usage_tracker = UsageTracker(Path(cache_file))
        self.vram_monitor = VRAMMonitor(safety_margin_gb)
        self.eviction = VRAMEviction(self.usage_tracker, self.vram_monitor)

        logger.info(
            "VRAM-aware model cache initialized",
            safety_margin_gb=safety_margin_gb,
            device=self.vram_monitor.device,
        )

    def get(
        self, model_tag: str, loader_func: ModelLoader
    ) -> tuple[torch.nn.Module, AutoTokenizer | None]:
        """Load model with VRAM awareness."""
        with self.lock:
            self.usage_tracker.track_access(model_tag)

            if model_tag in self.cache:
                self.cache.move_to_end(model_tag)
                logger.debug(
                    "VRAM cache hit", model=model_tag, cache_size=len(self.cache)
                )
                return self.cache[model_tag]

        logger.debug("Loading model for VRAM cache", model=model_tag)
        model_data = loader_func(model_tag)
        model = model_data[0]

        estimated_vram = self.vram_monitor.estimate_model_vram(model)
        logger.debug(
            "Model VRAM estimated",
            model=model_tag,
            estimated_gb=estimated_vram / (1024**3),
        )

        with self.lock:
            models_to_evict = self.eviction.find_models_to_evict(
                self.cache, estimated_vram
            )

            total_freed_vram = 0
            for evict_model_tag in models_to_evict:
                freed_vram = self.eviction.evict_model(self.cache, evict_model_tag)
                total_freed_vram += freed_vram

            if not self.vram_monitor.can_fit_model(estimated_vram):
                vram_info = self.vram_monitor.get_vram_info()
                logger.error(
                    "Cannot fit model even after eviction",
                    model=model_tag,
                    estimated_gb=estimated_vram / (1024**3),
                    available_gb=vram_info.get("available_gb", 0),
                    evicted_models=len(models_to_evict),
                    freed_gb=total_freed_vram / (1024**3),
                )

            self.cache[model_tag] = model_data
            self.eviction.track_model_vram(model_tag, model)
            self.usage_tracker.save_stats()

            vram_info = self.vram_monitor.get_vram_info()
            logger.info(
                "Model cached with VRAM awareness",
                model=model_tag,
                cache_size=len(self.cache),
                model_vram_gb=estimated_vram / (1024**3),
                total_vram_used_gb=vram_info.get("used_gb", 0),
                vram_available_gb=vram_info.get("available_gb", 0),
            )

        return model_data

    def get_info(self) -> dict:
        """Get cache information with VRAM details."""
        with self.lock:
            vram_info = self.vram_monitor.get_vram_info()

            cached_models_sorted = []
            model_vram_usage = {}
            total_cached_vram = 0

            if self.cache:
                model_scores = []
                for model_tag in self.cache:
                    score = self.usage_tracker.calculate_score(model_tag)
                    vram_size = self.eviction.model_vram_sizes.get(model_tag, 0)
                    vram_gb = vram_size / (1024**3)

                    model_scores.append((model_tag, score))
                    model_vram_usage[model_tag] = vram_gb
                    total_cached_vram += vram_gb
                model_scores.sort(key=operator.itemgetter(1), reverse=True)
                cached_models_sorted = [model_tag for model_tag, _ in model_scores]

            readable_stats = {}
            for model_tag, stats in self.usage_tracker.get_stats().items():
                readable_stats[model_tag] = {
                    "count": stats["count"],
                    "last_access": _timestamp_to_iso(stats["last_access"]),
                }

            return {
                "cached_models": cached_models_sorted,
                "cache_size": len(self.cache),
                "max_size": "dynamic (VRAM-limited)",
                "vram_info": vram_info,
                "model_vram_usage_gb": model_vram_usage,
                "total_cached_vram_gb": total_cached_vram,
                "usage_stats": readable_stats,
            }

    def clear(self) -> None:
        """Clear cache and free all VRAM."""
        with self.lock:
            for model_tag in self.cache:
                self.eviction.evict_model(self.cache, model_tag)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("VRAM cache cleared completely")

    def get_vram_utilization(self) -> float:
        """Return VRAM utilization in percent."""
        if not self.vram_monitor.is_cuda:
            return 0.0

        total = self.vram_monitor.get_total_vram()
        used = self.vram_monitor.get_used_vram()
        return (used / total) * 100 if total > 0 else 0.0
