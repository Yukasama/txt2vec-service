"""Module for tracking model usage statistics for cache management."""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = ["UsageTracker"]


class UsageTracker:
    """Manages usage statistics for model caching."""

    def __init__(self, cache_file: Path) -> None:
        """Initialize the usage tracker.

        Args:
            cache_file: Path to the file for storing usage statistics
        """
        self.cache_file = cache_file
        self.stats = defaultdict(
            lambda: {"count": 0, "last_reset": 0, "last_access": 0}
        )
        self._load_stats()

    def track_access(self, model_tag: str) -> None:
        """Register a model access."""
        current_time = time.time()
        stats = self.stats[model_tag]

        if current_time - stats["last_reset"] > (30 * 24 * 3600):
            stats["count"] = 0
            stats["last_reset"] = int(current_time)

        stats["count"] += 1
        stats["last_access"] = int(current_time)

    def calculate_score(self, model_tag: str) -> float:
        """Calculate usage score for eviction decision."""
        current_time = time.time()
        stats = self.stats[model_tag]

        if current_time - stats["last_reset"] > (30 * 24 * 3600):
            return 0.0

        return float(stats["count"])

    def get_stats(self) -> dict[str, Any]:
        """Return all usage statistics."""
        return dict(self.stats)

    def save_stats(self) -> None:
        """Save statistics to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with Path.open(self.cache_file, "w") as f:
                json.dump(dict(self.stats), f, indent=2)
            logger.debug("Usage stats saved", models=len(self.stats))
        except Exception as e:
            logger.warning("Failed to save usage stats", error=str(e))

    def _load_stats(self) -> None:
        """Load statistics from disk."""
        if not self.cache_file.exists():
            return

        try:
            with Path.open(self.cache_file) as f:
                data = json.load(f)
                for model_tag, stats in data.items():
                    self.stats[model_tag] = stats
            logger.debug("Usage stats loaded", models=len(data))
        except Exception as e:
            logger.warning("Failed to load usage stats", error=str(e))
