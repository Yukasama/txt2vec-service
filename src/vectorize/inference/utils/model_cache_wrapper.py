"""Wrapper module for cached model loading with singleton cache instance."""

import torch
from transformers import AutoTokenizer

from ..cache.cache_factory import create_model_cache
from ..cache.model_cache import ModelCache
from ..cache.vram_model_cache import VRAMModelCache
from .model_loader import load_model as _original_load_model

__all__ = ["clear_model_cache", "get_cache", "get_cache_status", "load_model_cached"]


_cache = create_model_cache()


def get_cache() -> ModelCache | VRAMModelCache:
    """Get the global cache instance.

    Returns:
        The global model cache instance
    """
    return _cache


def load_model_cached(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
    """Cached version of the load_model function."""
    return _cache.get(model_tag, _original_load_model)


def get_cache_status() -> dict:
    """Get cache status for monitoring."""
    return _cache.get_info()


def clear_model_cache() -> None:
    """Clear cache (admin function)."""
    _cache.clear()
