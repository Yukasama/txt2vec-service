"""Model delete utilities."""

import shutil
from pathlib import Path

from loguru import logger

__all__ = ["remove_model_from_memory"]


async def remove_model_from_memory(model_tag: str) -> None:  # noqa: RUF029 NOSONAR
    """Remove a AI model from memory.

    Args:
        model_tag (str): The model tag
    """
    base_path = Path("/app/data/models")
    model_folder = base_path / model_tag

    if model_folder.exists() and model_folder.is_dir():
        shutil.rmtree(model_folder)
        logger.info("Deleted model folder from disk: {}", model_folder)
    else:
        logger.info("Model folder not found on disk: {}", model_folder)
