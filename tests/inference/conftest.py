"""Pytest configuration and fixtures for inference tests."""

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

import aiofiles
import pytest


@pytest.fixture
async def temp_cache_file() -> AsyncGenerator[str]:
    """Create temporary cache file."""
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"{uuid4()}.json"
    cache_file_path = temp_dir / temp_filename

    async with aiofiles.open(cache_file_path, "w", encoding="utf-8") as f:
        await f.write("{}")

    yield str(cache_file_path)

    if cache_file_path.exists():
        cache_file_path.unlink()
