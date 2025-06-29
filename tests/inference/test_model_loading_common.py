# ruff: noqa: S101

"""Common utilities and fixtures for model loading tests."""

import logging
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi import status
from fastapi.testclient import TestClient
from transformers import AutoModel, AutoTokenizer

from vectorize.config import settings
from vectorize.inference.cache.model_cache import ModelLoader

MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
MINILM_MODEL_SOURCE = Path(
    "test_data/training/models--sentence-transformers--all-MiniLM-L6-v2"
)
NONEXISTENT_MODEL = "definitely_not_a_model"

HTTP_OK = status.HTTP_200_OK
HTTP_NO_CONTENT = status.HTTP_204_NO_CONTENT
HTTP_NOT_FOUND = status.HTTP_404_NOT_FOUND

logger = logging.getLogger(__name__)


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for inference tests."""
    src = MINILM_MODEL_SOURCE
    dst = settings.model_inference_dir / MINILM_MODEL_TAG
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def make_embedding_request(
    client: TestClient,
    model: str,
    input_text: str = "Test sentence for embedding generation.",
    expected_status: int = HTTP_OK,
) -> dict[str, Any]:
    """Make embedding request and verify response."""
    payload = {"model": model, "input": input_text}
    response = client.post("/embeddings", json=payload)
    assert response.status_code == expected_status

    if expected_status == HTTP_OK:
        return response.json()
    return {}


class RealModelLoader:
    """Real model loader implementation for tests."""

    def __init__(self, model_tag: str) -> None:
        """Initialize the model loader with a specific model tag.

        Args:
            model_tag: The tag of the model to be loaded.
        """
        self.model_tag = model_tag

    def __call__(self, model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
        """Load model and tokenizer."""
        if model_tag == self.model_tag:
            model_path = settings.model_inference_dir / model_tag
            model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True
            )
            return model, tokenizer
        raise FileNotFoundError(f"Model {model_tag} not found")


def create_real_model_loader(model_tag: str) -> ModelLoader:
    """Create a model loader function for the given model tag."""
    return RealModelLoader(model_tag)


@pytest.fixture
def temp_cache_file() -> Generator[str]:
    """Create temporary cache file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        cache_file = f.name
    yield cache_file
    if Path(cache_file).exists():
        Path(cache_file).unlink()
