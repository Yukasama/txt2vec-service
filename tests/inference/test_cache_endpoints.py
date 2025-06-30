# ruff: noqa: S101

"""Tests for inference cache endpoints."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from .test_model_loading_common import (
    MINILM_MODEL_TAG,
    ensure_minilm_model_available,
    make_embedding_request,
)


@pytest.mark.inference
@pytest.mark.integration
class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    @staticmethod
    def test_cache_status_endpoint(client: TestClient) -> None:
        """Test cache status endpoint returns correct structure."""
        response = client.get("/embeddings/cache/status")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "cache_size" in data
        assert "cached_models" in data
        assert isinstance(data["cache_size"], int)
        assert isinstance(data["cached_models"], list)

    @staticmethod
    def test_cache_clear_endpoint(client: TestClient) -> None:
        """Test cache clearing functionality."""
        ensure_minilm_model_available()

        make_embedding_request(client, MINILM_MODEL_TAG)

        clear_response = client.delete("/embeddings/cache/clear")
        assert clear_response.status_code == status.HTTP_204_NO_CONTENT
        assert clear_response.headers.get("X-Cache-Status") == "cleared"
