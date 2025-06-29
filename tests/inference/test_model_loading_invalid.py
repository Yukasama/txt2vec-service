# ruff: noqa: S101

"""Invalid model loading and error handling tests."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.inference.cache.cache_factory import create_model_cache
from vectorize.inference.cache.model_cache import ModelLoader

from .test_model_loading_common import (
    HTTP_NOT_FOUND,
    MINILM_MODEL_TAG,
    NONEXISTENT_MODEL,
    ensure_minilm_model_available,
    make_embedding_request,
)

HTTP_BAD_REQUEST = status.HTTP_400_BAD_REQUEST
HTTP_UNPROCESSABLE_ENTITY = status.HTTP_422_UNPROCESSABLE_ENTITY

VALIDATION_ERROR_CODES = {HTTP_BAD_REQUEST, HTTP_UNPROCESSABLE_ENTITY}
MODEL_ERROR_CODES = {HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_UNPROCESSABLE_ENTITY}


@pytest.mark.cache
@pytest.mark.integration
@pytest.mark.cache_invalid
class TestCacheInvalidOperations:
    """Invalid operations and error handling tests."""

    @staticmethod
    def test_nonexistent_model_handling(client: TestClient) -> None:
        """Test that nonexistent models don't affect cache state."""
        ensure_minilm_model_available()

        client.delete("/embeddings/cache/clear")
        make_embedding_request(client, MINILM_MODEL_TAG)

        cache_response = client.get("/embeddings/cache/status")
        initial_data = cache_response.json()
        initial_size = initial_data["cache_size"]

        make_embedding_request(
            client, NONEXISTENT_MODEL, expected_status=HTTP_NOT_FOUND
        )

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert cache_data["cache_size"] == initial_size
        assert MINILM_MODEL_TAG in cache_data["cached_models"]

    @staticmethod
    def test_empty_model_name(client: TestClient) -> None:
        """Test handling of empty model name."""
        payload = {"model": "", "input": "Test text"}
        response = client.post("/embeddings", json=payload)
        assert response.status_code in VALIDATION_ERROR_CODES

    @staticmethod
    def test_invalid_model_characters(client: TestClient) -> None:
        """Test handling of model names with invalid characters."""
        invalid_models = [
            "model/with/slashes",
            "model with spaces",
            "model@with@symbols",
            "../path/traversal",
        ]

        for invalid_model in invalid_models:
            payload = {"model": invalid_model, "input": "Test text"}
            response = client.post("/embeddings", json=payload)
            assert response.status_code in MODEL_ERROR_CODES

    @staticmethod
    def test_missing_model_parameter(client: TestClient) -> None:
        """Test handling of missing model parameter."""
        payload = {"input": "Test text"}
        response = client.post("/embeddings", json=payload)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY

    @staticmethod
    def test_missing_input_parameter(client: TestClient) -> None:
        """Test handling of missing input parameter."""
        payload = {"model": MINILM_MODEL_TAG}
        response = client.post("/embeddings", json=payload)
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY


@pytest.mark.cache
@pytest.mark.integration
@pytest.mark.cache_factory
class TestCacheFactoryInvalidOperations:
    """Invalid cache factory operations and error handling."""

    @staticmethod
    def test_cache_factory_with_nonexistent_model(temp_cache_file: str) -> None:
        """Test cache factory behavior with nonexistent model."""
        cache = create_model_cache(temp_cache_file)

        class FailingModelLoader:
            """Model loader that always fails."""

            def __call__(self, model_tag: str) -> tuple:
                raise FileNotFoundError(f"Model {model_tag} not found")

        failing_loader: ModelLoader = FailingModelLoader()

        with pytest.raises(FileNotFoundError):
            cache.get(NONEXISTENT_MODEL, failing_loader)

        info = cache.get_info()
        assert info["cache_size"] == 0

    @staticmethod
    def test_cache_factory_with_loader_exception(temp_cache_file: str) -> None:
        """Test cache factory behavior when loader raises unexpected exception."""
        cache = create_model_cache(temp_cache_file)

        class ExceptionModelLoader:
            """Model loader that raises RuntimeError."""

            def __call__(self, model_tag: str) -> tuple:
                raise RuntimeError(
                    f"Unexpected error during loading of model {model_tag}"
                )

        exception_loader: ModelLoader = ExceptionModelLoader()

        with pytest.raises(RuntimeError, match="Unexpected error"):
            cache.get(MINILM_MODEL_TAG, exception_loader)

        info = cache.get_info()
        assert info["cache_size"] == 0
