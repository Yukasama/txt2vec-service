# ruff: noqa: S101

"""Valid model loading and cache operation tests."""

import time

import pytest
from fastapi.testclient import TestClient

from vectorize.inference.cache.cache_factory import create_model_cache
from vectorize.inference.cache.model_cache import ModelCache

from .test_model_loading_common import (
    HTTP_NO_CONTENT,
    MINILM_MODEL_TAG,
    create_real_model_loader,
    ensure_minilm_model_available,
    make_embedding_request,
)


@pytest.mark.cache
@pytest.mark.integration
@pytest.mark.cache_valid
class TestCacheValidOperations:
    """Valid cache operations with real model."""

    @staticmethod
    def test_model_loads_into_cache_via_api(client: TestClient) -> None:
        """Test that model gets loaded and cached through API call."""
        ensure_minilm_model_available()

        clear_response = client.delete("/embeddings/cache/clear")
        assert clear_response.status_code == HTTP_NO_CONTENT

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert cache_data["cache_size"] == 0

        embedding_response = make_embedding_request(client, MINILM_MODEL_TAG)

        assert "data" in embedding_response
        assert len(embedding_response["data"]) > 0
        assert "embedding" in embedding_response["data"][0]

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert cache_data["cache_size"] == 1
        assert MINILM_MODEL_TAG in cache_data["cached_models"]

    @staticmethod
    def test_cache_hit_performance(client: TestClient) -> None:
        """Test that cached model requests are faster than initial load."""
        ensure_minilm_model_available()

        client.delete("/embeddings/cache/clear")

        start_time = time.time()
        make_embedding_request(client, MINILM_MODEL_TAG)
        first_request_time = time.time() - start_time

        start_time = time.time()
        make_embedding_request(client, MINILM_MODEL_TAG)
        second_request_time = time.time() - start_time

        assert second_request_time < first_request_time * 0.5

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert MINILM_MODEL_TAG in cache_data["cached_models"]

    @staticmethod
    def test_usage_tracking(client: TestClient) -> None:
        """Test that model usage is tracked correctly."""
        ensure_minilm_model_available()

        client.delete("/embeddings/cache/clear")

        request_count = 3
        for i in range(request_count):
            input_text = f"Usage tracking test sentence {i}."
            make_embedding_request(client, MINILM_MODEL_TAG, input_text)

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()

        assert MINILM_MODEL_TAG in cache_data["cached_models"]

        if (
            "usage_stats" in cache_data
            and MINILM_MODEL_TAG in cache_data["usage_stats"]
        ):
            model_stats = cache_data["usage_stats"][MINILM_MODEL_TAG]
            assert "count" in model_stats
            assert model_stats["count"] >= request_count

    @staticmethod
    def test_cache_persistence_across_clear_reload(client: TestClient) -> None:
        """Test that models can be loaded again after cache clear."""
        ensure_minilm_model_available()

        make_embedding_request(client, MINILM_MODEL_TAG)

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert MINILM_MODEL_TAG in cache_data["cached_models"]

        clear_response = client.delete("/embeddings/cache/clear")
        assert clear_response.status_code == HTTP_NO_CONTENT

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert cache_data["cache_size"] == 0

        make_embedding_request(client, MINILM_MODEL_TAG)

        cache_response = client.get("/embeddings/cache/status")
        cache_data = cache_response.json()
        assert cache_data["cache_size"] >= 1
        assert MINILM_MODEL_TAG in cache_data["cached_models"]


@pytest.mark.cache
@pytest.mark.integration
@pytest.mark.cache_factory
class TestCacheFactoryValidOperations:
    """Valid cache factory operations with real model loading."""

    @staticmethod
    def test_cache_factory_with_real_model(temp_cache_file: str) -> None:
        """Test that cache factory works with real model loading."""
        ensure_minilm_model_available()

        cache = create_model_cache(temp_cache_file)

        assert isinstance(cache, ModelCache)
        assert cache.eviction.max_models > 0

        real_model_loader = create_real_model_loader(MINILM_MODEL_TAG)

        model, tokenizer = cache.get(MINILM_MODEL_TAG, real_model_loader)

        assert hasattr(model, "forward")
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")

        info = cache.get_info()
        assert info["cache_size"] == 1
        assert MINILM_MODEL_TAG in info["cached_models"]

    @staticmethod
    def test_cache_performance_with_real_model(temp_cache_file: str) -> None:
        """Test cache performance with real model loading."""
        ensure_minilm_model_available()

        cache = create_model_cache(temp_cache_file)
        real_model_loader = create_real_model_loader(MINILM_MODEL_TAG)

        start_time = time.time()
        model1, _ = cache.get(MINILM_MODEL_TAG, real_model_loader)
        first_load_time = time.time() - start_time

        start_time = time.time()
        model2, _ = cache.get(MINILM_MODEL_TAG, real_model_loader)
        second_load_time = time.time() - start_time

        assert model1 is model2

        assert second_load_time < first_load_time * 0.1
