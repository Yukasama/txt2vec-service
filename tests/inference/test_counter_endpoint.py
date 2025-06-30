# ruff: noqa: S101

"""Tests for inference counter endpoint."""

from datetime import datetime

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from .test_model_loading_common import (
    MINILM_MODEL_TAG,
    NONEXISTENT_MODEL,
    ensure_minilm_model_available,
    make_embedding_request,
)

# Constants
DATE_FORMAT_LENGTH = 10


@pytest.mark.inference
@pytest.mark.integration
class TestCounterEndpoint:
    """Tests for model inference counter endpoint."""

    @staticmethod
    def test_counter_endpoint_structure(client: TestClient) -> None:
        """Test counter endpoint returns correct structure."""
        ensure_minilm_model_available()

        response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert isinstance(data, dict)

        for date_key, count in data.items():
            assert isinstance(date_key, str)
            assert isinstance(count, int)
            assert count >= 0
            assert len(date_key) == DATE_FORMAT_LENGTH
            assert date_key[4] == "-"
            assert date_key[7] == "-"

    @staticmethod
    def test_counter_increment_on_inference(client: TestClient) -> None:
        """Test that counter increments when inference is performed."""
        ensure_minilm_model_available()

        counter_response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        assert counter_response.status_code == status.HTTP_200_OK

        initial_data = counter_response.json()
        today_key = next(iter(initial_data))
        initial_count = initial_data[today_key]

        make_embedding_request(client, MINILM_MODEL_TAG)

        counter_response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        assert counter_response.status_code == status.HTTP_200_OK

        updated_data = counter_response.json()
        assert updated_data[today_key] == initial_count + 1

    @staticmethod
    def test_counter_multiple_inferences(client: TestClient) -> None:
        """Test counter with multiple inferences."""
        ensure_minilm_model_available()

        counter_response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        initial_data = counter_response.json()
        today_key = next(iter(initial_data))
        initial_count = initial_data[today_key]

        inference_count = 5
        for _ in range(inference_count):
            make_embedding_request(client, MINILM_MODEL_TAG)

        counter_response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        updated_data = counter_response.json()
        assert updated_data[today_key] == initial_count + inference_count

    @staticmethod
    def test_counter_nonexistent_model(client: TestClient) -> None:
        """Test counter endpoint with nonexistent model."""
        response = client.get(f"/embeddings/counter/{NONEXISTENT_MODEL}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @staticmethod
    def test_counter_empty_model_tag(client: TestClient) -> None:
        """Test counter endpoint with empty model tag."""
        response = client.get("/embeddings/counter/")

        assert response.status_code in {
            status.HTTP_404_NOT_FOUND,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        }

    @staticmethod
    def test_counter_special_characters_model_tag(client: TestClient) -> None:
        """Test counter endpoint with special characters in model tag."""
        special_model_tags = [
            "model/with/slashes",
            "model-with-dashes",
            "model_with_underscores",
            "model.with.dots",
            "model%20with%20spaces",
            "model@with#special$chars",
        ]

        for model_tag in special_model_tags:
            response = client.get(f"/embeddings/counter/{model_tag}")
            assert response.status_code == status.HTTP_404_NOT_FOUND

    @staticmethod
    def test_counter_very_long_model_tag(client: TestClient) -> None:
        """Test counter endpoint with very long model tag."""
        long_model_tag = "A" * 1000
        response = client.get(f"/embeddings/counter/{long_model_tag}")

        assert response.status_code in {
            status.HTTP_404_NOT_FOUND,
            status.HTTP_400_BAD_REQUEST,
        }

    @staticmethod
    def test_counter_sql_injection_attempts(client: TestClient) -> None:
        """Test counter endpoint against SQL injection."""
        malicious_tags = [
            "'; DROP TABLE models; --",
            "model' OR '1'='1",
            "model'; UPDATE models SET name='hacked' WHERE id=1; --",
            "../../../etc/passwd",
        ]

        for malicious_tag in malicious_tags:
            response = client.get(f"/embeddings/counter/{malicious_tag}")
            assert response.status_code == status.HTTP_404_NOT_FOUND

    @staticmethod
    def test_counter_date_range_coverage(client: TestClient) -> None:
        """Test that counter returns reasonable date range."""
        ensure_minilm_model_available()

        make_embedding_request(client, MINILM_MODEL_TAG)

        response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        assert len(data) >= 1

        for date_str in data:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")  # noqa: DTZ007
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")

    @staticmethod
    def test_counter_zero_values_for_unused_dates(client: TestClient) -> None:
        """Test that counter includes zero values for dates without inferences."""
        ensure_minilm_model_available()

        response = client.get(f"/embeddings/counter/{MINILM_MODEL_TAG}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()

        for count in data.values():
            assert isinstance(count, int)
            assert count >= 0
