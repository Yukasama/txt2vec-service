# ruff: noqa: S101

"""Tests for inference endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_MODEL_NAME = "pytorch_model"
_BAD_MODEL_NAME = "nonexistent_model"


@pytest.mark.asyncio
@pytest.mark.inference
class TestInferenceCounter:
    """Tests for the embeddings inference counter."""

    @classmethod
    async def test_basic_counter_increment(cls, client: TestClient) -> None:
        """Test basic counter increment with successful embedding requests."""
        payload = {"model": _MODEL_NAME, "input": "This is a test sentence."}

        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        current_count = counter_response.json()[first_key]

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count + 1

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        counter_response = client.get(f"/embeddings/counter/{_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_200_OK
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count + 2

    @classmethod
    async def test_nonexistent_model_counter(cls, client: TestClient) -> None:
        """Test counter behavior with non-existent model requests."""
        payload = {"model": _BAD_MODEL_NAME, "input": "This is a test sentence."}

        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        current_count = counter_response.json()[first_key]

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND

        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count

        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND

        counter_response = client.get(f"/embeddings/counter/{_BAD_MODEL_NAME}")
        assert counter_response.status_code == status.HTTP_404_NOT_FOUND
        first_key = next(iter(counter_response.json()))
        assert counter_response.json()[first_key] == current_count
