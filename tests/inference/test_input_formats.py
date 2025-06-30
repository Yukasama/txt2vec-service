# ruff: noqa: S101

"""Tests for different input formats for the embeddings endpoint."""

import base64

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from .test_model_loading_common import MINILM_MODEL_TAG, ensure_minilm_model_available


@pytest.mark.inference
@pytest.mark.integration
class TestInputFormats:
    """Tests for various input formats."""

    @staticmethod
    def test_string_input(client: TestClient) -> None:
        """Test simple string input."""
        ensure_minilm_model_available()

        payload = {"model": MINILM_MODEL_TAG, "input": "Test sentence."}
        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]

    @staticmethod
    def test_list_of_strings_input(client: TestClient) -> None:
        """Test list of strings input."""
        ensure_minilm_model_available()

        payload = {
            "model": MINILM_MODEL_TAG,
            "input": ["First sentence.", "Second sentence."]
        }
        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        expected_count = 2
        assert len(data["data"]) == expected_count
        for i, item in enumerate(data["data"]):
            assert item["index"] == i
            assert "embedding" in item

    @staticmethod
    def test_token_array_input(client: TestClient) -> None:
        """Test token array input."""
        ensure_minilm_model_available()

        payload = {
            "model": MINILM_MODEL_TAG,
            "input": [101, 2023, 2003, 1037, 3231, 1012, 102]  # Simple token sequence
        }
        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        # Token arrays are interpreted as individual tokens,
        # creating one embedding per token
        expected_tokens = 7
        assert len(data["data"]) == expected_tokens  # 7 tokens = 7 embeddings
        assert all("embedding" in item for item in data["data"])

    @staticmethod
    def test_dimensions_parameter(client: TestClient) -> None:
        """Test dimensions parameter."""
        ensure_minilm_model_available()

        expected_dimensions = 10
        payload = {
            "model": MINILM_MODEL_TAG,
            "input": "Test sentence.",
            "dimensions": expected_dimensions
        }
        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert len(embedding) == expected_dimensions

    @staticmethod
    def test_base64_encoding_format(client: TestClient) -> None:
        """Test base64 encoding format."""
        ensure_minilm_model_available()

        payload = {
            "model": MINILM_MODEL_TAG,
            "input": "Test sentence.",
            "encoding_format": "base64"
        }
        response = client.post("/embeddings", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, str)  # Should be base64 string
        # Verify it's valid base64
        try:
            base64.b64decode(embedding)
        except Exception:
            pytest.fail("Invalid base64 encoding")
