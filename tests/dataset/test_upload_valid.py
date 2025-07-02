# ruff: noqa: S101

"""Tests for valid datasets."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from tests.dataset.utils import build_files

_TRAINING_FOLDER = "test_data"
_VALID_FOLDER = "valid"
_MALICIOUS_FOLDER = "malicious"

_DEFAULT = "default"
_CUSTOM_FORMAT = "custom_fields.csv"
_INFER_FORMAT = "infer_fields.csv"

_NULL_BYTE_INJECTION = "%00nullbyte.csv"
_COMMAND_INJECTION = "; rm -rf %2F.csv"


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_valid
class TestValidDatasets:
    """Tests for valid dataset uploads."""

    _base_dir = Path(__file__).parent.parent.parent / _TRAINING_FOLDER / "datasets"
    valid_dir = _base_dir / _VALID_FOLDER
    malicious_dir = _base_dir / _MALICIOUS_FOLDER

    @staticmethod
    async def _upload_and_verify(
        client: TestClient,
        file_path: Path,
        extra_data: dict[str, Any] | None = None,
    ) -> UUID:
        """Upload a file and verify the dataset is created."""
        response = client.post(
            "/datasets", files=build_files(file_path), data=extra_data or {}
        )
        assert response.status_code == status.HTTP_201_CREATED

        dataset_id = response.headers["Location"].split("/")[-1]
        assert dataset_id, "Location header must contain dataset id"

        response = client.get(f"/datasets/{dataset_id}")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == dataset_id

        return UUID(dataset_id)

    @pytest.mark.parametrize("ext", ["csv", "json", "jsonl", "xml", "xlsx"])
    async def test_formats_upload(self, ext: str, client: TestClient) -> None:
        """Uploading single-file datasets in multiple formats succeeds."""
        file_path = self.valid_dir / f"{_DEFAULT}.{ext}"
        await self._upload_and_verify(client, file_path)

    async def test_big_file_upload(self, client: TestClient) -> None:
        """Uploading a large JSONL file."""
        file_path = self.valid_dir / "default-big.jsonl"
        await self._upload_and_verify(client, file_path)

    @pytest.mark.parametrize(
        "file_name,file_length",
        [("default.zip", 5), ("big.zip", 400), ("very_big.zip", 10)],
    )
    async def test_zip_upload(
        self, file_name: str, file_length: int, client: TestClient
    ) -> None:
        """Uploading a ZIP archive succeeds and returns 201."""
        file_path = self.valid_dir / file_name

        response = client.get("/datasets?limit=500")
        assert response.status_code == status.HTTP_200_OK
        datasets_length = len(response.json()["items"])

        response = client.post("/datasets?limit=500", files=build_files(file_path))
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["successful_uploads"] == file_length

        response = client.get("/datasets?limit=500")
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["items"]) == datasets_length + file_length

    async def test_zip_partial_invalid_upload(
        self, client: TestClient
    ) -> None:  # NOSONAR #noqa
        """Uploading a ZIP archive succeeds and returns 201."""
        file_path = self.valid_dir / "partial.zip"

        valid_files = 4
        response = client.post("/datasets", files=build_files(file_path))
        assert response.status_code == status.HTTP_201_CREATED
        assert len(response.json()["failed"]) == valid_files
        assert response.json()["successful_uploads"] == valid_files

    async def test_custom_fields(self, client: TestClient) -> None:
        """Uploading a CSV with custom field mapping succeeds."""
        file_path = self.valid_dir / _CUSTOM_FORMAT

        column_mapping = {
            "question": "anchor",
            "positive": "answer",
            "negative": "no_context",
        }

        await self._upload_and_verify(
            client, file_path, {"options": json.dumps(column_mapping)}
        )

    async def test_infer_fields(self, client: TestClient) -> None:
        """Uploading a CSV where the API infers the fields succeeds."""
        file_path = self.valid_dir / _INFER_FORMAT
        await self._upload_and_verify(client, file_path)

    @pytest.mark.parametrize("file_name", [_NULL_BYTE_INJECTION, _COMMAND_INJECTION])
    async def test_malicious_files(self, client: TestClient, file_name: str) -> None:
        """Test that URL-encoded files with suspicious names are handled safely."""
        file_path = self.malicious_dir / file_name
        await self._upload_and_verify(client, file_path)
