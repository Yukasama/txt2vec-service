# ruff: noqa: S101

"""Tests for invalid datasets."""

from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.config.errors import ErrorCode

from .utils import build_files

_TRAINING_FOLDER = "test_data"
_INVALID_FOLDER = "invalid"
_MALICIOUS_FOLDER = "malicious"

_INVALID_FORMAT = "invalid_format"
_EMPTY_FILE = "empty.csv"
_UNSUPPORTED = "unsupported.txt"

_NO_FILE_NAME = ".csv"
_FORMULA_INJECTION = "formula_injection.csv"
_FIELD_SEPARATOR = "field_separator.csv"


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_invalid
class TestInvalidDatasets:
    """Tests for invalid datasets."""

    _base_dir = Path(__file__).parent.parent.parent / _TRAINING_FOLDER / "datasets"
    invalid_dir = _base_dir / _INVALID_FOLDER
    malicious_dir = _base_dir / _MALICIOUS_FOLDER

    @staticmethod
    async def _upload_and_verify(
        client: TestClient,
        file_path: Path,
        expected_status: int,
        expected_code: ErrorCode,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """Upload a file and verify the dataset is created."""
        response = client.post(
            "/datasets", files=build_files(file_path), data=extra_data or {}
        )

        assert response.status_code == expected_status
        assert response.json()["code"] == expected_code

    @pytest.mark.parametrize("ext", ["csv", "json", "jsonl", "xml", "xlsx"])
    async def test_invalid_format(self, client: TestClient, ext: str) -> None:
        """Test uploading an invalid file format."""
        test_file_path = self.invalid_dir / f"{_INVALID_FORMAT}.{ext}"

        await self._upload_and_verify(
            client,
            test_file_path,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            expected_code=ErrorCode.INVALID_CSV_FORMAT,
        )

    async def test_zip_all_invalid_upload(self, client: TestClient) -> None:  # NOSONAR
        """Uploading a ZIP archive succeeds and returns 201."""
        file_path = self.invalid_dir / "invalid.zip"

        invalid_files = 7
        response = client.post("/datasets", files=build_files(file_path))
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert len(response.json()["failed"]) == invalid_files
        assert response.json()["successful_uploads"] == 0

    async def test_empty(self, client: TestClient) -> None:
        """Test uploading an empty file."""
        test_file_path = self.invalid_dir / _EMPTY_FILE

        await self._upload_and_verify(
            client,
            test_file_path,
            expected_status=status.HTTP_400_BAD_REQUEST,
            expected_code=ErrorCode.EMPTY_FILE,
        )

    async def test_unsupported_format(self, client: TestClient) -> None:
        """Test uploading an unsupported format."""
        test_file_path = self.invalid_dir / _UNSUPPORTED

        await self._upload_and_verify(
            client,
            test_file_path,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            expected_code=ErrorCode.UNSUPPORTED_FORMAT,
        )

    async def test_no_file_name(self, client: TestClient) -> None:
        """Test uploading a file with no name."""
        test_file_path = self.invalid_dir / _NO_FILE_NAME

        await self._upload_and_verify(
            client,
            test_file_path,
            expected_status=status.HTTP_400_BAD_REQUEST,
            expected_code=ErrorCode.INVALID_FILE,
        )

    @pytest.mark.parametrize("file_name", [_FORMULA_INJECTION, _FIELD_SEPARATOR])
    async def test_malicious_files(self, client: TestClient, file_name: str) -> None:
        """Test uploading a file with no name."""
        test_file_path = self.malicious_dir / file_name

        await self._upload_and_verify(
            client,
            test_file_path,
            expected_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            expected_code=ErrorCode.INVALID_CSV_FORMAT,
        )
