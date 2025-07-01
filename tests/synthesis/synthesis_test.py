# ruff: noqa: S101

"""Tests for synthesis endpoints."""

import io
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.task.task_status import TaskStatus

_INVALID_DATASET_ID = "abc12345-6789-0123-4567-89abcdef0123"
_MALFORMED_UUID = "not-a-valid-uuid"
_SYNTHESIS_MEDIA = "/synthesis/media"
_SYNTHESIS_ENDPOINT = "/synthesis"

# Test dataset ID (from existing test data)
TEST_DATASET_ID = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"

# Test data paths
TEST_DATA_DIR = (
    Path(__file__).parent.parent.parent / "test_data" / "synthesis" / "valid"
)

# Test file names
TEST_PDF_FILENAME = "test_data.pdf"
TEST_JPG_FILENAME = "test_data.jpg"
TEST_PNG_FILENAME = "test_data.png"

# Test file paths
TEST_PDF_PATH = TEST_DATA_DIR / TEST_PDF_FILENAME
TEST_JPG_PATH = TEST_DATA_DIR / TEST_JPG_FILENAME
TEST_PNG_PATH = TEST_DATA_DIR / TEST_PNG_FILENAME

# Test constants
EXPECTED_FILE_COUNT_MULTIPLE = 3
LIMIT_PARAMETER_TEST = 5
LIMIT_PARAMETER_MAX = 100
HTTP_STATUS_CREATED = 201

# MIME types
MIME_TEXT_PLAIN = "text/plain"
MIME_APPLICATION_PDF = "application/pdf"
MIME_IMAGE_JPEG = "image/jpeg"
MIME_IMAGE_PNG = "image/png"
MIME_APPLICATION_DOCX = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# Valid task statuses
VALID_TASK_STATUSES = {
    TaskStatus.QUEUED,
    TaskStatus.RUNNING,
    TaskStatus.DONE,
    TaskStatus.FAILED,
}

RUNNING_TASK_STATUSES = {TaskStatus.QUEUED, TaskStatus.RUNNING}


@pytest.mark.asyncio
@pytest.mark.synthesis
class TestSynthesisTasks:
    """Tests for synthesis task endpoints."""

    @classmethod
    async def test_upload_media_with_invalid_dataset_id(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task with non-existent dataset ID."""
        response = client.post(
            _SYNTHESIS_MEDIA, data={"dataset_id": _INVALID_DATASET_ID}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @classmethod
    async def test_upload_media_with_malformed_uuid(cls, client: TestClient) -> None:
        """Test creating synthesis task with malformed UUID."""
        response = client.post(_SYNTHESIS_MEDIA, data={"dataset_id": _MALFORMED_UUID})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @classmethod
    async def test_upload_media_without_files_or_dataset(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task without providing files or dataset ID."""
        response = client.post(_SYNTHESIS_MEDIA)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert (
            "either files or existing dataset id must be provided"
            in response.json()["detail"].lower()
        )

    @classmethod
    async def test_upload_media_with_pdf_file(cls, client: TestClient) -> None:
        """Test creating synthesis task with real PDF file upload."""
        pdf_content = TEST_PDF_PATH.read_bytes()

        files = {
            "files": (
                TEST_PDF_FILENAME,
                io.BytesIO(pdf_content),
                MIME_APPLICATION_PDF,
            )
        }

        response = client.post(_SYNTHESIS_MEDIA, files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert "status_url" in response_data
        assert "message" in response_data
        assert "file_count" in response_data
        assert response_data["file_count"] == 1

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

        task_data = status_response.json()
        assert task_data["id"] == task_id
        assert task_data["task_status"] in RUNNING_TASK_STATUSES

    @classmethod
    async def test_upload_media_with_jpg_file(cls, client: TestClient) -> None:
        """Test creating synthesis task with real JPG file upload."""
        jpg_content = TEST_JPG_PATH.read_bytes()

        files = {
            "files": (
                TEST_JPG_FILENAME,
                io.BytesIO(jpg_content),
                MIME_IMAGE_JPEG,
            )
        }

        response = client.post(_SYNTHESIS_MEDIA, files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert "status_url" in response_data
        assert response_data["file_count"] == 1

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_upload_media_with_png_file(cls, client: TestClient) -> None:
        """Test creating synthesis task with real PNG file upload."""
        png_content = TEST_PNG_PATH.read_bytes()

        files = {
            "files": (
                TEST_PNG_FILENAME,
                io.BytesIO(png_content),
                MIME_IMAGE_PNG,
            )
        }

        response = client.post(_SYNTHESIS_MEDIA, files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert response_data["file_count"] == 1

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_upload_media_with_multiple_real_files(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task with multiple real test files."""
        pdf_content = TEST_PDF_PATH.read_bytes()
        jpg_content = TEST_JPG_PATH.read_bytes()
        png_content = TEST_PNG_PATH.read_bytes()

        files = [
            (
                "files",
                (TEST_PDF_FILENAME, io.BytesIO(pdf_content), MIME_APPLICATION_PDF),
            ),
            (
                "files",
                (TEST_JPG_FILENAME, io.BytesIO(jpg_content), MIME_IMAGE_JPEG),
            ),
            (
                "files",
                (TEST_PNG_FILENAME, io.BytesIO(png_content), MIME_IMAGE_PNG),
            ),
        ]

        response = client.post(_SYNTHESIS_MEDIA, files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert "file_count" in response_data
        assert response_data["file_count"] == EXPECTED_FILE_COUNT_MULTIPLE

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_get_synthesis_task_success(cls, client: TestClient) -> None:
        """Test successful retrieval of an existing synthesis task."""
        text_content = "Sample text for testing task retrieval"
        files = {
            "files": ("test.txt", io.BytesIO(text_content.encode()), MIME_TEXT_PLAIN)
        }

        create_response = client.post(_SYNTHESIS_MEDIA, files=files)
        assert create_response.status_code == status.HTTP_202_ACCEPTED

        task_id = create_response.json()["task_id"]

        response = client.get(f"/synthesis/tasks/{task_id}")
        assert response.status_code == status.HTTP_200_OK

        task_data = response.json()

        assert "id" in task_data
        assert "task_status" in task_data
        assert "created_at" in task_data
        assert "updated_at" in task_data
        assert task_data["id"] == task_id
        assert task_data["task_status"] in VALID_TASK_STATUSES

        assert task_data["created_at"] is not None
        assert task_data["updated_at"] is not None

    @classmethod
    async def test_synthesis_task_status_url_integration(
        cls, client: TestClient
    ) -> None:
        """Test that the status URL returned in upload response works correctly."""
        text_content = "Integration test content"
        files = {
            "files": (
                "integration_test.txt",
                io.BytesIO(text_content.encode()),
                MIME_TEXT_PLAIN,
            )
        }

        create_response = client.post(_SYNTHESIS_MEDIA, files=files)
        assert create_response.status_code == status.HTTP_202_ACCEPTED

        response_data = create_response.json()
        status_url = response_data["status_url"]

        task_id = response_data["task_id"]
        assert f"/synthesis/tasks/{task_id}" in status_url

        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_upload_media_with_large_file(cls, client: TestClient) -> None:
        """Test creating synthesis task with larger file content."""
        large_content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20

        files = {
            "files": (
                "large_test.txt",
                io.BytesIO(large_content.encode()),
                MIME_TEXT_PLAIN,
            )
        }

        response = client.post(_SYNTHESIS_MEDIA, files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert response_data["file_count"] == 1

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_get_synthesis_task_invalid_uuid(cls, client: TestClient) -> None:
        """Test retrieving synthesis task with invalid UUID."""
        invalid_uuid = "not-a-uuid"

        response = client.get(f"/synthesis/tasks/{invalid_uuid}")
        assert response.status_code in {
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
        }

    @classmethod
    async def test_get_nonexistent_synthesis_task(cls, client: TestClient) -> None:
        """Test retrieving non-existent synthesis task."""
        nonexistent_id = "12345678-1234-5678-1234-567812345678"

        response = client.get(f"/synthesis/tasks/{nonexistent_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "synthesis task not found" in response.json()["detail"].lower()

    @classmethod
    async def test_list_synthesis_tasks_endpoint_exists(
        cls, client: TestClient
    ) -> None:
        """Test that the list endpoint exists and returns proper format."""
        response = client.get(_SYNTHESIS_ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)

    @classmethod
    async def test_list_synthesis_tasks_with_created_task(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks after creating one."""
        text_content = "Test content for listing"
        files = {
            "files": (
                "list_test.txt",
                io.BytesIO(text_content.encode()),
                MIME_TEXT_PLAIN,
            )
        }

        create_response = client.post(_SYNTHESIS_MEDIA, files=files)
        assert create_response.status_code == status.HTTP_202_ACCEPTED

        created_task_id = create_response.json()["task_id"]

        # Now list tasks
        response = client.get(_SYNTHESIS_ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)

        task_ids = [task["id"] for task in tasks]
        assert created_task_id in task_ids

        for task in tasks:
            assert "id" in task
            assert "task_status" in task
            assert "created_at" in task
            assert "updated_at" in task

    @classmethod
    async def test_list_synthesis_tasks_with_limit_parameter(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks with limit parameter."""
        response = client.get(f"{_SYNTHESIS_ENDPOINT}?limit={LIMIT_PARAMETER_TEST}")
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= LIMIT_PARAMETER_TEST

    @classmethod
    async def test_list_synthesis_tasks_with_invalid_limit(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks with invalid limit parameter."""
        response = client.get(f"{_SYNTHESIS_ENDPOINT}?limit=-1")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @classmethod
    async def test_list_synthesis_tasks_with_max_limit(cls, client: TestClient) -> None:
        """Test listing synthesis tasks with maximum allowed limit."""
        response = client.get(f"{_SYNTHESIS_ENDPOINT}?limit={LIMIT_PARAMETER_MAX}")
        assert response.status_code == status.HTTP_200_OK

        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) <= LIMIT_PARAMETER_MAX

    @classmethod
    async def test_list_synthesis_tasks_with_excessive_limit(
        cls, client: TestClient
    ) -> None:
        """Test listing synthesis tasks with limit exceeding maximum."""
        excessive_limit = LIMIT_PARAMETER_MAX + 1
        response = client.get(f"{_SYNTHESIS_ENDPOINT}?limit={excessive_limit}")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @classmethod
    async def test_synthesis_router_registration(cls, client: TestClient) -> None:
        """Test that synthesis endpoints are properly registered."""
        response = client.post(_SYNTHESIS_MEDIA)
        assert response.status_code != status.HTTP_404_NOT_FOUND

        response = client.get(_SYNTHESIS_ENDPOINT)
        assert response.status_code == status.HTTP_200_OK

        response = client.get(f"/synthesis/tasks/{_INVALID_DATASET_ID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @classmethod
    async def test_upload_media_with_existing_dataset_id(
        cls, client: TestClient
    ) -> None:
        """Test creating synthesis task with existing dataset ID."""
        # Use existing test dataset ID
        response = client.post(_SYNTHESIS_MEDIA, data={"dataset_id": TEST_DATASET_ID})

        # Expect successful synthesis task creation
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()

        assert "task_id" in response_data
        assert "status_url" in response_data
        assert "message" in response_data
        assert "dataset_id" in response_data
        assert response_data["dataset_id"] == TEST_DATASET_ID

        task_id = response_data["task_id"]
        status_response = client.get(f"/synthesis/tasks/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK

        task_data = status_response.json()
        assert task_data["id"] == task_id
        assert task_data["task_status"] in RUNNING_TASK_STATUSES
