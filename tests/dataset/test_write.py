# ruff: noqa: S101

"""Tests for dataset PUT (update) endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_VALID_ID = "8b8c7f3e-4d2a-4b5c-9f1e-0a6f3e4d2a5b"
_INVALID_ID = "5d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_NON_EXISTENT_ID = "12345678-1234-5678-1234-567812345678"


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_write
class TestUpdateDatasets:
    """Tests for the PUT /datasets/{dataset_id} endpoint."""

    @classmethod
    async def test_successful_update(cls, client: TestClient) -> None:
        """Test successfully updating a dataset with valid data and matching ETag."""
        update_data = {"name": "Updated Dataset Name"}
        response = client.put(
            f"/datasets/{_VALID_ID}", json=update_data, headers={"If-Match": '"0"'}
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert "Location" in response.headers
        assert response.headers["Location"].endswith(str(_VALID_ID))

        assert "ETag" in response.headers
        assert response.headers["ETag"] == '"1"'

        get_response = client.get(f"/datasets/{_VALID_ID}")
        assert get_response.status_code == status.HTTP_200_OK
        updated_dataset = get_response.json()
        assert updated_dataset["name"] == "Updated Dataset Name"

    @classmethod
    async def test_failed_update_missing_name(cls, client: TestClient) -> None:
        """Test failed update when required field 'name' is missing."""
        update_data = {}
        response = client.put(
            f"/datasets/{_INVALID_ID}", json=update_data, headers={"If-Match": '"0"'}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_body = response.json()
        assert response_body["detail"][0]["loc"] == ["body", "name"]
        assert response_body["detail"][0]["msg"] == "Field required"

    @classmethod
    async def test_version_mismatch(cls, client: TestClient) -> None:
        """Test failed update when ETag doesn't match current version (lost update)."""
        update_data = {"name": "This Update Should Fail"}
        response = client.put(
            f"/datasets/{_INVALID_ID}",
            json=update_data,
            headers={"If-Match": '"999"'},  # Incorrect ETag
        )

        assert response.status_code == status.HTTP_412_PRECONDITION_FAILED
        response_body = response.json()
        assert response_body["code"] == "VERSION_MISMATCH"

        assert "ETag" in response.headers
        assert response.headers["ETag"] == '"0"'

    @classmethod
    async def test_missing_if_match_header(cls, client: TestClient) -> None:
        """Test failed update when If-Match header is not provided."""
        update_data = {"name": "This Update Should Fail Too"}
        response = client.put(
            f"/datasets/{_INVALID_ID}",
            json=update_data,
            # No If-Match header
        )

        assert response.status_code == status.HTTP_428_PRECONDITION_REQUIRED
        response_body = response.json()
        assert response_body["code"] == "VERSION_MISSING"

    @classmethod
    async def test_delete(cls, client: TestClient) -> None:
        """Test successful deletion of a dataset."""
        response = client.get("/datasets?limit=500")
        assert response.status_code == status.HTTP_200_OK
        datasets_length = len(response.json()["items"])

        response = client.delete(f"/datasets/{_VALID_ID}")
        assert response.status_code == status.HTTP_204_NO_CONTENT

        dataset = client.get(f"/datasets/{_VALID_ID}")
        assert dataset.status_code == status.HTTP_404_NOT_FOUND

        response = client.get("/datasets?limit=500")
        assert len(response.json()["items"]) == datasets_length - 1

    @classmethod
    async def test_delete_not_exist(cls, client: TestClient) -> None:
        """Test deletion of a non-existent dataset."""
        response = client.get("/datasets")
        assert response.status_code == status.HTTP_200_OK
        datasets_length = len(response.json()["items"])

        response = client.delete(f"/datasets/{_NON_EXISTENT_ID}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_body = response.json()
        assert response_body["code"] == "NOT_FOUND"

        response = client.get("/datasets")
        assert len(response.json()["items"]) == datasets_length
