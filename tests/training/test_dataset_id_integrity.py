# ruff: noqa: S101

"""Test dataset ID integrity in training tasks."""

import shutil
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import settings
from vectorize.training.repository import get_train_task_by_id_db

# Test constants matching those in other test files
DATASET_ID_1 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"
DATASET_ID_2 = "0a9d5e87-e497-4737-9829-2070780d10df"
MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"

# HTTP Status codes
HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED


def ensure_minilm_model_available() -> None:
    """Ensure the required model files are present for training tests."""
    src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
    dst = settings.model_upload_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if not dst.exists() and src.exists():
        shutil.copytree(src, dst)


@pytest.mark.training
class TestDatasetIdIntegrity:
    """Test that training tasks store correct dataset IDs."""

    @staticmethod
    def test_training_task_stores_correct_single_dataset_id(
        client: TestClient,
    ) -> None:
        """Test that training task stores correct dataset ID for single dataset."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Get task ID from location header
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # Check the status immediately to get the stored dataset IDs
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        assert "train_dataset_ids" in status_data
        assert status_data["train_dataset_ids"] == [DATASET_ID_1]
        assert status_data.get("val_dataset_id") is None

    @staticmethod
    def test_training_task_stores_correct_multiple_dataset_ids(
        client: TestClient,
    ) -> None:
        """Test that training task stores correct dataset IDs for multiple datasets."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1, DATASET_ID_2],
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Get task ID from location header
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # Check the status immediately to get the stored dataset IDs
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        assert "train_dataset_ids" in status_data
        assert status_data["train_dataset_ids"] == [DATASET_ID_1, DATASET_ID_2]
        assert status_data.get("val_dataset_id") is None

    @staticmethod
    def test_training_task_stores_correct_validation_dataset_id(
        client: TestClient,
    ) -> None:
        """Test that training task stores correct validation dataset ID."""
        ensure_minilm_model_available()

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": [DATASET_ID_1],
            "val_dataset_id": DATASET_ID_2,
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Get task ID from location header
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # Check the status immediately to get the stored dataset IDs
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        assert "train_dataset_ids" in status_data
        assert "val_dataset_id" in status_data
        assert status_data["train_dataset_ids"] == [DATASET_ID_1]
        assert status_data["val_dataset_id"] == DATASET_ID_2

    @staticmethod
    @pytest.mark.asyncio
    async def test_dataset_ids_match_between_request_and_database(
        client: TestClient,
        session: AsyncSession,
    ) -> None:
        """Test that dataset IDs in the database match the original request."""
        ensure_minilm_model_available()

        original_train_ids = [DATASET_ID_1, DATASET_ID_2]
        original_val_id = DATASET_ID_1  # Using DATASET_ID_1 as validation

        payload = {
            "model_tag": MINILM_MODEL_TAG,
            "train_dataset_ids": original_train_ids,
            "val_dataset_id": original_val_id,
            "epochs": 1,
            "learning_rate": 0.00005,
            "per_device_train_batch_size": 8,
        }

        response = client.post("/training/train", json=payload)
        assert response.status_code == HTTP_202_ACCEPTED

        # Get task ID from location header
        location = response.headers.get("Location")
        assert location is not None
        task_id = location.split("/")[-2]

        # Directly check the database record
        task = await get_train_task_by_id_db(session, UUID(task_id))
        assert task is not None

        # Verify that the stored IDs match the original request exactly
        assert task.train_dataset_ids == original_train_ids
        assert task.val_dataset_id == original_val_id

        # Also verify via API response
        status_response = client.get(f"/training/{task_id}/status")
        assert status_response.status_code == HTTP_200_OK

        status_data = status_response.json()
        assert status_data["train_dataset_ids"] == original_train_ids
        assert status_data["val_dataset_id"] == original_val_id
