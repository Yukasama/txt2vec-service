# ruff: noqa: S101, PLR6301

"""Tests for evaluation service integration with training tasks."""

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import settings
from vectorize.dataset.classification import Classification
from vectorize.dataset.dataset_source import DatasetSource
from vectorize.dataset.models import Dataset
from vectorize.dataset.repository import upload_dataset_db
from vectorize.evaluation.schemas import EvaluationRequest
from vectorize.evaluation.service import resolve_evaluation_dataset_svc
from vectorize.training.models import TrainingTask
from vectorize.training.repository import (
    save_training_task_db,
    update_training_task_validation_dataset_db,
)

# Test constants
MINILM_MODEL_TAG = "models--sentence-transformers--all-MiniLM-L6-v2"
HTTP_200_OK = status.HTTP_200_OK
HTTP_202_ACCEPTED = status.HTTP_202_ACCEPTED
MAX_RESULTS_SAMPLES = 10  # Constant for magic number
FLOAT_TOLERANCE = 0.001  # Constant for float comparison tolerance

# Common test data
DEFAULT_TEST_DATA = [
    {"question": "test", "positive": "pos", "negative": "neg"}
]
MULTI_ITEM_TEST_DATA = [
    {
        "question": "What is AI?",
        "positive": "AI is artificial intelligence",
        "negative": "AI is not real",
    },
    {
        "question": "What is ML?",
        "positive": "ML is machine learning",
        "negative": "ML is not learning",
    },
]
CONCURRENT_TEST_DATA = [
    {
        "question": "Concurrent test",
        "positive": "Good answer",
        "negative": "Bad answer",
    }
]


class TestDatasetFactory:
    """Factory for creating test datasets with automatic cleanup."""

    @staticmethod
    @asynccontextmanager
    async def create_dataset(
        session: AsyncSession,
        test_data: "list[dict[str, str]] | None" = None,
        dataset_name: "str | None" = None
    ) -> "AsyncIterator[tuple[uuid.UUID, Path]]":
        """Create a test dataset with automatic cleanup."""
        if test_data is None:
            test_data = DEFAULT_TEST_DATA

        dataset_id = uuid.uuid4()
        dataset_path = settings.dataset_upload_dir / f"{dataset_id}.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Write test data to file
        with dataset_path.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Save dataset to database
        dataset = Dataset(
            id=dataset_id,
            file_name=f"{dataset_id}.jsonl",
            name=dataset_name or f"Test Dataset {dataset_id}",
            classification=Classification.SENTENCE_TRIPLES,
            rows=len(test_data),
            source=DatasetSource.LOCAL,
        )
        await upload_dataset_db(session, dataset)

        try:
            yield dataset_id, dataset_path
        finally:
            # Cleanup
            if dataset_path.exists():
                dataset_path.unlink()


class EvaluationTestHelper:
    """Helper class for common evaluation test operations."""

    @staticmethod
    def ensure_test_model_available() -> None:
        """Ensure the required model files are present for integration tests."""
        src = Path("test_data/training/models--sentence-transformers--all-MiniLM-L6-v2")
        dst = (
            settings.model_upload_dir
            / "models--sentence-transformers--all-MiniLM-L6-v2"
        )
        if not dst.exists() and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

    @staticmethod
    def wait_for_evaluation_completion(
        client: TestClient, task_id: str, max_wait: int = 10
    ) -> dict[str, Any]:
        """Wait for evaluation task to complete and return final status."""
        for _ in range(max_wait):
            response = client.get(f"/evaluation/{task_id}/status")
            if response.status_code == HTTP_200_OK:
                data = response.json()
                if data["status"] in {"D", "F"}:
                    return data
            time.sleep(1)

        # Return last known status
        response = client.get(f"/evaluation/{task_id}/status")
        return response.json() if response.status_code == HTTP_200_OK else {}

    @staticmethod
    def extract_task_id_from_location(location: str) -> str:
        """Extract task ID from Location header."""
        return location.split("/")[-2]  # Extract from "/evaluation/{task_id}/status"

    @staticmethod
    def start_evaluation(
        client: TestClient,
        dataset_id: str,
        model_tag: str = MINILM_MODEL_TAG,
        baseline_model_tag: "str | None" = None,
        max_samples: int = 1
    ) -> tuple[str, dict]:
        """Start an evaluation and return task_id and response data."""
        evaluation_payload = {
            "model_tag": model_tag,
            "dataset_id": str(dataset_id),
            "max_samples": max_samples,
        }

        if baseline_model_tag:
            evaluation_payload["baseline_model_tag"] = baseline_model_tag

        response = client.post("/evaluation/evaluate", json=evaluation_payload)
        assert response.status_code == HTTP_202_ACCEPTED

        location = response.headers.get("Location")
        assert location is not None
        task_id = EvaluationTestHelper.extract_task_id_from_location(location)

        return task_id, response.json() if response.content else {}

    @staticmethod
    def verify_status_response_structure(status_data: dict, task_id: str) -> None:
        """Verify the structure of a status response."""
        # Verify required fields are present
        required_fields = ["task_id", "status", "created_at", "updated_at", "progress"]
        for field in required_fields:
            assert field in status_data, f"Missing required field: {field}"

        # Verify metadata fields are present in schema (even if None)
        metadata_fields = ["model_tag", "baseline_model_tag"]
        for field in metadata_fields:
            assert field in status_data, f"Missing metadata field: {field}"

        # Verify task ID is correct
        assert status_data["task_id"] == task_id
        assert status_data["status"] in {"Q", "R", "D", "F"}


class TestEvaluationIntegration:
    """Test integration between evaluation and training systems."""

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_training_task_id(
        self, session: AsyncSession
    ) -> None:
        """Test resolving dataset using training_task_id."""
        # Create test training task
        task_id = uuid4()
        task = TrainingTask(id=task_id, model_tag="test-model")
        await save_training_task_db(session, task)

        # Create validation dataset and save to database
        async with TestDatasetFactory.create_dataset(
            session, DEFAULT_TEST_DATA, "Validation Test Dataset"
        ) as (dataset_id, validation_path):
            # Update task with validation dataset ID
            await update_training_task_validation_dataset_db(
                session, task_id, str(dataset_id)
            )

            # Test request with training_task_id
            request = EvaluationRequest(
                model_tag="test-model", training_task_id=str(task_id), max_samples=100
            )

            result_path = await resolve_evaluation_dataset_svc(session, request)
            assert result_path == validation_path

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_both_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing both dataset_id and training_task_id fails."""
        request = EvaluationRequest(
            model_tag="test-model",
            dataset_id=str(uuid4()),
            training_task_id=str(uuid4()),
            max_samples=100,
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            await resolve_evaluation_dataset_svc(session, request)

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_no_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing neither dataset_id nor training_task_id fails."""
        request = EvaluationRequest(model_tag="test-model", max_samples=100)

        with pytest.raises(ValueError, match="Must specify either"):
            await resolve_evaluation_dataset_svc(session, request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluation_metadata_integration(
        self, client: TestClient, session: AsyncSession
    ) -> None:
        """Test end-to-end integration of evaluation metadata functionality."""
        EvaluationTestHelper.ensure_test_model_available()

        async with TestDatasetFactory.create_dataset(session) as (dataset_id, _):
            # Start evaluation
            task_id, _ = EvaluationTestHelper.start_evaluation(client, str(dataset_id))

            # Check immediate status
            immediate_status = client.get(f"/evaluation/{task_id}/status")
            assert immediate_status.status_code == HTTP_200_OK
            immediate_data = immediate_status.json()
            assert immediate_data["task_id"] == task_id

            # Verify metadata fields are present
            metadata_fields = ["model_tag", "baseline_model_tag"]
            for field in metadata_fields:
                assert field in immediate_data, f"Missing metadata field: {field}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluation_service_integration(
        self, client: TestClient, session: AsyncSession
    ) -> None:
        """Test end-to-end integration of the evaluation service."""
        EvaluationTestHelper.ensure_test_model_available()

        async with TestDatasetFactory.create_dataset(session) as (dataset_id, _):
            # Start evaluation
            task_id, _ = EvaluationTestHelper.start_evaluation(client, str(dataset_id))

            # Check status
            status_response = client.get(f"/evaluation/{task_id}/status")
            assert status_response.status_code == HTTP_200_OK

            status_data = status_response.json()
            EvaluationTestHelper.verify_status_response_structure(status_data, task_id)


@pytest.mark.evaluation
@pytest.mark.integration
class TestEvaluationMetadataIntegration:
    """Integration tests for evaluation API with metadata functionality."""

    @staticmethod
    def test_evaluation_api_metadata_with_dataset(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that evaluation API stores and returns metadata."""

        async def run_test() -> None:
            EvaluationTestHelper.ensure_test_model_available()

            async with TestDatasetFactory.create_dataset(
                session,
                MULTI_ITEM_TEST_DATA,
                "Test Dataset with Multiple Items"
            ) as (dataset_id, _):
                # Start evaluation with baseline model
                task_id, _ = EvaluationTestHelper.start_evaluation(
                    client,
                    str(dataset_id),
                    baseline_model_tag=MINILM_MODEL_TAG,
                    max_samples=2
                )

                # Check immediate status
                immediate_status = client.get(f"/evaluation/{task_id}/status")
                assert immediate_status.status_code == HTTP_200_OK
                immediate_data = immediate_status.json()
                assert immediate_data["task_id"] == task_id

                # Wait briefly and check for metadata
                final_status = EvaluationTestHelper.wait_for_evaluation_completion(
                    client, task_id, max_wait=5
                )

                # Verify basic response structure
                assert "task_id" in final_status
                assert "status" in final_status
                assert final_status["task_id"] == task_id

        # Run the async test
        asyncio.run(run_test())

    @staticmethod
    def test_evaluation_api_response_structure_with_metadata(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that evaluation API response includes metadata fields."""

        async def run_test() -> None:
            EvaluationTestHelper.ensure_test_model_available()

            async with TestDatasetFactory.create_dataset(
                session,
                [
                    {
                        "question": "Test question",
                        "positive": "Positive answer",
                        "negative": "Negative answer",
                    }
                ],
                "Simple Test Dataset",
            ) as (dataset_id, _):
                # Start evaluation
                task_id, _ = EvaluationTestHelper.start_evaluation(
                    client,
                    str(dataset_id),
                )

                # Check status response structure
                status_response = client.get(
                    f"/evaluation/{task_id}/status"
                )
                assert status_response.status_code == HTTP_200_OK

                status_data = status_response.json()
                EvaluationTestHelper.verify_status_response_structure(
                    status_data,
                    task_id,
                )

        # Run the async test
        asyncio.run(run_test())

    @staticmethod
    def test_evaluation_api_concurrent_requests_with_metadata(
        client: TestClient, session: AsyncSession
    ) -> None:
        """Test that concurrent evaluation requests handle metadata correctly."""

        async def run_test() -> None:
            EvaluationTestHelper.ensure_test_model_available()

            # Create two test datasets concurrently (SIM117: single with statement)
            async with TestDatasetFactory.create_dataset(
                session, CONCURRENT_TEST_DATA, "Concurrent Test Dataset 1"
            ) as (dataset_id_1, _), TestDatasetFactory.create_dataset(
                session, CONCURRENT_TEST_DATA, "Concurrent Test Dataset 2"
            ) as (dataset_id_2, _):

                # Start two evaluations concurrently (E501: break lines)
                task_id_1, _ = EvaluationTestHelper.start_evaluation(
                    client, str(dataset_id_1)
                )
                task_id_2, _ = EvaluationTestHelper.start_evaluation(
                    client, str(dataset_id_2)
                )

                # Verify both tasks have different IDs
                assert task_id_1 != task_id_2

                # Verify both have valid status responses
                for task_id in [task_id_1, task_id_2]:
                    status_response = client.get(f"/evaluation/{task_id}/status")
                    assert status_response.status_code == HTTP_200_OK

                    status_data = status_response.json()
                    EvaluationTestHelper.verify_status_response_structure(
                        status_data, task_id
                    )

        # Run the async test
        asyncio.run(run_test())
