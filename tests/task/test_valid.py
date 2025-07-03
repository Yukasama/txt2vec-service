# ruff: noqa: S101

"""Valid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.task.task_status import TaskStatus

_TASK_TYPE_OPTIONS = {
    "model_upload",
    "synthesis",
    "dataset_upload",
    "training",
    "evaluation",
}

_BASELINE_ID = "7d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_DATASET_ID1 = "0a9d5e87-e497-4737-9829-2070780d10df"
_DATASET_ID2 = "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"


@pytest.mark.asyncio
@pytest.mark.tasks
class TestAllTasksValid:
    """Tests for valid tasks endpoint requests."""

    @classmethod
    async def test_get_all_tasks_default(cls, client: TestClient) -> None:
        """Test getting all tasks with default parameters."""
        default_limit = 10
        response = client.get("/tasks")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        assert isinstance(page, dict)
        assert set(page.keys()) == {"items", "total", "limit", "offset"}

        tasks = page["items"]
        assert isinstance(tasks, list)
        assert page["limit"] == default_limit
        assert page["offset"] == 0
        assert page["total"] >= len(tasks)

    @classmethod
    async def test_get_tasks_pagination_offset_behavior(
        cls, client: TestClient
    ) -> None:
        """Test that pagination offset works correctly."""
        limit = 3

        first_page_response = client.get(f"/tasks?limit={limit}&offset=0")
        assert first_page_response.status_code == status.HTTP_200_OK
        first_page = first_page_response.json()

        second_page_response = client.get(f"/tasks?limit={limit}&offset={limit}")
        assert second_page_response.status_code == status.HTTP_200_OK
        second_page = second_page_response.json()

        first_ids = {task["id"] for task in first_page["items"]}
        second_ids = {task["id"] for task in second_page["items"]}
        assert first_ids.isdisjoint(second_ids)

        assert first_page["total"] == second_page["total"]

    @classmethod
    async def test_get_tasks_pagination_limit_constraints(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint pagination limits."""
        limit = 5
        response = client.get(f"/tasks?limit={5}")
        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        assert page["limit"] == limit
        assert len(page["items"]) <= limit

    @classmethod
    async def test_get_tasks_status_filter_running(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by RUNNING status."""
        response = client.get("/tasks?status=R")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.RUNNING.value

    @classmethod
    async def test_get_tasks_status_filter_done(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by DONE status."""
        response = client.get("/tasks?status=D")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for action in tasks:
            assert action["task_status"] == TaskStatus.DONE.value

    @classmethod
    async def test_get_tasks_multiple_status_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by multiple statuses."""
        response = client.get("/tasks?status=R&status=D")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        valid_statuses = {TaskStatus.RUNNING.value, TaskStatus.DONE.value}
        for action in tasks:
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_tasks_within_hours_recent(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=1 (recent tasks only)."""
        response = client.get("/tasks?within_hours=1")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for action in tasks:
            assert "created_at" in action
            assert "end_date" in action

    @classmethod
    async def test_get_tasks_within_hours_extended(cls, client: TestClient) -> None:
        """Test tasks endpoint with within_hours=3 (includes older tasks)."""
        results = 2
        response = client.get("/tasks?within_hours=3")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) >= results

    @classmethod
    async def test_get_tasks_combined_filters(cls, client: TestClient) -> None:
        """Test tasks endpoint with multiple filters combined."""
        limit = 5
        response = client.get(f"/tasks?limit={limit}&status=R&within_hours=2")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

        for action in tasks:
            assert action["task_status"] == TaskStatus.RUNNING.value
            assert "created_at" in action
            assert action["task_type"] in _TASK_TYPE_OPTIONS

    @classmethod
    async def test_tasks_response_structure(cls, client: TestClient) -> None:
        """Test that tasks response has correct structure."""
        response = client.get("/tasks?limit=1")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        if tasks:
            action = tasks[0]
            required_fields = ["id", "task_status", "created_at", "task_type"]
            for field in required_fields:
                assert field in action

            assert action["task_type"] in _TASK_TYPE_OPTIONS
            valid_statuses = [status.value for status in TaskStatus]
            assert action["task_status"] in valid_statuses

    @classmethod
    async def test_get_tasks_tag_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by specific tag."""
        response = client.get("/tasks?tag=example-hf-model")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for task in tasks:
            if task["tag"] is not None:
                assert task["tag"] == "example-hf-model"

    @classmethod
    async def test_get_tasks_tag_filter_nonexistent(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by non-existent tag."""
        response = client.get("/tasks?tag=nonexistent-tag")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    @classmethod
    async def test_get_tasks_single_task_type_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by single task type."""
        response = client.get("/tasks?task_type=model_upload")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for task in tasks:
            assert task["task_type"] == "model_upload"

    @classmethod
    async def test_get_tasks_multiple_task_type_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by multiple task types."""
        response = client.get("/tasks?task_type=model_upload&task_type=dataset_upload")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        valid_types = {"model_upload", "dataset_upload"}
        for task in tasks:
            assert task["task_type"] in valid_types

    @classmethod
    async def test_get_tasks_task_type_filter_all_types(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with all possible task types."""
        all_types = [
            "model_upload",
            "synthesis",
            "dataset_upload",
            "training",
            "evaluation",
        ]

        for task_type in all_types:
            response = client.get(f"/tasks?task_type={task_type}")

            assert response.status_code == status.HTTP_200_OK
            page = response.json()
            tasks = page["items"]
            assert isinstance(tasks, list)

            for task in tasks:
                assert task["task_type"] == task_type

    @classmethod
    async def test_get_tasks_combined_tag_and_task_type_filter(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with both tag and task type filters."""
        response = client.get("/tasks?tag=example-hf-model&task_type=model_upload")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        for task in tasks:
            assert task["task_type"] == "model_upload"
            if task["tag"] is not None:
                assert task["tag"] == "example-hf-model"

    @classmethod
    async def test_get_tasks_complex_filters_combination(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with tag, task type, status, and limit filters."""
        limit = 5
        response = client.get(
            f"/tasks?tag=training_task&task_type=training&status=R&limit={limit}"
        )

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) <= limit

        for task in tasks:
            assert task["task_type"] == "training"
            assert task["task_status"] == TaskStatus.RUNNING.value
            if task["tag"] is not None:
                assert task["tag"] == "training_task"

    @classmethod
    async def test_get_tasks_no_task_type_returns_all(cls, client: TestClient) -> None:
        """Test that not specifying task_type returns all task types."""
        response = client.get("/tasks?limit=100")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)

        found_types = {task["task_type"] for task in tasks}
        assert len(found_types) > 1
        assert found_types.issubset(_TASK_TYPE_OPTIONS)

    @classmethod
    async def test_get_tasks_empty_task_type_array(cls, client: TestClient) -> None:
        """Test tasks endpoint behavior with empty task type array."""
        response = client.get("/tasks?task_type=")

        assert response.status_code in {
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        }

    @classmethod
    async def test_get_tasks_invalid_task_type(cls, client: TestClient) -> None:
        """Test tasks endpoint with invalid task type."""
        response = client.get("/tasks?task_type=invalid_type")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTrainPaginatedTasksValid:
    """Tests for valid paginated training and evaluation tasks endpoint requests."""

    @classmethod
    async def test_get_tasks_with_limit(cls, client: TestClient) -> None:
        """Test tasks endpoint with limit parameter."""
        limit = 2
        response = client.get(f"/tasks?limit={limit}")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) <= limit
        assert page["limit"] == limit

    @classmethod
    async def test_get_tasks_with_offset(cls, client: TestClient) -> None:
        """Test tasks endpoint with offset parameter."""
        all_response = client.get("/tasks?limit=100")
        all_page = all_response.json()
        all_tasks = all_page["items"]

        if len(all_tasks) > 1:
            response = client.get("/tasks?offset=1")
            assert response.status_code == status.HTTP_200_OK
            page = response.json()
            offset_tasks = page["items"]
            assert isinstance(offset_tasks, list)
            assert len(offset_tasks) <= len(all_tasks)
            assert page["offset"] == 1

    @classmethod
    async def test_get_tasks_pagination_structure(cls, client: TestClient) -> None:
        """Test tasks endpoint pagination structure."""
        limit = 5
        response = client.get(f"/tasks?limit={limit}")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()

        required_fields = {"items", "total", "limit", "offset"}
        assert set(page.keys()) == required_fields

        assert isinstance(page["items"], list)
        assert isinstance(page["total"], int)
        assert isinstance(page["limit"], int)
        assert isinstance(page["offset"], int)

        assert page["limit"] == limit
        assert page["offset"] == 0
        assert page["total"] >= 0
        assert len(page["items"]) <= limit


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTrainEvaluationTasksValid:
    """Tests for valid training and evaluation tasks endpoint requests."""

    @classmethod
    async def test_get_tasks_baseline_id_filter(cls, client: TestClient) -> None:
        """Test tasks endpoint filtering by baseline_id."""
        response = client.get(f"/tasks?baseline_id={_BASELINE_ID}")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) == 1

        for task in tasks:
            assert task["task_type"] == "training"
            assert task["baseline_id"] == _BASELINE_ID

    @classmethod
    async def test_get_tasks_dataset_id_filter_training_dataset_1(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint filtering by dataset_id."""
        tasks_with_datasetid = 2
        response = client.get(f"/tasks?dataset_id={_DATASET_ID1}")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) >= tasks_with_datasetid

        found_types = {task["task_type"] for task in tasks}
        assert found_types == {"training", "evaluation"}

    @classmethod
    async def test_get_tasks_dataset_id_filter_training_dataset_2(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint filtering by dataset_id."""
        response = client.get(f"/tasks?dataset_id={_DATASET_ID2}")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) >= 1

        found_types = {task["task_type"] for task in tasks}
        assert found_types == {"training", "evaluation"}

    @classmethod
    async def test_get_tasks_dataset_id_with_task_type_filter(
        cls, client: TestClient
    ) -> None:
        """Test tasks endpoint with dataset_id and task_type filters combined."""
        response = client.get(f"/tasks?dataset_id={_DATASET_ID1}&task_type=training")

        assert response.status_code == status.HTTP_200_OK
        page = response.json()
        tasks = page["items"]
        assert isinstance(tasks, list)
        assert len(tasks) >= 1

        found_types = {task["task_type"] for task in tasks}
        assert found_types == {"training", "evaluation"}
