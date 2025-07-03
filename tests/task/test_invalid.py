# ruff: noqa: S101

"""Invalid tests for tasks endpoint."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.tasks
class TestTasksInvalidParams:
    """Tests for invalid query parameter values on the /tasks endpoint."""

    @classmethod
    @pytest.mark.parametrize(
        "url",
        [
            # status
            "/tasks?status=RUNNING",
            "/tasks?status=UNKNOWN",
            "/tasks?status=Q&status=FOO",
            "/tasks?status=1",
            "/tasks?status=",
            # limit
            "/tasks?limit=0",
            "/tasks?limit=999",
            "/tasks?limit=-5",
            "/tasks?limit=abc",
            # offset
            "/tasks?offset=-1",
            "/tasks?offset=abc",
            # within_hours
            "/tasks?within_hours=0",
            "/tasks?within_hours=-1",
            "/tasks?within_hours=abc",
            "/tasks?within_hours=",
            # task_type
            "/tasks?task_type=invalid_type",
            "/tasks?task_type=UPLOAD",
            "/tasks?task_type=model_upload_wrong",
            "/tasks?task_type=123",
            "/tasks?task_type=training&task_type=invalid_type",
            "/tasks?task_type=",
            # tag
            "/tasks?tag=/tasks?tag=" + "x" * 256,
            # baseline_id
            "/tasks?baseline_id=not-a-uuid",
            "/tasks?baseline_id=123456789",
            "/tasks?baseline_id=invalid-uuid-format",
            "/tasks?baseline_id=",
            "/tasks?baseline_id=00000000-0000-0000-0000-00000000000",
            "/tasks?baseline_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "/tasks?baseline_id=12345678-1234-1234-1234-12345678901z",
            # dataset_id
            "/tasks?dataset_id=not-a-uuid",
            "/tasks?dataset_id=123456789",
            "/tasks?dataset_id=invalid-uuid-format",
            "/tasks?dataset_id=",
            "/tasks?dataset_id=00000000-0000-0000-0000-00000000000",
            "/tasks?dataset_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "/tasks?dataset_id=12345678-1234-1234-1234-12345678901z",
            # combined invalid
            "/tasks?limit=-5&offset=abc&status=WRONG&completed=1.23&within_hours=-2",
            "/tasks?task_type=invalid&tag=&limit=abc",
            "/tasks?task_type=wrong_type&tag=invalid<>tag&status=UNKNOWN",
            "/tasks?baseline_id=invalid&dataset_id=also-invalid&limit=abc",
            "/tasks?baseline_id=not-uuid&task_type=invalid_type",
            "/tasks?dataset_id=bad-format&status=UNKNOWN&offset=-1",
        ],
    )
    async def test_invalid_query_params(cls, client: TestClient, url: str) -> None:
        """Test that invalid query param values result in 422 Unprocessable Entity."""
        response = client.get(url)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
