# ruff: noqa: S101

"""Tests for dataset upload from Hugging Face."""

import json
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from tests.wait_for_task import wait_for_task
from vectorize.config.config import settings

_FILE_SIZE_THRESHOLD = 100


@pytest.mark.asyncio
@pytest.mark.dataset
@pytest.mark.dataset_hf
class TestHuggingFaceUpload:
    """Tests for invalid datasets."""

    @pytest.mark.parametrize(
        "dataset_tag,expected_splits,expected_subsets",
        [
            ("Intel/orca_dpo_pairs", ["train"], ["default"]),
            (
                "argilla/ultrafeedback-binarized-preferences-cleaned",
                ["train"],
                ["default"],
            ),
            ("Dahoas/full-hh-rlhf", ["train", "test"], ["default"]),
            ("argilla/distilabel-intel-orca-dpo-pairs", ["train"], ["default"]),
        ],
    )
    @staticmethod
    async def test_hf_upload_valid(
        client: TestClient,
        dataset_tag: str,
        expected_splits: list[str],
        expected_subsets: list[str],
    ) -> None:
        """Test uploading a valid Hugging Face dataset."""
        expected_columns = {"question", "positive", "negative"}

        response = client.post(
            "/datasets/huggingface", json={"dataset_tag": dataset_tag}
        )
        assert response.status_code == status.HTTP_201_CREATED

        task_id = response.headers["Location"].split("/")[-1]
        task_data = await wait_for_task(
            client, task_id, f"/datasets/huggingface/status/{task_id}"
        )

        assert task_data is not None, f"Task {task_id} did not complete"
        assert task_data["id"] == task_id
        assert task_data["task_status"] == "D"
        assert task_data["tag"] is not None
        assert task_data["error_msg"] is None

        dataset_name_base = dataset_tag.replace("/", "_")

        for subset in expected_subsets:
            for split in expected_splits:
                if subset == "default":
                    file_name = f"{dataset_name_base}_{split}.jsonl"
                else:
                    file_name = f"{dataset_name_base}_{split}_{subset}.jsonl"

                file_path = settings.dataset_upload_dir / file_name
                assert file_path.exists(), (
                    f"File {file_path} was not created for split '{split}', "
                    f"subset '{subset}'"
                )

                file_size = file_path.stat().st_size
                assert file_size > 0, f"File {file_path} is empty"
                assert file_size > _FILE_SIZE_THRESHOLD, (
                    f"File {file_path} is too small ({file_size} bytes)"
                )

                _validate_jsonl_structure(file_path, expected_columns)

    @pytest.mark.parametrize(
        "dataset_tag",
        [
            "FutureMa/realhouse",
            "Hcompany/WebClick",
            "Jiahao004/DeepTheorem",
            "yandex/alchemist",
        ],
    )
    @staticmethod
    async def test_hf_upload_invalid(client: TestClient, dataset_tag: str) -> None:
        """Test uploading an invalid file format."""
        response = client.post(
            "/datasets/huggingface", json={"dataset_tag": dataset_tag}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Location" not in response.headers

    @staticmethod
    async def test_hf_upload_not_found(client: TestClient) -> None:
        """Test uploading an invalid file format."""
        response = client.post(
            "/datasets/huggingface",
            json={"dataset_tag": "nonexistent/dataset_tag"},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Location" not in response.headers


def _validate_jsonl_structure(file_path: Path, expected_columns: set[str]) -> None:
    """Validate that JSONL file has correct structure and columns."""
    line_count = 0
    sample_size = 100

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                if line_num > sample_size:
                    break

                line = raw_line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON on line {line_num} in {file_path}: {e}")

                assert isinstance(data, dict), (
                    f"Line {line_num} in {file_path} is not a JSON object"
                )

                actual_columns = set(data.keys())
                missing_columns = expected_columns - actual_columns
                assert not missing_columns, (
                    f"Line {line_num} in {file_path} missing columns: "
                    f"{missing_columns}. Found columns: {actual_columns}"
                )

                extra_columns = actual_columns - expected_columns
                assert not extra_columns, (
                    f"Line {line_num} in {file_path} has unexpected columns: "
                    f"{extra_columns}. Expected only: {expected_columns}"
                )

                for column in expected_columns:
                    value = data[column]
                    assert value is not None and str(value).strip(), (
                        f"Line {line_num} in {file_path}: column '{column}' "
                        f"is empty or None"
                    )

                line_count += 1

        assert line_count > 0, f"No valid JSON lines found in {file_path}"

    except Exception as e:
        pytest.fail(f"Error reading/validating {file_path}: {e}")
