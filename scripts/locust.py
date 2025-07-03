"""Load testing with Locust.

Run with: uvx locust -f scripts/locust.py
Run headless with: uvx locust -f scripts/locust.py --host=http://localhost:8000 --headless -u 1 -r 1
"""  # noqa: E501

from pathlib import Path

import urllib3
from locust import HttpUser, constant_pacing, task

from tests.dataset.utils import build_files

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_DATASET_IDS = [
    "8b8c7f3e-4d2a-4b5c-9f1e-0a6f3e4d2a5b",
    "5d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b",
    "6d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b",
    "7d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b",
    "8d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b",
    "9d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b",
]
_DATASET_PATH = "/datasets"
_PRECONDITION_FAILED = 412


class AppLoadTests(HttpUser):
    """Load tests for the API."""

    base_path = Path(__file__).parent.parent / "test_data" / "datasets"
    host = "http://localhost:8000"
    wait_time = constant_pacing(0)

    def on_start(self) -> None:
        """Set up the client to ignore SSL certificate validation."""
        self.client.verify = False

    @task
    def get_all_datasets(self) -> None:
        """Get all datasets from the server."""
        self.client.get("/datasets")

    @task
    def get_single_dataset(self) -> None:
        """Get a single dataset from the server."""
        for dataset_id in _DATASET_IDS:
            self.client.get(f"/datasets/{dataset_id}")

    @task
    def put_dataset(self) -> None:
        """Update a dataset on the server."""
        for dataset_id in _DATASET_IDS:
            response = self.client.get(f"/datasets/{dataset_id}")
            etag = response.headers.get("ETag")
            with self.client.put(
                f"{_DATASET_PATH}/{dataset_id}",
                json={"name": "Updated Dataset Name"},
                headers={"If-Match": etag},
            ) as response:
                if response.status_code == _PRECONDITION_FAILED:
                    response.success()  # type: ignore[attr-defined]

    @task
    def upload_dataset(self) -> None:
        """Upload a dataset to the server."""
        files_to_upload = [
            "default.csv",
            "default.json",
            "default.xml",
            "default.xlsx",
            "infer_fields.csv",
            "custom_fields.csv",
            "default.zip",
            "partial.zip",
        ]
        for file in files_to_upload:
            file_path = self.base_path / "valid" / file
            self.client.post(_DATASET_PATH, files=build_files(file_path))

    @task
    def upload_invalid_dataset(self) -> None:
        """Upload a dataset to the server."""
        files_to_upload = ["empty.csv", "invalid.zip"]
        for file in files_to_upload:
            file_path = self.base_path / "invalid" / file
            with self.client.post(
                _DATASET_PATH, files=build_files(file_path), catch_response=True
            ) as response:
                if response.status_code in {400, 422}:
                    response.success()  # type: ignore[attr-defined]

    @task
    def get_embeddings(self) -> None:
        """Test the embeddings endpoint with different models."""
        for model_tag in ["pytorch_model", "big_model", "huge_model"]:
            payload = {"model": model_tag, "input": "This is a test sentence."}
            self.client.post("/embeddings", json=payload)
