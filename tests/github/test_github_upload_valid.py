# ruff: noqa: S101
"""Github endpoint valid input checks."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

_REPO_OWNER = "domoar"
_REPO_NAME = "BogusModel"
_BRANCH_DEFAULT = "main"
_BRANCH_DIFF = "DifferentBranch-Full"


@pytest.mark.github
def test_load_bogus_model_tag(client: TestClient) -> None:
    """Test uploading a model with a specific tag succeeds."""
    payload_main = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DEFAULT
    }

    response_main = client.post("/uploads/github", json=payload_main)
    assert response_main.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_branch_tag(client: TestClient) -> None:
    """Test uploading a model with a non-default branch tag succeeds."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DIFF
    }
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_without_tag(client: TestClient) -> None:
    """Test uploading a model without specifying a tag defaults to main."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME
    }
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_empty_tag(client: TestClient) -> None:
    """Test uploading a model with an empty tag is treated as main."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": ""
    }
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
