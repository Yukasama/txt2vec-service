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
    """Tests the main branch upload."""
    payload_main = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DEFAULT
    }

    response_main = client.post("uploads/github", json=payload_main)
    assert response_main.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_branch_tag(client: TestClient) -> None:
    """Tests different branch upload."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": _BRANCH_DIFF
    }
    response = client.post("uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_without_tag(client: TestClient) -> None:
    """Tests that a model can be uploaded without specifying a tag."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME
    }
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.github
def test_load_bogus_model_empty_tag(client: TestClient) -> None:
    """Tests that an empty tag string is treated like 'main'."""
    payload = {
        "owner": _REPO_OWNER,
        "repo_name": _REPO_NAME,
        "tag": ""
    }
    response = client.post("/uploads/github", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
