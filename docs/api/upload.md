# Upload API

The upload module provides model upload capabilities from various sources including Hugging Face, GitHub, and local files.

## API Endpoints

::: vectorize.upload.router

### HuggingFace Service

::: vectorize.upload.huggingface_service

### Local Service

::: vectorize.upload.local_service

### Github Service

::: vectorize.upload.github_service

## Repository Layer

::: vectorize.upload.repository

## Data Models

::: vectorize.upload.models

## Schemas

::: vectorize.upload.schemas

## Background Tasks

::: vectorize.upload.tasks

## Exceptions

::: vectorize.upload.exceptions

This document provides an overview of the three primary endpoints for uploading AI models to the system, including usage instructions and a brief explanation of their internal processes. The endpoints support uploading models from Hugging Face, GitHub, and local ZIP archives.

## Endpoints Overview

The API provides three endpoints for uploading models, each tailored to a specific source. All endpoints are asynchronous, initiating background tasks to process uploads and returning a `201 Created` response with a `Location` header pointing to the task status endpoint (`/v1/upload/tasks/{task_id}`).

The three sources to upload models from are:

- [Huggingface](#1-upload-hugging-face-model-post-huggingface)
- [GitHub](#2-upload-github-model-post-github)
- [Local](#3-upload-local-zip-archive-post-local)

### 1. Upload Hugging Face Model (`POST /huggingface`)

**Purpose**: Upload a model from the Hugging Face model hub by specifying its model tag and revision.

**Usage**:

- **Method**: `POST`
- **Endpoint**: `/huggingface`
- **Content-Type**: `application/json`
- **Request Body**:

  ```json
  {
    "model_tag": "bert-base-uncased",
    "revision": "main"
  }
  ```

  - `model_tag` (required): The Hugging Face model identifier (e.g., `bert-base-uncased`).
  - `revision` (optional): The model version or branch (defaults to `main`).
- **Response**:
  - Status: `201 Created`
  - Headers: `Location: /v1/upload/tasks/{task_id}`
  - Body: Empty
- **Errors**:
  - `409 Conflict`: If the model already exists in the database.
  - `404 Not Found`: If the model or revision is not found on Hugging Face.
  - `500 Internal Server Error`: If an unexpected error occurs during validation.

**Example**:

```bash
curl -X POST http://api.example.com/v1/upload/huggingface \
-H "Content-Type: application/json" \
-d '{"model_tag": "bert-base-uncased", "revision": "main"}'
```

**Under the Hood**:

- Validates the model’s existence on Hugging Face using the `huggingface_hub` library.
- Checks if the model already exists in the database.
- Creates an `UploadTask` with `PENDING` status and saves it to the database.
- Initiates a background task (`process_huggingface_model_bg`) using Dramatiq to download the model, cache it locally, and save metadata to the database.
- Updates the task status to `DONE` or `FAILED` based on the outcome.

---

### 2. Upload GitHub Model (`POST /github`)

**Purpose**: Upload a model from a GitHub repository by specifying the owner, repository name, and revision (branch or tag).

**Usage**:

- **Method**: `POST`
- **Endpoint**: `/github`
- **Content-Type**: `application/json`
- **Request Body**:

  ```json
  {
    "owner": "huggingface",
    "repo_name": "transformers",
    "revision": "main"
  }
  ```

  - `owner` (required): GitHub username or organization.
  - `repo_name` (required): Repository name.
  - `revision` (optional): Branch or tag name (defaults to `main`).
- **Response**:
  - Status: `201 Created`
  - Headers: `Location: /v1/upload/tasks/{task_id}`
  - Body: Empty
- **Errors**:
  - `409 Conflict`: If the model already exists in the database.
  - `404 Not Found`: If the repository or revision is not found on GitHub.
  - `400 Bad Request`: If the GitHub URL is invalid.

**Example**:

```bash
curl -X POST http://api.example.com/v1/upload/github \
-H "Content-Type: application/json" \
-d '{"owner": "huggingface", "repo_name": "transformers", "revision": "main"}'
```

**Under the Hood**:

- Verifies the repository’s existence using the GitHub API.
- Checks if the model already exists in the database.
- Creates an `UploadTask` with `PENDING` status and saves it to the database.
- Initiates a background task (`process_github_model_bg`) to clone the repository, validate required files (`pytorch_model.bin`, `config.json`, `tokenizer.json`), cache them locally, and save metadata to the database.
- Updates the task status to `DONE` or `FAILED` based on the outcome.

---

### 3. Upload Local ZIP Archive (`POST /local`)

**Purpose**: Upload a ZIP archive containing one or multiple model directories, treating each top-level directory as a separate model.

**Usage**:

- **Method**: `POST`
- **Endpoint**: `/local`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `file` (required): A ZIP file containing model directories.
  - `model_name` (optional, query parameter): Base name for models if not derived from the ZIP file.
- **Response**:
  - Status: `201 Created`
  - Headers: `Location: /v1/upload/local/{model_id}` (for the first model if multiple are uploaded)
  - Body: Empty
- **Errors**:
  - `400 Bad Request`: If the file is not a valid ZIP archive or is empty.
  - `409 Conflict`: If one or more models already exist in the database.
  - `404 Not Found`: If no valid models are found in the archive.
  - `500 Internal Server Error`: If an unexpected error occurs during processing.

**Example**:

```bash
curl -X POST http://api.example.com/v1/upload/local?model_name=my_model \
-F "file=@models.zip" \
-H "Content-Type: multipart/form-data"
```

**Under the Hood**:

- Validates the ZIP file format and checks for emptiness.
- Extracts the archive to a temporary directory.
- Processes each top-level directory as a separate model, validating required files and checking for duplicates in the database.
- Saves valid models to the configured model storage directory and registers them in the database.
- Cleans up temporary files and returns a response with details of processed models.

---

### Checking Upload Status (`GET /{task_id}`)

**Purpose**: Retrieve the status of an upload task for any of the above endpoints.

**Usage**:

- **Method**: `GET`
- **Endpoint**: `/{task_id}`
- **Response**:
  - Status: `200 OK`
  - Body: JSON object containing task details (e.g., `model_tag`, `task_status`, `source`, `error_msg`).
- **Errors**:
  - `404 Not Found`: If the task ID is invalid or not found.

**Example**:

```bash
curl http://api.example.com/v1/upload/tasks/123e4567-e89b-12d3-a456-426614174000
```

**Response Example**:

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "model_tag": "bert-base-uncased@main",
  "task_status": "DONE",
  "source": "HUGGINGFACE",
  "created_at": "2025-06-22T12:00:00Z",
  "updated_at": "2025-06-22T12:05:00Z"
}
```

**Under the Hood**:

- Queries the database for the `UploadTask` associated with the provided `task_id`.
- Returns the task’s current status (`PENDING`, `DONE`, or `FAILED`) and any error messages if applicable.

---

## Notes

- **Task Status**: Use the `/v1/upload/tasks/{task_id}` endpoint to monitor the progress of uploads, as processing occurs in the background.
- **Error Handling**: The system logs detailed errors for debugging, and clients receive meaningful error messages in responses.
- **Scalability**: Background tasks are managed by Dramatiq with a maximum of three retries for robustness.
- **Caching**: Models from Hugging Face and GitHub are cached locally to avoid redundant downloads.

For further details on the API, refer to the OpenAPI documentation or contact the development team.
