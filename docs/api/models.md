# AI Model API

The AI model module provides model management, retrieval, and CRUD operations.

## API Endpoints

::: vectorize.ai_model.router

## Service Layer

::: vectorize.ai_model.service

## Repository Layer

::: vectorize.ai_model.repository

## Data Models

::: vectorize.ai_model.models

## Schemas

::: vectorize.ai_model.schemas

## Exceptions

::: vectorize.ai_model.exceptions

This document describes the available API endpoints for managing AI models in the system, including retrieval, updates, and deletion. The endpoints provide ETag-based version control, pagination, and follow RESTful conventions using FastAPI and SQLModel.

## Endpoints Overview

The following REST endpoints are available:

- [Get All Models (paged)](#1-get-all-models)
- [Get Single Model by Tag (with ETag)](#2-get-model-by-tag)
- [Update Model (ETag / If-Match)](#3-update-model)
- [Delete Model](#4-delete-model)

---

### 1. Get All Models

**Purpose**: Returns a paginated list of all registered AI models.

**Usage**:

- **Method**: `GET`
- **Endpoint**: `/v1/models`
- **Query Parameters**:
  - `page` (int): Page number, starting from `1`. Default: `1`.
  - `size` (int): Number of items per page (`5–100`). Default: `5`.

**Response**:

- Status: `200 OK`
- Body:

```json
{
  "page": 1,
  "size": 5,
  "totalpages": 3,
  "items": [
    {
      "id": "uuid",
      "name": "bert-base",
      "model_tag": "bert-base-uncased",
      "source": "HUGGINGFACE",
      "version": 1,
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

**Errors**:

- `404 Not Found`: No models in the database.

**Under the Hood**:

- Executes a `SELECT COUNT(*)` and paged query via SQLModel.
- Constructs a `PagedResponse` and logs context with pagination metadata.

---

### 2. Get Model by Tag

**Purpose**: Fetch a single model by its unique `model_tag`.

**Usage**:

- **Method**: `GET`
- **Endpoint**: `/v1/models/{model_tag}`
- **Headers (optional)**:
  - `If-None-Match`: ETag for conditional request.

**Response**:

- Status: `200 OK` (with model payload and `ETag`)
- Status: `304 Not Modified` if ETag matches current version.

**Errors**:

- `404 Not Found`: If no model exists with that tag.

**Under the Hood**:

- Looks up model via `model_tag`, transforms to public model.
- Uses `ETag` for efficient client-side caching and conditional GETs.

---

### 3. Update Model

**Purpose**: Update model fields (e.g., `name`) — requires ETag match.

**Usage**:

- **Method**: `PUT`
- **Endpoint**: `/v1/models/{model_id}`
- **Headers**:
  - `If-Match`: Current ETag (version) required for optimistic locking.
- **Request Body**:

```json
{
  "name": "bert-base-renamed"
}
```

**Response**:

- Status: `204 No Content`
  - Headers:
    - `ETag`: New version
    - `Location`: Updated resource path

**Errors**:

- `412 Precondition Failed`: Missing `If-Match`
- `409 Conflict`: Version mismatch
- `404 Not Found`: Model not found

**Under the Hood**:

- Uses version from ETag for safe update.
- Increments version on success and updates DB.
- Returns new version in `ETag`.

---

### 4. Delete Model

**Purpose**: Permanently deletes the model from the database.

**Usage**:

- **Method**: `DELETE`
- **Endpoint**: `/v1/models/{model_id}`

**Response**:

- Status: `204 No Content`

**Errors**:

- `404 Not Found`: Model does not exist.

**Under the Hood**:

- Deletes model and cascaded relations.
- Logs operation and commits transaction.

---

## Additional Notes

- **Versioning**: All modifying operations (PUT, DELETE) rely on ETag-based versioning.
- **Error Handling**: Uses domain-specific exceptions like `ModelNotFoundError`, `VersionMismatchError`.
- **ETag Support**: Implements `If-None-Match` and `If-Match` per RFC 7232.
- **Logging**: All actions are logged with model context using `loguru`.

---

## Example: Safe Update with curl

```bash
curl -X PUT http://api.example.com/v1/models/8f5e0a52... \
-H "Content-Type: application/json" \
-H 'If-Match: "1"' \
-d '{"name": "renamed-model"}'
```