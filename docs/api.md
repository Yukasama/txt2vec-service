# API Reference

Welcome to the Vectorize API documentation. This section provides comprehensive documentation for all API endpoints, services, and modules.

## API Overview

The Vectorize API is organized into several main modules:

- **[Common/Health](api/common.md)**: Health checks and basic utilities
- **[Datasets](api/datasets.md)**: Dataset upload from HuggingFace and local sources plus CRUD operations
- **[AI Models](api/models.md)**: Model management and CRUD operations
- **[Training](api/training.md)**: Model training with background task management
- **[Evaluation](api/evaluation.md)**: Model evaluation and performance metrics
- **[Inference](api/inference.md)**: Text embedding generation and inference
- **[Upload](api/upload.md)**: Model upload from HuggingFace, GitHub and local sources
- **[Synthesis](api/synthesis.md)**: Synthetic data generation from Binary data and datasets
- **[Tasks](api/tasks.md)**: Background task management and monitoring

## Quick Reference

### Base URL

```
http://localhost:8000
```

or when working with Docker:

```
https://localhost/v1/api
```

### Authentication

Currently, the API does not require authentication.

### Content Types

- **Request**: `application/json`, `multipart/form-data` (for file uploads)
- **Response**: `application/json` or an empty response

### Error Handling

The API uses standardized error codes defined in `config/errors.py` for consistent error handling across all endpoints. This helps differenciating errors easier in the Web application. Error responses follow this format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error description"
}
```

Common error codes include:

- `SERVER_ERROR`: Internal server errors
- `NOT_FOUND`: Resource not found
- `INVALID_FILE`: File validation errors
- `VERSION_MISMATCH`: ETag version conflicts
- `MODEL_ALREADY_EXISTS`: Duplicate model creation attempts

### Common Response Codes

- `200 OK`: Successful operation
- `201 Created`: Resource created successfully
- `202 Accepted`: Request accepted for background processing
- `304 Not Modified`: Resource is already up to date
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `412 Precondition Failed`: ETag condition not met
- `422 Unprocessable Entity`: Validation error
- `428 Precondition Required`: ETag header required
- `500 Internal Server Error`: Server error
