# Training Guide

This module contains all logic for training SBERT models (Sentence-BERT), including data validation, training orchestration, error handling, and utilities for REST API integration.

## Training Principle

Training is performed using **triplet losses** based on triplets consisting of:
- **Anchor (Question)**: The initial question or sentence
- **Positive**: Semantically similar answer
- **Negative**: Semantically dissimilar answer

**Goal:** The embeddings of anchor and positive should be as similar as possible (high cosine similarity), while the embeddings of anchor and negative should be as dissimilar as possible (low cosine similarity).

For more details, see the [SBERT documentation](https://www.sbert.net/docs/package_reference/losses.html).

## Overview

The training system is modular and provides:

- **Data validation**: Ensures training data is correct and complete
- **Training orchestration**: Encapsulates the entire training process (preparation, training, saving, cleanup)
- **Error handling**: Clear error classes for all typical error cases
- **Integration**: Easy connection to FastAPI endpoints and database

## Architecture

### Main Components

1. **`tasks.py`**: Central training logic (background tasks)
   - Training tasks for asynchronous execution
   - Integration with Dramatiq for task management
   - Error handling and status updates

2. **`service.py`**: Service layer for API integration
   - Request validation
   - Calling training tasks
   - Error and status management

3. **`router.py`**: REST API endpoints
   - `POST /train`: Start training
   - `GET /{task_id}/status`: Query status
   - HTTP response handling

4. **`repository.py`**: Database operations
   - CRUD operations for training tasks
   - Database model management

5. **`utils/`**: Helper functions
   - **`input_examples.py`**: Creation and filtering of training examples
   - **`model_loader.py`**: Loading and initializing models
   - **`cleanup.py`**: Resource management and cleanup
   - **`safetensors_finder.py`**: Finding model files

## Example: Training via Python

```python
from vectorize.training.service import TrainingOrchestrator
from vectorize.training.schemas import TrainRequest
from sqlmodel.ext.asyncio.session import AsyncSession

# Create training request
train_request = TrainRequest(
    model_tag="models--sentence-transformers--all-MiniLM-L6-v2",
    train_dataset_ids=["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
    epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# Use training orchestrator
db = AsyncSession()  # Your DB session
orchestrator = TrainingOrchestrator(db, task_id)

# Start training (async)
await orchestrator.run_training(
    model_path=model_path,
    train_request=train_request,
    dataset_paths=dataset_paths,
    output_dir=output_dir
)
```

## Example: Training via REST API

Training is started as a background task and returns a task ID:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
    "train_dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"],
    "epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "evaluation_steps": 500
  }'
```

Response:

```json
{
  "status_code": 202,
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

### Query status

```bash
curl "http://localhost:8000/training/{task_id}/status"
```


### All available parameters

The API supports all important Sentence-Transformers parameters:

| Parameter                   | Type           | Description                                       |
|-----------------------------|----------------|---------------------------------------------------|
| `model_tag`                 | string         | Base model from the database                      |
| `train_dataset_ids`         | string[]       | List of dataset IDs (merged)                      |
| `val_dataset_id`            | string         | Optional, for validation                          |
| `epochs`                    | int            | Number of training epochs                         |
| `per_device_train_batch_size`| int           | Batch size per device                             |
| `learning_rate`             | float          | Learning rate                                     |
| `warmup_steps`              | int            | Steps for warmup                                  |
| `optimizer_name`            | string         | Optimizer (e.g. AdamW)                            |
| `scheduler`                 | string         | Scheduler (e.g. constantlr)                       |
| `weight_decay`              | float          | Weight decay                                      |
| `max_grad_norm`             | float          | Max gradient norm                                 |
| `use_amp`                   | bool           | Automatic Mixed Precision                         |
| `show_progress_bar`         | bool           | Show progress bar                                 |
| `evaluation_steps`          | int            | Steps between evaluations                         |
| `output_path`               | string         | Output directory for model                        |
| `save_best_model`           | bool           | Save best model                                   |
| `save_each_epoch`           | bool           | Save after each epoch                             |
| `save_optimizer_state`      | bool           | Save optimizer state                              |
| `dataloader_num_workers`    | int            | Number of workers for DataLoader                  |
| `device`                    | string         | Device (e.g. cuda, cpu)                           |
| `timeout_seconds`           | int            | Timeout for training (seconds)                    |


## Error Handling

The system provides detailed error classes for various scenarios:

- **`InvalidDatasetIdError`**: Invalid UUID or missing dataset ID
- **`InvalidModelIdError`**: Invalid model or model tag not found
- **`TrainingDatasetNotFoundError`**: Training dataset does not exist in the filesystem
- **`TrainingModelWeightsNotFoundError`**: Model weights not found
- **`TrainingTaskNotFoundError`**: Training task with the given ID not found
- **`DatasetValidationError`**: Dataset has wrong structure or invalid data
- **Timeout errors**: Training exceeds the defined time limit

All errors are logged in detail and returned via the API.

## Complete JSON API Reference

### Training Request Examples

**Minimal training request:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": ["0b30b284-f7fe-4e6c-a270-17cafc5b5bcb"]
}
```

**Standard training request:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": [
    "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
    "0a9d5e87-e497-4737-9829-2070780d10df"
  ],
  "val_dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
  "epochs": 3,
  "learning_rate": 0.00005,
  "per_device_train_batch_size": 8
}
```

**Full training request with all parameters:**

```json
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": [
    "0b30b284-f7fe-4e6c-a270-17cafc5b5bcb",
    "0a9d5e87-e497-4737-9829-2070780d10df"
  ],
  "val_dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
  "epochs": 3,
  "per_device_train_batch_size": 16,
  "learning_rate": 5e-5,
  "warmup_steps": 200,
  "optimizer_name": "AdamW",
  "scheduler": "constantlr",
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "use_amp": true,
  "show_progress_bar": true,
  "evaluation_steps": 1000,
  "save_best_model": true,
  "save_each_epoch": false,
  "save_optimizer_state": false,
  "dataloader_num_workers": 0,
  "device": "cuda",
  "timeout_seconds": 7200
}
```

### Training Response Examples

**Training start response (202 Accepted):**

```json
{
  "status_code": 202,
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

**Training status response:**

```json
{
  "task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "status": "DONE",
  "created_at": "2025-06-15T21:34:47.000Z",
  "end_date": "2025-06-15T22:15:23.000Z",
  "error_msg": null,
  "trained_model_id": "trained_models/models--sentence-transformers--all-MiniLM-L6-v2-finetuned-20250615-213447-7ef54ba0"
}
```

### Data Splitting and Validation Logic

**Scenario 1: Multiple datasets with explicit validation**

```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"],
  "val_dataset_id": "validation-dataset-uuid"
}
```

→ **Result**: `validation-dataset-uuid` is used as the validation dataset

**Scenario 2: Multiple datasets without explicit validation**

```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"]
}
```

→ **Result**: System merges all datasets and splits 90% training / 10% validation

**Scenario 3: Single dataset without validation**

```json
{
  "train_dataset_ids": ["single-dataset-uuid"]
}
```

→ **Result**: Auto-split of `single-dataset-uuid` → 90% training / 10% validation

### Data Path Examples

**Training creates these paths:**

- Explicit validation: `data/datasets/my_validation_dataset.jsonl`
- Auto-split validation: `data/datasets/my_training_dataset.jsonl#auto-split`

### Parameter Rules

**Required:**

- `model_tag`: Always required
- `train_dataset_ids`: At least one dataset (array with min. 1 element)

**Optional:**

- `val_dataset_id`: For explicit validation data
- All other parameters have default values

### Invalid Combinations

**Invalid per_device_train_batch_size:**
```json
{
  "model_tag": "...",
  "train_dataset_ids": ["..."],
  "per_device_train_batch_size": 0 // Error: must be > 0
}
```

**Invalid learning_rate:**

```json
{
  "model_tag": "...",
  "train_dataset_ids": ["..."],
  "learning_rate": 0 // Error: must be > 0
}
```

## Integration with Evaluation

After successful training, the trained model can be evaluated directly:

**1. Start training:**

```bash
POST /train
{
  "model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "train_dataset_ids": ["dataset-uuid"],
  "val_dataset_id": "validation-uuid",
  "epochs": 3
}
```

**2. Extract training task ID from response:**

```json
{
  "headers": {
    "Location": "/training/7ef54ba0-2d87-4864-8360-81de8035369a/status"
  }
}
```

**3. Evaluate trained model:**

```bash
POST /evaluation/evaluate
{
  "model_tag": "trained_models/models--sentence-transformers--all-MiniLM-L6-v2-finetuned-20250615-213447-7ef54ba0",
  "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "baseline_model_tag": "models--sentence-transformers--all-MiniLM-L6-v2"
}
```

This integration enables consistent evaluation with the same validation data used during training.

## Testing

Tests for the training module are located in `tests/training/`:

```bash
pytest tests/training/ -v
```

- **Valid tests**: Successful training with correct data
- **Invalid tests**: Error cases and invalid data

## Data Format

The system expects JSONL files with the following structure:

```json
{"question": "What is machine learning?", "positive": "ML is a subfield of AI", "negative": "The weather is nice today"}
{"question": "How does deep learning work?", "positive": "With neural networks", "negative": "Pizza tastes good"}
```

**Required columns:**

- `question`: The initial question or anchor
- `positive`: Semantically similar/correct answer
- `negative`: Semantically dissimilar/incorrect answer

**Validation:**

- All columns must be present
- No NULL values or empty strings
- At least one training entry required

## Further Reading

### What are Sentence-Transformers?

Sentence-Transformers are specialized models based on BERT, RoBERTa, or similar architectures, trained to map entire sentences or text passages as dense vectors (embeddings) in semantic space. This allows semantically similar sentences to be represented by similar vectors.

---

For details, also see the docstrings in the respective modules and the API documentation (Swagger/OpenAPI).
