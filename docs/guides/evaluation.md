# Evaluation Guide

This module provides comprehensive evaluation capabilities for SBERT (Sentence-BERT) training quality assessment using cosine similarity metrics with seamless training-evaluation integration.

## Overview

The evaluation system computes various metrics to assess how well a trained sentence transformer model performs on question-positive-negative triplets:

- **Average Cosine Similarity**: Between questions and positive/negative examples
- **Similarity Ratio**: Ratio of positive to negative similarities
- **Spearman Correlation**: Measures ranking quality between positive and negative examples
- **Baseline Comparison**: Compare trained models against baseline models with improvement metrics
- **Training-Evaluation Integration**: Seamlessly evaluate models using their training validation datasets

## Architecture

### Core Components

1. **`evaluation.py`**: Main evaluation logic

   - `EvaluationMetrics`: Container for computed metrics with baseline comparison
   - `TrainingEvaluator`: Main evaluation orchestrator with model comparison capabilities

2. **`service.py`**: Service layer for FastAPI integration

   - Database lookups for models and datasets
   - `resolve_evaluation_dataset()`: Smart dataset resolution (explicit or from training task)
   - Background task evaluation with progress tracking
   - Error handling and validation

3. **`router.py`**: FastAPI endpoints

   - RESTful evaluation API with async background processing
   - Status tracking and result retrieval

4. **`schemas.py`**: Pydantic models for API requests/responses
   - `training_task_id`: Use validation dataset from training tasks
   - Enhanced request validation and response schemas

### Utility Modules (`utils/`)

1. **`similarity_calculator.py`**: Optimized similarity computation

   - Vectorized cosine similarity calculations
   - Spearman correlation computation
   - Performance optimized for large datasets

2. **`dataset_validator.py`**: Dataset validation

   - JSONL format validation
   - Required columns checking
   - Null/empty value detection

3. **`model_resolver.py`**: Model path resolution
   - Handles HuggingFace cache structure
   - Recursive model file discovery

## Dataset Resolution

The evaluation system supports **two ways** to specify which dataset to use:

### Method 1: Training Task ID (`training_task_id`)

**Recommended approach** - uses the exact validation dataset from training

When you train a model, the system automatically:
1. **With explicit `val_dataset_id`**: Uses that dataset as validation
2. **Without `val_dataset_id`**: Auto-splits the first training dataset (90% train, 10% validation)

The `training_task_id` points to this validation dataset automatically.

### Method 2: Explicit Dataset ID (`dataset_id`)

- Uses any specific dataset from the database
- Useful for evaluating on different datasets than training
- Must be a valid dataset UUID in the system

### Dataset Splitting in Training

**Scenario 1: Multiple Datasets with Validation**
```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"],
  "val_dataset_id": "validation-dataset-uuid"
}
```
→ **Result**: `validation-dataset-uuid` becomes the validation dataset for evaluation

**Scenario 2: Multiple Datasets without Validation**
```json
{
  "train_dataset_ids": ["dataset1-uuid", "dataset2-uuid"]
}
```
→ **Result**: System concatenates all datasets, then splits 90% train / 10% validation

**Scenario 3: Single Dataset without Validation**
```json
{
  "train_dataset_ids": ["single-dataset-uuid"]
}
```
→ **Result**: Auto-split of `single-dataset-uuid` → 90% train / 10% validation

### Dataset Path Examples

**Training creates these paths:**
- Explicit validation: `data/datasets/my_validation_dataset.jsonl`
- Auto-split validation: `data/datasets/my_training_dataset.jsonl#auto-split`

## Usage Examples

### 1. Basic Evaluation with Training Task Integration

```python
from vectorize.evaluation import TrainingEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = TrainingEvaluator("data/models/my-fine-tuned-model")

# Evaluate on dataset
metrics = evaluator.evaluate_dataset(
    dataset_path=Path("data/datasets/eval.jsonl"),
    max_samples=1000
)

print(f"Similarity ratio: {metrics.similarity_ratio:.3f}")
print(f"Spearman correlation: {metrics.spearman_correlation:.3f}")
print(f"Positive similarity: {metrics.avg_positive_similarity:.3f}")
print(f"Negative similarity: {metrics.avg_negative_similarity:.3f}")
```

### 2. Model Comparison with Baseline

```python
# Compare trained model against baseline
comparison = evaluator.compare_models(
    baseline_model_path="sentence-transformers/all-MiniLM-L6-v2",
    dataset_path=Path("data/datasets/eval.jsonl")
)

trained_metrics = comparison["trained"]
baseline_metrics = comparison["baseline"]

# Get improvement metrics
improvement = trained_metrics.get_improvement_over_baseline(baseline_metrics)
print(f"Ratio improvement: {improvement['ratio_improvement']:.3f}")
print(f"Trained ratio: {trained_metrics.similarity_ratio:.3f}")
print(f"Baseline ratio: {baseline_metrics.similarity_ratio:.3f}")
```

### 3. API Usage - Training Task Integration

**Evaluate using training's validation dataset:**

```bash
# Evaluate using training task validation dataset
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "trained_models/my-model-finetuned-20250615-213447-7ef54ba0",
    "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
    "baseline_model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
    "max_samples": 1000
  }'
```

### 4. API Usage - Explicit Dataset

```bash
# Evaluate using explicit dataset
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_tag": "trained_models/my-model-finetuned-20250615-213447-7ef54ba0",
    "dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
    "max_samples": 1000
  }'
```

### 5. Check Evaluation Status

```bash
# Get evaluation status and results
curl -X GET "http://localhost:8000/evaluation/{task_id}/status"
```

## JSON API Reference

### Evaluation Request Options

**Using Training Task ID (Recommended):**
```json
{
  "model_tag": "trained_models/my-model-finetuned-20250615-213447-7ef54ba0",
  "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "max_samples": 1000
}
```

**Using Explicit Dataset:**
```json
{
  "model_tag": "trained_models/my-model-finetuned-20250615-213447-7ef54ba0",
  "dataset_id": "0a9d5e87-e497-4737-9829-2070780d10df",
  "max_samples": 1000
}
```

**With Baseline Comparison:**
```json
{
  "model_tag": "trained_models/my-model-finetuned-20250615-213447-7ef54ba0",
  "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
  "baseline_model_tag": "models--sentence-transformers--all-MiniLM-L6-v2",
  "max_samples": 1000
}
```

### Response Format

**Success Response (202 Accepted):**
```json
{
  "status_code": 202,
  "headers": {
    "Location": "/evaluation/{task_id}/status"
  }
}
```

**Status Response:**
```json
{
  "task_id": "uuid",
  "status": "QUEUED|RUNNING|DONE|FAILED",
  "created_at": "2025-06-15T10:30:00.000Z",
  "end_date": "2025-06-15T11:30:00.000Z",
  "error_msg": null,
  "evaluation_metrics": "{...json...}",
  "baseline_metrics": "{...json...}",
  "evaluation_summary": "Similarity ratio improved by 0.523 (1.245 → 1.768)"
}
```

### Parameter Rules

**Required:**
- `model_tag`: Tag of the model to evaluate (always required)
- **Exactly one of:** `training_task_id` OR `dataset_id` (never both!)

**Optional:**
- `baseline_model_tag`: For comparison evaluation
- `max_samples`: Default is 1000, must be > 0

### Baseline Comparison Metrics

When using `baseline_model_tag`, you get additional improvement metrics:

```json
{
  "ratio_improvement": 0.523,
  "positive_similarity_improvement": 0.156,
  "negative_similarity_improvement": -0.089,
  "correlation_improvement": 0.234
}
```

## Configuration

Key constants are now managed through the central configuration system:

```python
# From vectorize.config import settings
DEFAULT_MAX_SAMPLES = settings.evaluation_default_max_samples  # Default: 1000
DEFAULT_RANDOM_SEED = settings.evaluation_default_random_seed  # Default: 42
```

### Environment Variables

```bash
# Optional: Custom evaluation settings
EVALUATION_MAX_SAMPLES=1000
EVALUATION_BATCH_SIZE=32
EVALUATION_DEVICE=cuda
```

## Error Handling

The module provides comprehensive error handling:

- **`DatasetValidationError`**: Invalid or malformed datasets
- **`ModelNotFoundError`**: Missing model files or model not in database
- **`InvalidDatasetIdError`**: Invalid dataset UUIDs or training task IDs
- **`TrainingDatasetNotFoundError`**: Dataset files not found or training task has no validation dataset
- **`EvaluationTaskNotFoundError`**: Evaluation task not found
- **`ValueError`**: Invalid parameter combinations (both `dataset_id` and `training_task_id` provided)

### Common Error Scenarios

**422 Unprocessable Content:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "model_tag"],
      "msg": "Field required"
    }
  ]
}
```

**404 Model Not Found:**
```json
{
  "detail": "Model not found: trained_models/nonexistent-model"
}
```

**400 Bad Request (Invalid Combination):**
```json
{
  "detail": "Cannot specify both dataset_id and training_task_id. Use dataset_id for explicit dataset or training_task_id to use the validation dataset from that training."
}
```

**404 Training Task Has No Validation Dataset:**
```json
{
  "detail": "Training task 7ef54ba0-2d87-4864-8360-81de8035369a has no validation dataset path"
}
```

## Testing

### Run Evaluation Tests

```bash
# All evaluation tests
pytest tests/evaluation/ -v

# Specific test categories
pytest tests/evaluation/test_evaluation.py -v          # Unit tests
pytest tests/evaluation/test_service_integration.py -v # Integration tests

# With evaluation marker
pytest -m evaluation -v
```

### Test Categories

- **Unit tests**: Individual component testing (`TestEvaluationMetrics`, `TestTrainingEvaluator`)
- **Integration tests**: End-to-end evaluation with database integration (`TestEvaluationIntegration`)
- **Service integration tests**: Training-evaluation integration (`test_resolve_dataset_with_training_task_id`)
- **Validation tests**: Dataset validation edge cases

### Test Data Requirements

For integration tests that require actual models:

```bash
# Optional: Run with actual models (requires model downloads)
pytest tests/evaluation/ -m integration -v
```

## Workflow Integration

### Complete Training → Evaluation Pipeline

1. **Train Model:**
   ```json
   POST /training/train
   {
     "model_tag": "sentence-transformers/all-MiniLM-L6-v2",
     "train_dataset_ids": ["uuid1", "uuid2"],
     "val_dataset_id": "uuid3",
     "epochs": 3
   }
   ```

2. **Get Training Task ID** from response Location header

3. **Evaluate Trained Model:**
   ```json
   POST /evaluation/evaluate
   {
     "model_tag": "trained_models/model-finetuned-20250615-213447-7ef54ba0",
     "training_task_id": "7ef54ba0-2d87-4864-8360-81de8035369a",
     "baseline_model_tag": "sentence-transformers/all-MiniLM-L6-v2"
   }
   ```

4. **Monitor Progress:** `GET /evaluation/{task_id}/status`

5. **Analyze Results:** Review metrics and baseline comparison

## Related Documentation

- **[Training Module](training.md)**: SBERT training pipeline
- **[API Documentation](../diagrams/src/api-endpoints.plantuml)**: Complete API reference
- **[Use Cases](../diagrams/src/use-cases-evaluation.plantuml)**: Evaluation workflow diagrams
- **[Model Upload](upload.md)**: How to upload and manage models
