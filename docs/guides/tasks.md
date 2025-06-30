# Tasks Guide

The Tasks module provides centralized monitoring and management for all background operations in Vectorize. This unified interface allows you to track the status of uploads, training, evaluation, synthesis, and other asynchronous processes.


## What the Tasks Endpoint Can Do

The `/tasks` endpoint serves as your central command center for monitoring all background operations:

- **Unified Task View**: Monitor all background tasks from different modules in one place
- **Advanced Filtering**: Filter tasks by type, status, time windows, and custom tags
- **Real-time Status**: Get up-to-date information on running, completed, and failed tasks
- **Pagination Support**: Handle large numbers of tasks efficiently with limit/offset pagination

### Supported Task Types

- **`model_upload`**: Model uploads from HuggingFace, GitHub, or local files
- **`dataset_upload`**: Dataset uploads and processing operations
- **`training`**: Model training and fine-tuning processes
- **`evaluation`**: Model evaluation and benchmarking tasks
- **`synthesis`**: Synthetic data generation operations


## Available Filters

### Core Filters

| Filter         | Type    | Description                               | Example Values   |
| -------------- | ------- | ----------------------------------------- | ---------------- |
| `limit`        | integer | Maximum number of tasks to return (1-100) | `20`, `50`       |
| `offset`       | integer | Number of tasks to skip for pagination    | `0`, `20`        |
| `within_hours` | integer | Time window for filtering tasks (≥1)      | `1`, `24`, `168` |

### Content Filters

| Filter      | Type   | Description                   | Example Values                          |
| ----------- | ------ | ----------------------------- | --------------------------------------- |
| `task_type` | list   | Filter by specific task types | `model_upload`, `training`              |
| `status`    | list   | Filter by task status         | `R` (running), `D` (done), `F` (failed) |
| `tag`       | string | Filter by custom tag          | `my-model`, `experiment-1`              |

### Status Codes

| Code | Status    | Description                      |
| ---- | --------- | -------------------------------- |
| `Q`  | QUEUED    | Task is waiting to be processed  |
| `R`  | RUNNING   | Task is currently being executed |
| `D`  | DONE      | Task completed successfully      |
| `F`  | FAILED    | Task failed with an error        |
| `C`  | CANCELLED | Task was manually cancelled      |


## Usage Examples

### Basic Usage

Get all recent tasks (default: last hour):

```bash
curl "http://localhost:8000/tasks"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8000/tasks")
    tasks = response.json()
    print(f"Found {len(tasks)} recent tasks")
```

### Filter by Task Type

Get only training and evaluation tasks:

```bash
curl "http://localhost:8000/tasks?task_type=training&task_type=evaluation"
```

```python
response = await client.get(
    "http://localhost:8000/tasks",
    params={
        "task_type": ["training", "evaluation"],
        "within_hours": 6
    }
)
```

### Filter by Status

Get only running tasks:

```bash
curl "http://localhost:8000/tasks?status=R"
```

Check for failures in the last 24 hours:

```python
response = await client.get(
    "http://localhost:8000/tasks",
    params={
        "status": ["F"],
        "within_hours": 24
    }
)

failed_tasks = response.json()
if failed_tasks:
    print(f"Found {len(failed_tasks)} failed tasks!")
```

### Combined Filters

Complex filtering with multiple criteria:

```bash
curl "http://localhost:8000/tasks?task_type=training&status=R&tag=experiment-1&limit=5"
```

```python
response = await client.get(
    "http://localhost:8000/tasks",
    params={
        "task_type": ["training", "evaluation"],
        "status": ["Q", "R"],  # Queued or running
        "tag": "production-model-v2",
        "within_hours": 4,
        "limit": 10
    }
)

active_tasks = response.json()
for task in active_tasks:
    print(f"{task['task_type']}: {task['task_status']} ({task['tag']})")
```

### Pagination

```bash
# Get first 10 tasks
curl "http://localhost:8000/tasks?limit=10&offset=0"

# Get next 10 tasks
curl "http://localhost:8000/tasks?limit=10&offset=10"
```


## Response Structure

Each task in the response includes:

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "task_type": "training",
  "task_status": "R",
  "tag": "my-experiment",
  "created_at": "2024-01-15T10:30:00Z",
  "end_date": null
}
```

### Response Fields

| Field         | Type     | Description                                    |
| ------------- | -------- | ---------------------------------------------- |
| `id`          | UUID     | Unique task identifier                         |
| `task_type`   | string   | Type of background task                        |
| `task_status` | string   | Current status code                            |
| `tag`         | string   | Custom tag (may be null)                       |
| `created_at`  | datetime | When the task was created                      |
| `end_date`    | datetime | When the task finished (null if still running) |


## How It Works Behind the Scenes

The tasks endpoint uses a query building system that aggregates data from multiple database tables:

1. **Task Type Selection**: Based on filters, the system determines which tables to query:

   - `model_upload` → `UploadTask` table
   - `training` → `TrainingTask` table
   - `evaluation` → `EvaluationTask` table
   - `synthesis` → `SynthesisTask` table
   - `dataset_upload` → `UploadDatasetTask` table

2. **Query Construction**: Separate SQL queries are built with filtering conditions for status, time windows, and tags

3. **Query Combination**: Individual queries are combined using SQL `UNION ALL` to create a unified result set

4. **Sorting and Pagination**: Results are ordered by creation date (newest first) with pagination applied at the database level


## Best Practices

- **Use appropriate time windows**: Don't fetch all historical tasks
- **Apply status filters**: Focus on relevant task states (Q, R for active tasks)
- **Set reasonable limits**: Use pagination for large result sets
- **Handle errors gracefully**: Check response status codes
- **Cache results briefly**: Avoid repeated identical requests

---

The Tasks endpoint provides powerful monitoring capabilities for all background operations in Vectorize. Use it to build monitoring systems and track your text embedding workflows.
