# Dataset Guide

The Dataset module makes uploading, processing, and managing training data simple. It supports multiple formats, validates schemas, and integrates directly with HuggingFace datasets.

## What Can It Do?

- **Multi-Format Upload**: CSV, JSON, JSONL, XML, Excel, ZIP
- **HuggingFace Integration**: Download & process datasets automatically
- **Format Conversion**: Standardizes uploads to JSONL
- **Schema Validation**: Ensures correct columns
- **Batch Uploads**: Handle multiple files/archives
- **CRUD Operations**: Create, read, update, delete datasets

---


## Local File Upload

### Supported Formats

- **CSV**, **JSON**, **JSONL**
- **XML**, **Excel** (.xlsx, .xls)
- **ZIP** archives with these formats

See [Dataset Configuration](../configuration.md#dataset-configuration-appdataset) for details.

---

### Upload Examples

#### Single File

```bash
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@my_dataset.csv"
```

```python
with open("dataset.json", "rb") as f:
    response = await client.post(
        "http://localhost:8000/datasets",
        files={"files": ("dataset.json", f, "application/json")}
    )
print(f"Uploaded: {response.headers['Location']}")
```

#### Multiple Files

```bash
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@dataset1.csv" \
  -F "files=@dataset2.json"
```

```python
files = [
    ("files", ("train.csv", open("train.csv", "rb"), "text/csv")),
    ("files", ("val.json", open("val.json", "rb"), "application/json")),
]
await client.post("http://localhost:8000/datasets", files=files)
```

#### ZIP Archive

```bash
curl -X POST "http://localhost:8000/datasets" \
  -F "files=@datasets.zip"
```

```python
with open("datasets.zip", "rb") as f:
    await client.post(
        "http://localhost:8000/datasets",
        files={"files": ("datasets.zip", f, "application/zip")}
    )
```

---

### Upload Options

| Parameter       | Description               | Default      |
| --------------- | ------------------------- | ------------ |
| `question_name` | Column name for questions | `"question"` |
| `positive_name` | Column name for positives | `"positive"` |
| `negative_name` | Column name for negatives | `"negative"` |
| `sheet_index`   | Excel sheet index         | `0`          |

---

### Upload Pipeline

1. **Validation**: Format, size, required columns
2. **Parsing**: Load as DataFrame
3. **Schema Mapping**: Rename to standard columns
4. **Conversion**: Save as JSONL
5. **Database**: Create dataset record

---


## HuggingFace Dataset Integration

Download and process datasets directly from HuggingFace. Columns are mapped automatically, datasets split into train/test, and results stored in your system.

### Supported Schemas

Examples:

- `["prompt", "chosen", "rejected"]` – preference learning
- `["instruction", "output_1", "output_2"]` – instruction tuning
- `["question", "positive", "negative"]` – Q&A

See [HuggingFace Schema Validation](../configuration.md#hugging-face-schema-validation) for full list.

---

### Column Mapping

Columns are standardized automatically:

- **Questions**: `prompt`, `anchor`, `query` → `question`
- **Positives**: `chosen`, `answer` → `positive`
- **Negatives**: `rejected`, `random` → `negative`

Mappings are configurable in [Dataset Configuration](../configuration.md#dataset-configuration-appdataset).

---

### Example HuggingFace Upload

```bash
curl -X POST "http://localhost:8000/datasets/huggingface" \
  -H "Content-Type: application/json" \
  -d '{"dataset_tag": "squad"}'
```

```python
response = await client.post(
    "http://localhost:8000/datasets/huggingface",
    json={"dataset_tag": "Anthropic/hh-rlhf"}
)
task_id = response.headers["Location"].split("/")[-1]
print(f"Upload task ID: {task_id}")
```

---

### Monitor Progress

```python
async def wait_for_upload(task_id: str):
    while True:
        response = await client.get(f"/datasets/huggingface/status/{task_id}")
        status = response.json()["task_status"]

        if status == "D":
            print("Upload completed!")
            break
        elif status == "F":
            print(f"Upload failed: {response.json().get('error_msg', 'Unknown error')}")
            break

        print("🔄 Upload in progress...")
        await asyncio.sleep(10)
```

---


## Dataset Management Operations

### List Datasets

```bash
curl "http://localhost:8000/datasets?limit=20&offset=0"
```

```python
response = await client.get("http://localhost:8000/datasets", params={"limit": 50, "offset": 0})
datasets = response.json()
print(f"Found {datasets['total']} datasets")
```

### Get Dataset Details

```bash
curl "http://localhost:8000/datasets/{dataset_id}"
```

```python
response = await client.get(f"http://localhost:8000/datasets/{dataset_id}")
dataset = response.json()
print(f"Dataset: {dataset['name']}, Rows: {dataset['rows']}")
```

### Update Dataset

```bash
curl -X PUT "http://localhost:8000/datasets/{dataset_id}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Dataset Name"}'
```

```python
response = await client.put(
    f"http://localhost:8000/datasets/{dataset_id}",
    json={"name": "Updated Dataset"}
)
```

### Delete Dataset

```bash
curl -X DELETE "http://localhost:8000/datasets/{dataset_id}"
```

```python
response = await client.delete(f"http://localhost:8000/datasets/{dataset_id}")
print("Dataset deleted")
```

---


## Common Errors

| Error                    | Cause                     | Solution                                 |
| ------------------------ | ------------------------- | ---------------------------------------- |
| `UnsupportedFormatError` | File format not supported | Use CSV, JSON, JSONL, XML, Excel, or ZIP |
| `EmptyFileError`         | File is empty             | Upload files with data                   |
| `InvalidCSVFormatError`  | Missing required columns  | Include question, positive, negative     |
| `FileTooLargeError`      | File exceeds size limit   | Split large files or adjust limits       |
| `DatasetNotFoundError`   | HF dataset doesn’t exist  | Check the dataset tag on HF Hub          |

---

This module makes dataset uploads and HuggingFace integration seamless – get started quickly and keep your data standardized!
