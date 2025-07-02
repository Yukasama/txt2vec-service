# Synthesis Guide

This guide covers the synthetic data generation system in Vectorize, including media file processing, text extraction, and dataset creation.

## Overview

The Synthesis module provides **synthetic dataset generation** from various media formats and existing datasets:

- **Multi-Format Media Processing**: PDF, image, and dataset file support
- **Text Extraction**: OCR and document parsing capabilities
- **Background Processing**: Asynchronous task execution with status tracking
- **Dataset Integration**: Seamless integration with the dataset management system
- **Quality Control**: Validation and error handling for generated data

## Architecture

### Core Components

1. **`router.py`**: FastAPI endpoints
   - `/synthesis/media` - Media upload and processing
   - `/synthesis/tasks/{task_id}` - Task status monitoring
   - `/synthesis` - List synthesis tasks

2. **`tasks.py`**: Background processing engine
   - Media file processing and text extraction
   - Dataset enhancement and augmentation
   - Error handling and retry logic

3. **`text_extractor.py`**: Media processing utilities
   - PDF text extraction and OCR capabilities
   - Dataset parsing and validation

4. **`models.py`**: Data models and schemas
   - SynthesisTask for tracking processing status
   - Integration with task management system

## Supported File Formats

### Media Files
- **PDF**: Text extraction and document parsing
- **Images**: OCR text extraction (PNG, JPG, JPEG)

### Dataset Enhancement
- Existing datasets in the system for augmentation
- JSONL and CSV format support

## API Reference

### Media Upload

**Endpoint**: `POST /synthesis/media`

**Form Data Fields:**
- `files`: Media files to process (PDF, PNG, JPG, JPEG)
- `dataset_id`: UUID of existing dataset to enhance (optional)

**Request Examples:**

```bash
# Upload media files
curl -X POST "http://localhost:8000/synthesis/media" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "files=@image.png"

# Enhance existing dataset
curl -X POST "http://localhost:8000/synthesis/media" \
  -H "Content-Type: multipart/form-data" \
  -F "dataset_id=0a9d5e87-e497-4737-9829-2070780d10df"
```

**Success Response (202 Accepted):**
```json
{
  "message": "Media files upload accepted, processing in background.",
  "task_id": "12345678-1234-5678-9012-123456789abc",
  "status_url": "/synthesis/tasks/12345678-1234-5678-9012-123456789abc",
  "file_count": 2
}
```

### Task Status Monitoring

**Endpoint**: `GET /synthesis/tasks/{task_id}`

**Status Response:**
```json
{
  "id": "12345678-1234-5678-9012-123456789abc",
  "task_status": "RUNNING",
  "created_at": "2025-07-01T10:30:00.000Z",
  "updated_at": "2025-07-01T10:35:00.000Z",
  "end_date": null,
  "error_msg": null,
  "generated_dataset": {
    "id": "dataset-uuid",
    "name": "Synthesized Dataset",
    "file_path": "data/datasets/synthesized_data.jsonl"
  }
}
```

### List Tasks

**Endpoint**: `GET /synthesis`

```bash
curl -X GET "http://localhost:8000/synthesis?limit=20"
```

## Usage Examples

### Basic Media Processing

```python
import requests
import time

# Upload media files
files = {'files': [('files', open('document.pdf', 'rb'))]}
response = requests.post("http://localhost:8000/synthesis/media", files=files)

task_info = response.json()
task_id = task_info["task_id"]

# Monitor status
status_url = f"http://localhost:8000/synthesis/tasks/{task_id}"
while True:
    status_response = requests.get(status_url)
    status = status_response.json()
    
    if status['task_status'] in ['DONE', 'FAILED']:
        break
    time.sleep(5)

if status['task_status'] == 'DONE':
    dataset_id = status['generated_dataset']['id']
    print(f"Dataset created: {dataset_id}")
```

### Dataset Enhancement

```python
# Enhance existing dataset
response = requests.post(
    "http://localhost:8000/synthesis/media",
    data={'dataset_id': '0a9d5e87-e497-4737-9829-2070780d10df'}
)
```

## Output Format

Generated datasets are saved in JSONL format with the following structure:

```json
{"question":"What is discussed in this document?","positive":"Content extracted from the document.","negative":"Unrelated content."}
{"question":"What patterns can be found?","positive":"Specific patterns identified in the text.","negative":"Random unrelated information."}
```

**Storage Location:**
- Production: `data/datasets/`
- Test: `test_data/datasets/`
- Format: `{source_name}_{unique_id}.jsonl`

## Error Handling

### Common Errors

**Invalid File Format (422):**
```json
{
  "detail": "Unsupported file type: docx. Supported: {pdf, png, jpg, jpeg}"
}
```

**Missing Required Data (422):**
```json
{
  "detail": "Either files or existing dataset id must be provided."
}
```

**Processing Failed (500):**
```json
{
  "detail": "Text extraction failed: OCR processing error"
}
```

## Integration with Training

Use synthesized datasets directly in training workflows:

```python
# 1. Create synthetic dataset
synthesis_response = requests.post(
    "http://localhost:8000/synthesis/media",
    files={'files': [('files', open('docs.pdf', 'rb'))]}
)

# 2. Monitor completion
task_id = synthesis_response.json()["task_id"]
# ... wait for completion ...

# 3. Use in training
dataset_id = final_status['generated_dataset']['id']
training_response = requests.post(
    "http://localhost:8000/training/train",
    json={
        "model_tag": "sentence-transformers/all-MiniLM-L6-v2",
        "train_dataset_ids": [dataset_id],
        "epochs": 3
    }
)
```

## Best Practices

### File Preparation
- Use high-resolution images for better OCR results
- Ensure PDF files are text-searchable when possible
- Group related documents in batch uploads

### Performance Optimization
- Use batch processing for multiple files
- Monitor task status to avoid polling too frequently
- Implement proper error handling and retry logic

### Quality Control
- Validate generated datasets before training
- Review extracted text quality
- Implement data validation in your pipeline

## Related Documentation

- **[Dataset Management](datasets.md)**: Dataset upload and management
- **[Training Module](training.md)**: Using synthetic data for training
- **[Background Tasks](tasks.md)**: Task monitoring and management
- **[API Documentation](../api/synthesis.md)**: Complete API reference