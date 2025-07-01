# Inference Guide

This guide covers text embedding inference for uploaded models, including core API usage, caching strategies, and optimization features.

## Overview

The Inference module provides **OpenAI-compatible embedding generation** with intelligent model caching and GPU acceleration:

- **Multi-format inputs**: Text, tokens, batch processing
- **Smart caching**: VRAM-aware model management
- **GPU acceleration**: CUDA support with memory optimization
- **Background preloading**: Automatic model warming

## Core Features

### Basic Embedding Generation

```bash
# Single text embedding
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Text to embed"
  }'
```

### Batch Processing

```bash
# Multiple texts in one request
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["First text", "Second text", "Third text"]
  }'
```

### Token Input Support

```bash
# Pre-tokenized input
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": [101, 7592, 2088, 102]
  }'
```

## Architecture

### Model Sources

Models are loaded from multiple sources in priority order:

1. **Uploaded models**: Files via `/models/upload`
2. **Trained models**: Outputs from training jobs
3. **GitHub models**: Repositories via `/models/github`
4. **HuggingFace Hub**: Public model repositories

### Caching System

Two caching strategies optimize memory usage:

**Fixed-Size Cache**:
- Maintains fixed number of models in memory
- Simple LRU eviction policy

**VRAM-Aware Cache**:
- Dynamic sizing based on GPU memory
- Smart eviction considering usage patterns

## Configuration

### Environment Variables

```bash
# Inference configuration
INFERENCE_DEVICE=cuda        # or 'cpu'
CACHE_STRATEGY=vram_aware    # or 'fixed_size'
CACHE_MAX_MODELS=5           # for fixed_size strategy
CACHE_VRAM_SAFETY_MARGIN_GB=1.0  # for vram_aware
```

### Model Preloading

Configure automatic model warming:

```bash
# Preload frequently used models
curl -X POST "http://localhost:8000/embeddings/preload" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["sentence-transformers/all-MiniLM-L6-v2"],
    "max_models": 3
  }'
```

## API Reference

### Embedding Request

```python
{
    "model": str,                    # Model identifier
    "input": str | list[str] | list[int],  # Text or tokens
    "encoding_format": "float",      # Optional: output format
    "dimensions": int                # Optional: reduce dimensions
}
```

### Response Format

```python
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, 0.2, ...],  # Vector values
            "index": 0                      # Input index
        }
    ],
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10
    }
}
```

### Cache Management

```bash
# Cache status
GET /embeddings/cache/status

# Clear cache
POST /embeddings/cache/clear

# Usage statistics
GET /embeddings/counter/{model_tag}
```

## Error Handling

### Common Errors

**Model Not Found**:
```json
{
  "detail": "Model not found: nonexistent-model",
  "code": "NOT_FOUND"
}
```

**Invalid Input**:
```json
{
  "detail": "Field required: model",
  "code": "VALIDATION_ERROR"
}
```

**Out of Memory**:
```json
{
  "detail": "Model loading failed: CUDA out of memory",
  "code": "MEMORY_ERROR"
}
```

### Troubleshooting

**Performance Issues**:
- Enable GPU: `INFERENCE_DEVICE=cuda`
- Use batch processing for multiple texts
- Preload frequently used models

**Memory Issues**:
- Reduce cache size: `CACHE_MAX_MODELS=3`
- Increase safety margin: `CACHE_VRAM_SAFETY_MARGIN_GB=2.0`
- Switch to CPU inference

## Usage Examples

### Python Client

```python
import requests

# Single embedding
response = requests.post("http://localhost:8000/embeddings", json={
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Sample text"
})
embedding = response.json()["data"][0]["embedding"]

# Batch processing
response = requests.post("http://localhost:8000/embeddings", json={
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["Text 1", "Text 2", "Text 3"]
})
embeddings = [item["embedding"] for item in response.json()["data"]]
```

### Similarity Search

```python
import numpy as np

# Generate query embedding
query_response = requests.post("http://localhost:8000/embeddings", json={
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "search query"
})
query_embedding = query_response.json()["data"][0]["embedding"]

# Calculate similarity scores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare with document embeddings
for doc_embedding in document_embeddings:
    score = cosine_similarity(query_embedding, doc_embedding)
    print(f"Similarity: {score:.3f}")
```

## Best Practices

### Production Deployment

1. **Use GPU acceleration** for better performance
2. **Configure VRAM-aware caching** for optimal memory usage
3. **Preload frequently used models** to reduce latency
4. **Monitor cache metrics** to optimize configuration

### Development

1. **Test with various input formats** (text, tokens, batches)
2. **Implement retry logic** for transient failures
3. **Use batch processing** for multiple texts
4. **Monitor memory usage** patterns

## Related Documentation

- **[Model Upload](upload.md)**: Upload and manage models
- **[Training](training.md)**: Custom model training
- **[Evaluation](evaluation.md)**: Model performance assessment
- **[Tasks](tasks.md)**: Background task management
