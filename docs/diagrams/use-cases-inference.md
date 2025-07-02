# Inference Use Cases

This diagram illustrates the text embedding inference capabilities in the Vectorize system.

![file](out/use-cases-inference.svg)

## Use Case Overview

### Embedding Generation

The inference system provides a comprehensive embedding API that handles various input formats and processing requirements:

- **Single Text Embedding**: Generate embeddings for individual text strings
- **Batch Processing**: Process multiple texts efficiently in single requests
- **Token Input Support**: Handle pre-tokenized input arrays for advanced use cases
- **Dimension Control**: Generate embeddings with custom dimensions for optimization
- **Format Flexibility**: Support for different encoding formats and response structures
- **Multi-Model Support**: Switch between different embedding models dynamically

## Supported Input Types

- **Text strings**: Standard text input for general use cases
- **Text arrays**: Batch processing of multiple texts
- **Token arrays**: Pre-tokenized input for specialized applications
- **Integer tokens**: Single dimension token arrays

## Key Features

### Model Management

- Intelligent model caching with VRAM-aware strategies
- Automatic model loading from uploaded and trained models
- Background model preloading based on usage statistics
- Usage tracking and cache optimization

### Performance Optimization

- **GPU acceleration**: CUDA support with memory management
- **Smart caching**: LRU and usage-based eviction strategies
- **Batch processing**: Efficient handling of multiple inputs
- **Memory management**: VRAM monitoring and automatic optimization

### Cache Operations

- Real-time cache status monitoring with preload candidates
- Manual cache clearing and management
- Daily usage statistics and performance metrics
- Automatic cache optimization based on usage patterns

## Integration with Other Modules

The inference system seamlessly integrates with:

- **Model Upload**: Access to uploaded models from various sources
- **Training Module**: Use trained models for embedding generation
- **Evaluation Module**: Generate embeddings for evaluation datasets
- **Task Management**: Daily usage statistics and inference tracking

This unified approach provides fast, reliable embedding generation with intelligent resource management across the entire Vectorize platform.
