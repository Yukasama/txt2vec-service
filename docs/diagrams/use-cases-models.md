# AI Model Use Cases

This diagram illustrates the various use cases for AI model management in the Vectorize system.

![file](out/use-cases-models.svg)

## Use Case Categories

### Model Upload

- **Upload from Hugging Face Hub**: Direct integration with HuggingFace model repository using model tags and revisions
- **Upload from GitHub Repository**: Load models from public or private GitHub repositories with branch and commit support
- **Upload Local ZIP Archive**: Upload compressed model bundles from local filesystem with automatic extraction and validation

### Model Management

- **List Models with Pagination**: Browse all registered models with configurable page sizes (5-100 items) and efficient pagination
- **Get Model Details with ETag Support**: Retrieve comprehensive model information with conditional requests using ETags for caching
- **Update Model with Version Control**: Modify model metadata with optimistic locking using If-Match headers and version tracking
- **Delete Model and Files**: Remove models from database and filesystem with automatic cleanup of associated files

### Background Processing

- **Monitor Upload Task Status**: Track the progress of asynchronous upload operations with real-time status updates
- **Validate Model Files and Format**: Verify model file integrity, format compatibility, and required files (config.json, model weights)
- **Cache Model Locally**: Store downloaded models in local filesystem with optimized directory structure

## Supported Model Sources

### HuggingFace Hub Integration

- **Tag-Based Loading**: Load models using HuggingFace model identifiers (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **Revision Support**: Specify specific model versions, branches, or commits for reproducible deployments
- **Automatic Validation**: Verify model existence and accessibility before initiating download
- **Metadata Extraction**: Automatic extraction of model information and configuration

### GitHub Repository Support

- **URL-Based Access**: Load models from GitHub repositories using repository URLs
- **Branch and Commit Selection**: Access models from specific branches, tags, or commit hashes
- **Private Repository Support**: Handle private repositories with proper authentication
- **Repository Structure Parsing**: Automatic detection and extraction of model files from repository structure

### Local File Upload

- **ZIP Archive Processing**: Handle compressed model bundles with automatic extraction
- **Multiple Model Support**: Process ZIP archives containing multiple models with batch operations
- **File Validation**: Verify model file formats and required components before storage
- **Size Limitations**: Configurable upload size limits (default: 50GB) with validation

## Model File Requirements

### Supported Formats

- **PyTorch Models**: `.pt`, `.pth`, `.bin` files with state dictionaries
- **SafeTensors**: `.safetensors` files for secure model weight storage
- **Configuration**: `config.json` files for model architecture and parameters
- **Tokenizers**: Tokenizer configuration and vocabulary files
- **Sentence Transformers**: Complete sentence transformer model packages

### Validation Pipeline

- Model file integrity checks and format validation
- Required file verification (config.json, model weights)
- Architecture compatibility assessment
- File system path safety validation
- Storage space and size limit enforcement

## Key Features

### Version Control & Concurrency

- **Optimistic Locking**: ETag-based version control for concurrent access protection
- **Version Tracking**: Automatic version incrementation for model updates
- **Conditional Requests**: Support for If-Match and If-None-Match headers
- **Conflict Resolution**: Handle concurrent modification attempts gracefully

### Caching & Performance

- **Local Storage**: Efficient model caching in structured directory hierarchy
- **Deduplication**: Prevent duplicate model storage with tag-based identification
- **Memory Management**: Configurable model cache with VRAM optimization
- **Background Processing**: Asynchronous upload and processing to maintain API responsiveness

### Model Discovery & Metadata

- **Pagination Support**: Efficient browsing of large model collections
- **Search and Filtering**: Filter models by source, name, and other attributes
- **Usage Tracking**: Monitor model inference counts and usage patterns
- **Relationship Tracking**: Track model lineage and training relationships

## Workflow Integration

AI model management integrates seamlessly with:

- **Training Pipeline**: Provides base models for fine-tuning and custom training workflows
- **Evaluation System**: Supplies models for performance assessment and benchmarking
- **Inference Service**: Manages model availability for embedding generation and inference
- **Upload Task System**: Coordinates with unified task monitoring for upload progress tracking
- **Background Processing**: Asynchronous model download and validation with status reporting

## Error Handling & Validation

The system provides comprehensive error handling for:

- **Model Availability**: HuggingFace and GitHub repository accessibility validation
- **File Format Issues**: Unsupported formats and corrupted model files
- **Storage Limitations**: File size limits and disk space constraints
- **Concurrent Access**: Version conflicts and concurrent modification attempts
- **Authentication**: GitHub private repository and HuggingFace authentication failures
- **Network Issues**: Download failures and connectivity problems

This comprehensive model management system ensures reliable handling of AI models from multiple sources while maintaining data integrity and providing efficient access for downstream workflows.
