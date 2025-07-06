# Configuration

The Vectorize application uses a multi-layered configuration system that combines default values, TOML configuration files, and environment variables to provide flexible deployment options across different environments.

## Configuration Sources

The configuration system loads settings from multiple sources:

1. **Environment Variables** (`.env` file or system environment)
2. **TOML Configuration File** (`src/vectorize/config/resources/app.toml`)
3. **Default Values** (hardcoded in the application)

## Environment Variables

Environment variables are the primary method for configuring sensitive information and deployment-specific settings. Copy the example file to get started:

```sh
cp .env.example .env
```

**Important Security Note:** The `.env` file should contain all sensitive configuration values such as database URLs, API keys, and secrets. Never put sensitive information in the `app.toml` file, as it's part of the source code and may be committed to version control.

### Common Environment Variables

```sh
# Application Environment
TZ=Europe/Berlin
ENV=development|testing|production

# Database Configuration
DATABASE_URL=sqlite+aiosqlite:///app.db

# Logging Configuration
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL

# Directory Overrides (useful for containerized deployments)
DATASETS_DIR=/custom/path/datasets
MODELS_DIR=/custom/path/models
DB_DIR=/custom/path/db

# Dramatiq Queue
REDIS_URL=redis://localhost:56379
```

## Configuration File

The main non-sensitive configuration is stored in `src/vectorize/config/resources/app.toml`. This file contains application defaults and structure that are safe to include in version control.

**Security Note:** Only put non-sensitive configuration values in `app.toml`. All secrets, credentials, and environment-specific paths should be configured via environment variables.

### Server Configuration (`[app.server]`)

Controls the FastAPI server behavior and network settings:

- **`host_binding`** (default: `"127.0.0.1"`): Host address the server binds to
  - Set to `"0.0.0.0"` to accept connections from all network interfaces (required for Docker)
  - Set to `"127.0.0.1"` for localhost-only access (development security)
  - Override with `HOST_BINDING` environment variable
- **`port`** (default: `8000`): Network port the server listens on
  - Override with `PORT` environment variable
- **`version`** (default: `"0.1.0"`): Application version string for API documentation
- **`reload`** (default: `true`): Enable auto-reload on code changes
  - Automatically disabled in production environment
  - Useful for development workflows
- **`allow_origin_in_dev`**: CORS allowed origins for cross-origin requests during development
- **`root_path`** (default: `"/v1/api"`): Root path prefix for all API endpoints (e.g., `"/v1/api"` for reverse proxy setups)

### Database Configuration (`[app.db]`)

SQLite database connection and performance settings:

- **`logging`** (default: `false`): Enable SQL query logging for debugging
  - Useful for development and troubleshooting
  - Should be disabled in production for performance
- **`timeout`** (default: `30`): Database operation timeout in seconds
- **`future`** (default: `true`): Enable SQLAlchemy 2.0 future mode for better compatibility
- **`max_overflow`** (default: `10`): Maximum connections beyond base pool size
- **`pool_size`** (default: `5`): Base connection pool size
  - Increase for high-concurrency applications
  - Override with `POOL_SIZE` environment variable
- **`pool_pre_ping`** (default: `true`): Validate connections before use to handle stale connections
- **`pool_recycle`** (default: `300`): Recycle connections after N seconds to prevent timeout issues
- **`pool_timeout`** (default: `30`): Timeout for acquiring connections from the pool
- **`seed_db_on_start`** (default: `true`): Automatically populate the database with test data on application startup
  - Useful for development and testing environments to provide sample datasets, models, and tasks
  - Should be disabled in production to prevent test data from appearing in live systems

**Database URL:** The database location is configured via the `DATABASE_URL` environment variable, not in `app.toml`, for security and deployment flexibility.

### Logging Configuration (`[app.logging]`)

Application logging behavior and file management:

- **`rotation`** (default: `"10 MB"`): Log file rotation trigger
  - Size-based: `"10 MB"`, `"1 GB"`
  - Time-based: `"daily"`, `"weekly"`, `"monthly"`
  - Prevents log files from growing indefinitely
- **`log_dir`** (default: `"log"`): Directory for log files
  - Can be overridden with `LOG_DIR` environment variable
- **`log_file`** (default: `"app.log"`): Log filename within the log directory

**Log Level:** Configured via `LOG_LEVEL` environment variable. Production environments automatically upgrade DEBUG to INFO for security.

### Dataset Configuration (`[app.dataset]`)

Dataset upload, processing, and validation settings:

#### File Handling & Storage

- **`upload_dir`** (default: `"data/datasets"`): Directory for uploaded datasets
  - Override with `DATASETS_DIR` environment variable for containerized deployments
- **`max_upload_size`** (default: `53687091200`): Maximum file size in bytes (50 GB default)
  - Prevents disk space exhaustion from large uploads
  - Can be overridden with `MAX_UPLOAD_SIZE` environment variable
- **`max_filename_length`** (default: `255`): Maximum filename length to ensure filesystem compatibility
- **`max_zip_members`** (default: `10000`): Maximum files in ZIP archives to prevent zip bombs

#### Supported File Formats

- **`allowed_extensions`**: Supported file types for dataset upload
  - `csv`: Comma-separated values with configurable delimiters
  - `json`: Standard JSON format for structured data
  - `jsonl`: JSON Lines format (one JSON object per line)
  - `xml`: XML documents for structured data
  - `xlsx`, `xls`: Microsoft Excel spreadsheets
- **`default_delimiter`** (default: `";"`): Default CSV delimiter
  - Can be customized per upload request
  - Common alternatives: `","`, `"\t"` (tab), `"|"`

#### Hugging Face Schema Validation

- **`hf_allowed_schemas`**: Permitted column combinations for HuggingFace dataset compatibility

**Supported Schema Categories:**

1. **Standard Preference Formats** (for preference learning and RLHF)

   ```toml
   ["prompt", "chosen", "rejected"]
   ["prompt", "chosen_response", "rejected_response"]
   ["prompt", "response_chosen", "response_rejected"]
   ```

2. **Instruction-Based Formats** (for instruction tuning)

   ```toml
   ["instruction", "chosen", "rejected"]
   ["instruction", "output_1", "output_2"]
   ["instruction", "output"]
   ```

3. **Query/Question Formats** (for Q&A and retrieval)

   ```toml
   ["query", "chosen", "rejected"]
   ["question", "positive", "negative"]
   ["question", "answer"]
   ```

4. **Input/Output Formats** (for general supervised learning)

   ```toml
   ["input", "chosen", "rejected"]
   ["input", "response_a", "response_b"]
   ["input", "target"]
   ```

5. **Conversation Formats** (for dialogue systems)
   ```toml
   ["system", "user", "chosen", "rejected"]
   ["human", "assistant"]
   ["messages", "chosen", "rejected"]
   ```

### Model Configuration (`[app.model]`)

AI model storage and upload settings:

- **`upload_dir`** (default: `"data/models"`): Directory for uploaded model files
  - Override with `MODELS_DIR` environment variable
  - Should have sufficient storage for large transformer models
- **`max_upload_size`** (default: `53687091200`): Maximum model size in bytes (50 GB default)
  - Large language models can be several gigabytes
  - Adjust based on available storage and model requirements

### Inference Configuration (`[app.inference]`)

Model inference execution settings:

- **`device`** (default: `"cpu"`): Compute device for inference operations
  - `"cpu"`: Use CPU for inference (slower but more compatible)
  - `"cuda"`: Use GPU with CUDA for faster inference (requires NVIDIA GPU)
  - `"mps"`: Use Apple Metal Performance Shaders (for Apple Silicon Macs)
  - Override with `INFERENCE_DEVICE` environment variable

### Cache Configuration (`[app.cache]`)

Model caching and memory management settings for optimizing inference performance:

- **`strategy`** (default: `"fixed_size"`): Cache eviction strategy for managing model memory
  - `"fixed_size"`: Maintains a fixed number of models in cache (LRU eviction)
  - `"vram_aware"`: Dynamically manages cache based on available VRAM
  - `"usage_based"`: Prioritizes frequently used models for retention
- **`max_models`** (default: `5`): Maximum number of models to keep in memory simultaneously
  - Higher values improve performance but consume more memory
  - Should be adjusted based on available system memory and model sizes
  - Override with `CACHE_MAX_MODELS` environment variable
- **`vram_safety_margin_gb`** (default: `1.0`): VRAM safety margin in gigabytes
  - Reserved memory to prevent out-of-memory errors during inference
  - Only applies when using GPU devices (`cuda`, `mps`)
  - Larger margins provide more stability but reduce cache capacity
  - Override with `VRAM_SAFETY_MARGIN` environment variable

### Evaluation Configuration (`[app.evaluation]`)

Model evaluation and testing settings:

- **`default_max_samples`** (default: `1000`): Default sample size for evaluation runs
  - Limits evaluation time and computational requirements
  - Can be overridden per evaluation request
- **`default_random_seed`** (default: `42`): Random seed for reproducible evaluation results
  - Ensures consistent results across evaluation runs
  - Important for scientific reproducibility

## Configuration Loading Process

The application uses Pydantic Settings to automatically load and validate configuration:

1. **Defaults**: Built-in default values provide a working baseline
2. **TOML Loading**: `app.toml` values override defaults
3. **Environment Loading**: Environment variables override TOML values
4. **Validation**: All settings are validated for type and constraints
5. **Computed Fields**: Some settings are dynamically computed based on environment

## Environment-Specific Behavior

The configuration system automatically adjusts based on the `ENV` environment variable:

- **`development`**: Enables debugging features, auto-reload, and relaxed security
- **`testing`**: Uses isolated test directories and databases
- **`production`**: Enforces security settings, disables debugging, and optimizes performance

## Deployment Examples

### Development Setup

```sh
ENV=development
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite+aiosqlite:///dev.db
REDIS_URL=redis://localhost:56379
```

### Production

```sh
ENV=production
LOG_LEVEL=INFO
DATABASE_URL=sqlite+aiosqlite:///app/db/app.db
DATASETS_DIR=/app/data/datasets
MODELS_DIR=/app/data/models
REDIS_URL=redis://redis:6379
```

## Validation & Error Handling

The configuration system performs comprehensive validation on startup:

- **Type Validation**: Ensures all values match expected types
- **Constraint Validation**: Checks ranges, lengths, and format requirements
- **Path Validation**: Verifies directory paths are accessible
- **Environment Validation**: Applies environment-specific rules (e.g., production security)

Invalid configurations will cause the application to fail immediately with descriptive error messages, preventing runtime issues.
