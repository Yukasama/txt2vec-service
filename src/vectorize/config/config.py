"""Define configuration for the project."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings"]


_app_config_path = Path(__file__).parent / "resources" / "app.toml"

with Path.open(_app_config_path, "rb") as f:
    _config = tomllib.load(f)
    _app_config = _config.get("app", {})
    _server_config = _app_config.get("server", {})
    _db_config = _app_config.get("db", {})
    _log_config = _app_config.get("logging", {})
    _model_config = _app_config.get("model", {})
    _dataset_config = _app_config.get("dataset", {})
    _inference_config = _app_config.get("inference", {})
    _cache_config = _app_config.get("cache", {})
    _evaluation_config = _app_config.get("evaluation", {})


class Settings(BaseSettings):
    """Application configuration settings."""

    # Environment configuration
    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Current application environment determining behavior.",
        validation_alias="ENV",
    )

    # Server configuration
    host_binding: str = Field(
        default=_server_config.get("host_binding", "127.0.0.1"),
        description="Host address the server binds to.",
    )

    port: int = Field(
        default=_server_config.get("port", 8000),
        description="Network port the server listens on.",
    )

    version: str = Field(
        default=_server_config.get("version", "0.1.0"),
        description="Version of the application.",
    )

    allow_origin_in_dev: list[str] = Field(
        default=_server_config.get("allow_origin_in_dev", ["http://localhost:3000"]),
        description="List of CORS allowed origins for cross-origin requests.",
    )

    root_path: str = Field(
        default=_server_config.get("root_path", "/v1/api"),
        description="Root path for the application, useful for reverse proxies.",
    )

    # Database configuration
    db_url: str = Field(
        default="sqlite+aiosqlite:///app.db",
        description="Database connection URL.",
        validation_alias="DATABASE_URL",
    )

    db_logging: bool = Field(
        default=_db_config.get("logging", False),
        description="Whether to enable SQL query logging.",
    )

    db_future: bool = Field(
        default=_db_config.get("future", True),
        description="Whether to use future SQLAlchemy features.",
    )

    db_timeout: int = Field(
        default=_db_config.get("timeout", 30),
        description="Timeout for database operations in seconds.",
    )

    db_pool_size: int = Field(
        default=_db_config.get("pool_size", 5),
        description="Size of the database connection pool.",
    )

    db_max_overflow: int = Field(
        default=_db_config.get("max_overflow", 10),
        description="Maximum number of connections to create beyond the pool size.",
    )

    db_pool_timeout: int = Field(
        default=_db_config.get("pool_timeout", 30),
        description="Timeout for acquiring a connection from the pool.",
    )

    db_pool_recycle: int = Field(
        default=_db_config.get("pool_recycle", 300),
        description="Time in seconds to recycle a connection.",
    )

    db_pool_pre_ping: bool = Field(
        default=_db_config.get("pool_pre_ping", True),
        description="Whether to check if a connection is alive before using it.",
    )

    seed_db_on_start: bool = Field(
        default=_db_config.get("seed_db_on_start", False),
        description="Whether to seed the db with initial data on application start.",
    )

    clear_db_on_restart: bool = Field(
        default=True,
        validation_alias="CLEAR_DB_ON_RESTART",
        description="Whether to clear the database on application restart.",
    )

    # Log configuration
    log_dir: str = Field(
        default=_log_config.get("log_dir"),
        description="Directory for storing log files.",
    )

    log_file: str = Field(
        default=_log_config.get("log_file"), description="Name of the log file."
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level (e.g., DEBUG, INFO, WARNING, ERROR).",
    )

    log_rotation: str = Field(
        default=_log_config.get("rotation"),
        description="Log rotation strategy (time or size-based).",
    )

    # Dataset configuration
    dataset_upload_dir_config: Path = Field(
        default=Path(_dataset_config.get("upload_dir")),
        description="Base directory for storing uploaded dataset files.",
        exclude=True,
    )

    dataset_allowed_extensions: frozenset[str] = Field(
        default=frozenset(_dataset_config.get("allowed_extensions", [])),
        description="File extensions permitted for dataset uploads.",
    )

    dataset_base_columns: list[str] = Field(
        default=_dataset_config.get(
            "base_columns", ["question", "positive", "negative"]
        ),
        description="Base columns required for dataset classification.",
    )

    dataset_question_columns: list[str] = Field(
        default=_dataset_config.get(
            "question_columns", ["anchor", "q", "query", "prompt"]
        ),
        description="Columns that can be used for the question field in datasets.",
    )

    dataset_positive_columns: list[str] = Field(
        default=_dataset_config.get("positive_columns", ["answer", "chosen"]),
        description="Columns that can be used for the positive field in datasets.",
    )

    dataset_negative_columns: list[str] = Field(
        default=_dataset_config.get(
            "negative_columns", ["random", "rejected", "no_context"]
        ),
        description="Columns that can be used for the negative field in datasets.",
    )

    dataset_max_filename_length: int = Field(
        default=_dataset_config.get("max_filename_length"),
        description="Maximum allowed length for uploaded filenames.",
    )

    dataset_default_delimiter: str = Field(
        default=_dataset_config.get("default_delimiter"),
        description="Default delimiter used for CSV processing.",
    )

    dataset_max_zip_members: int = Field(
        default=_dataset_config.get("max_zip_members"),
        description="Maximum number of files allowed in a zip archive for datasets.",
    )

    dataset_max_upload_size: int = Field(
        default=_dataset_config.get("max_upload_size"),
        description="Maximum allowed file size for dataset uploads in bytes.",
    )

    dataset_hf_allowed_schemas: list[list[str]] = Field(
        default=_dataset_config.get("hf_allowed_schemas"),
        description="List of allowed schema field combinations for dataset validation.",
    )

    dataset_hf_cache_time: int = Field(
        default=_dataset_config.get("hf_cache_time", 3600),
        description="Time in seconds to cache Hugging Face datasets.",
    )

    # Model configuration
    model_upload_dir_config: Path = Field(
        default=Path(_model_config.get("upload_dir")),
        description="Base directory for storing uploaded model files.",
        exclude=True,
    )

    model_inference_limiter: int = Field(
        default=_model_config.get("inference_limiter"),
        description="Inference counter limiter in days.",
    )

    model_max_upload_size: int = Field(
        default=_model_config.get("max_upload_size"),
        description="Maximum allowed size for model uploads in bytes.",
    )

    # Inference configuration
    inference_device: Literal["cpu", "cuda"] = Field(
        default=_inference_config.get("device"),
        description="Device to use for model inference (CPU/GPU).",
    )

    # Cache Strategie

    cache_strategy: Literal["fixed_size", "vram_aware"] = Field(
        default=_cache_config.get("strategy", "fixed_size"),
        description="Cache strategy: fixed_size or vram_aware",
    )

    # Fixed Size Strategie Parameter

    cache_max_models: int = Field(
        default=_cache_config.get("max_models", 5),
        description="Maximum number of models in cache (fixed_size strategy)",
    )

    # Vram Aware Strategie Parameter

    cache_vram_safety_margin_gb: float = Field(
        default=_cache_config.get("vram_safety_margin_gb", 1.0),
        description="VRAM safety margin in GB (vram_aware strategy)",
    )

    # Evaluation configuration
    evaluation_default_max_samples: int = Field(
        default=_evaluation_config.get("default_max_samples", 1000),
        description="Default maximum number of samples used for evaluation.",
    )

    evaluation_default_random_seed: int = Field(
        default=_evaluation_config.get("default_random_seed", 42),
        description="Default random seed for reproducible evaluation sampling.",
    )

    @computed_field
    @property
    def dataset_upload_dir(self) -> Path:
        """Directory for storing uploaded dataset files."""
        if self.app_env == "testing":
            return Path("test_data/datasets")
        return self.dataset_upload_dir_config

    @computed_field
    @property
    def model_upload_dir(self) -> Path:
        """Directory for storing uploaded model files."""
        if self.app_env == "testing":
            return Path("test_data/models")
        return self.model_upload_dir_config

    @computed_field
    @property
    def model_inference_dir(self) -> Path:
        """Directory for storing model files used during inference."""
        if self.app_env == "testing":
            return Path("test_data/inference")
        return Path(_model_config.get("upload_dir"))

    @computed_field
    @property
    def log_path(self) -> Path:
        """Path where application logs are stored."""
        return Path(self.log_dir) / self.log_file

    @computed_field
    @property
    def reload(self) -> bool:
        """Whether to enable auto-reload on code changes."""
        return _server_config.get("reload") and (self.app_env == "development")

    @model_validator(mode="after")
    def validate_log_level(self) -> "Settings":
        """Adjust log level based on environment."""
        if self.app_env == "production" and self.log_level == "DEBUG":
            self.log_level = "INFO"
        return self

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Create a single instance of Settings to use throughout the application
settings = Settings()
