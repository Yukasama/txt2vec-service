"""Schemas for SBERT/SentenceTransformer triplet training."""

from pydantic import BaseModel, Field

from vectorize.training.models import TrainingTask


class TrainRequest(BaseModel):
    """Request for SBERT triplet training.

    Expects JSONL with columns: question, positive, negative. Supports all
    important and optional sentence-transformers parameters. If val_dataset_id is
    not provided, 10% of the first training dataset will be used for validation.
    """

    model_tag: str = Field(description="Tag of the local model in the database")
    train_dataset_ids: list[str] = Field(
        description=(
            "IDs of the training datasets (CSV/JSONL, columns: question, "
            "positive, negative). If multiple are given, they will be "
            "concatenated for training."
        ),
        min_length=1,
    )
    val_dataset_id: str | None = Field(
        default=None,
        description=(
            "Optional ID of the validation dataset (same format as training). "
            "If not set, 10% split from the first training dataset is used."
        ),
    )
    epochs: int = Field(1, description="Number of training epochs", gt=0)
    per_device_train_batch_size: int = Field(
        8, description="Batch size per device", gt=0
    )
    learning_rate: float = Field(2e-5, description="Learning rate", gt=0)
    warmup_steps: int | None = Field(None, description="Number of warmup steps")
    optimizer_name: str | None = Field(
        None, description="Optimizer (e.g. AdamW, Adam, RMSprop)"
    )
    scheduler: str | None = Field(
        None, description="Learning rate scheduler (e.g. 'constantlr', 'warmuplinear')"
    )
    weight_decay: float | None = Field(
        None, description="Weight decay (L2 regularization)"
    )
    max_grad_norm: float | None = Field(
        None, description="Max gradient norm for clipping"
    )
    use_amp: bool | None = Field(
        None, description="Use automatic mixed precision (AMP)"
    )
    show_progress_bar: bool | None = Field(None, description="Show progress bar")
    evaluation_steps: int | None = Field(
        None, description="Evaluate every X steps (optional, if val set)"
    )
    output_path: str | None = Field(
        None, description="Path to save model (optional, auto if not set)"
    )
    save_best_model: bool | None = Field(
        None, description="Save best model (if val set)"
    )
    save_each_epoch: bool | None = Field(
        None, description="Save model after each epoch"
    )
    save_optimizer_state: bool | None = Field(None, description="Save optimizer state")
    dataloader_num_workers: int | None = Field(
        None, description="Number of DataLoader workers (default: 0)"
    )
    device: str | None = Field(
        None, description="Device for training ('cpu', 'cuda', 'mps', etc.)"
    )
    timeout_seconds: int | None = Field(
        None,
        description="Timeout for training in seconds (default: 7200, i.e. 2 hours)",
    )


class TrainingStatusResponse(BaseModel):
    """Response schema for training status."""

    task_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    end_date: str | None = None
    error_msg: str | None = None
    trained_model_id: str | None = None

    train_dataset_ids: list[str] | None = None
    val_dataset_id: str | None = None
    baseline_model_id: str | None = None

    train_runtime: float | None = None
    train_samples_per_second: float | None = None
    train_steps_per_second: float | None = None
    train_loss: float | None = None
    epoch: float | None = None

    @classmethod
    def from_training_task(cls, task: "TrainingTask") -> "TrainingStatusResponse":
        """Create a TrainingStatusResponse from a TrainingTask object."""
        # Convert TaskStatus enum to string
        status_value = task.task_status.value if task.task_status else "F"

        return cls(
            task_id=str(task.id),
            status=status_value,
            created_at=str(task.created_at) if task.created_at else None,
            updated_at=str(task.updated_at) if task.updated_at else None,
            end_date=str(task.end_date) if task.end_date else None,
            error_msg=task.error_msg,
            trained_model_id=(
                str(task.trained_model_id) if task.trained_model_id else None
            ),
            train_dataset_ids=task.train_dataset_ids,
            val_dataset_id=task.val_dataset_id,
            baseline_model_id=task.baseline_model_id,
            train_runtime=task.train_runtime,
            train_samples_per_second=task.train_samples_per_second,
            train_steps_per_second=task.train_steps_per_second,
            train_loss=task.train_loss,
            epoch=task.epoch,
        )
