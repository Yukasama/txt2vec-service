"""Schemas for the training API."""

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request body for model training."""
    model_tag: str = Field(
        ..., description="Tag of the model (local or Huggingface)")
    dataset_path: str = Field(
        ..., description="Path to the training dataset (local)")
    output_dir: str = Field(
        ..., description="Path to save the trained model")
    epochs: int = Field(1, description="Number of training epochs")
    learning_rate: float = Field(5e-5, description="Learning rate for training")
    per_device_train_batch_size: int = Field(8, description="Batch size per device")
    # Optional: more parameters like seed, evaluation_strategy, etc.
