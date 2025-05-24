"""Service for model training (Transformers Trainer API)."""

from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .schemas import TrainRequest
from .exceptions import TrainingDatasetNotFoundError


def train_model_service(train_request: TrainRequest) -> None:
    """Training logic for local or Huggingface models.

    Loads model and dataset, starts training with Huggingface Trainer.
    Saves trained models under data/models/trained_models.
    """
    base_dir = Path("data/models/trained_models")
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir = base_dir / Path(train_request.output_dir).name

    # Check if dataset file exists before loading
    dataset_path = Path(train_request.dataset_path)
    if not dataset_path.is_file():
        from loguru import logger

        logger.error(
            "Training failed: Dataset file not found: {}", train_request.dataset_path
        )
        raise TrainingDatasetNotFoundError(train_request.dataset_path)

    # Load model and tokenizer (local or Huggingface)
    model = AutoModelForSequenceClassification.from_pretrained(
        train_request.model_tag
    )
    tokenizer = AutoTokenizer.from_pretrained(train_request.model_tag)

    # Load dataset (CSV, expects columns: text & label)
    data_files = {"train": train_request.dataset_path}
    dataset = load_dataset("csv", data_files=data_files)

    def preprocess_function(examples: dict) -> dict:
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_request.epochs,
        learning_rate=train_request.learning_rate,
        per_device_train_batch_size=train_request.per_device_train_batch_size,
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
