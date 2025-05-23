from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .schemas import TrainRequest


def train_model_task(train_request: TrainRequest):
    """Trainingslogik für lokale oder Huggingface-Modelle.
    Lädt Modell und Datensatz, startet Training mit Huggingface Trainer.
    Speichert trainierte Modelle unter data/models/trained_models.
    """
    # Zielverzeichnis für trainierte Modelle
    base_dir = Path("data/models/trained_models")
    base_dir.mkdir(parents=True, exist_ok=True)
    # output_dir ggf. relativ zu base_dir
    output_dir = base_dir / Path(train_request.output_dir).name

    print(f"Starte Training für {train_request.model_tag} mit Datensatz {train_request.dataset_path}")

    # Modell und Tokenizer laden (lokal oder Huggingface)
    model = AutoModelForSequenceClassification.from_pretrained(train_request.model_tag)
    tokenizer = AutoTokenizer.from_pretrained(train_request.model_tag)

    # Datensatz laden (CSV, Annahme: text & label Spalten)
    data_files = {"train": train_request.dataset_path}
    dataset = load_dataset("csv", data_files=data_files)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Trainingsargumente
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
    print(f"Training abgeschlossen und Modell gespeichert unter {output_dir}")
