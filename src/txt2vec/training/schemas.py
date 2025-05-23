from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model_tag: str = Field(..., description="Tag des Modells (lokal oder Huggingface)")
    dataset_path: str = Field(..., description="Pfad zum Trainingsdatensatz (lokal)")
    output_dir: str = Field(..., description="Pfad zum Speichern des trainierten Modells")
    epochs: int = Field(1, description="Anzahl der Trainingsepochen")
    learning_rate: float = Field(5e-5, description="Lernrate für das Training")
    per_device_train_batch_size: int = Field(8, description="Batchgröße pro Gerät")
    # Optional: weitere Parameter wie seed, evaluation_strategy, etc.
