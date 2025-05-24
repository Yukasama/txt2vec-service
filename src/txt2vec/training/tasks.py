"""Background-Task für das Modelltraining."""

from .schemas import TrainRequest
from .service import train_model_service


def train_model_task(train_request: TrainRequest) -> None:
    """Wrapper für die Hintergrund-Task, ruft die Trainings-Servicefunktion auf."""
    train_model_service(train_request)
