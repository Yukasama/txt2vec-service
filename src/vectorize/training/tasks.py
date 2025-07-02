"""Training tasks using Dramatiq for background processing."""

from uuid import UUID

import dramatiq
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import engine

from .schemas import TrainRequest
from .service import TrainingOrchestrator

__all__ = ["run_training_bg"]


@dramatiq.actor(max_retries=3)
async def run_training_bg(
    model_path: str,
    train_request_dict: dict,
    task_id: str,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Run SBERT training in the background using Dramatiq.

    Args:
        model_path: Path to the base model
        train_request_dict: Training configuration as dict (JSON serializable)
        task_id: Training task ID as string
        dataset_paths: List of dataset file paths
        output_dir: Output directory for the trained model
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            train_request = TrainRequest.model_validate(train_request_dict)

            logger.debug(
                "Starting background training task",
                task_id=task_id,
                model_path=model_path,
                dataset_paths=dataset_paths,
                output_dir=output_dir,
            )

            orchestrator = TrainingOrchestrator(db, UUID(task_id))
            await orchestrator.run_training_svc(
                model_path=model_path,
                train_request=train_request,
                dataset_paths=dataset_paths,
                output_dir=output_dir,
            )

            logger.debug(
                "Training task completed successfully",
                task_id=task_id,
                model_path=model_path,
            )

        except Exception as e:
            logger.error(
                "Error in training background task",
                task_id=task_id,
                model_path=model_path,
                error=str(e),
                exc_info=True,
            )
            raise
