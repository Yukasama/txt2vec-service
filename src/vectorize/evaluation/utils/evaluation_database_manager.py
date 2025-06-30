"""Database operations manager for evaluation tasks."""

from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db
from vectorize.evaluation.exceptions import EvaluationModelNotFoundError
from vectorize.task.task_status import TaskStatus

from ..repository import (
    update_evaluation_task_metadata_db,
    update_evaluation_task_results_db,
    update_evaluation_task_status_db,
)
from ..schemas import EvaluationRequest
from .model_resolver import resolve_model_path


class EvaluationDatabaseManager:
    """Handles database operations for evaluation tasks."""

    def __init__(self, db: AsyncSession, task_id: UUID) -> None:
        """Initialize the database manager.

        Args:
            db: Database session
            task_id: Evaluation task ID
        """
        self.db = db
        self.task_id = task_id

    async def setup_evaluation_task(self, evaluation_request: EvaluationRequest) -> str:
        """Set up evaluation task and return model path.

        Args:
            evaluation_request: Evaluation configuration

        Returns:
            Path to the model to evaluate

        Raises:
            ModelNotFoundError: If model not found
        """
        await self.update_task_status(TaskStatus.RUNNING, progress=0.1)

        model = await get_ai_model_db(self.db, evaluation_request.model_tag)
        if not model:
            raise ModelNotFoundError(evaluation_request.model_tag)

        model_path = resolve_model_path(model.model_tag)

        await self.update_task_status(TaskStatus.RUNNING, progress=0.2)

        return model_path

    async def update_task_metadata(
        self,
        model_tag: str,
        dataset_info: str | None,
        baseline_model_tag: str | None = None,
    ) -> None:
        """Update evaluation task metadata.

        Args:
            model_tag: Model tag being evaluated
            dataset_info: Dataset information string
            baseline_model_tag: Optional baseline model tag
        """
        await update_evaluation_task_metadata_db(
            self.db,
            self.task_id,
            model_tag=model_tag,
            dataset_info=dataset_info,
            baseline_model_tag=baseline_model_tag,
        )

        await self.update_task_status(TaskStatus.RUNNING, progress=0.3)

    async def update_task_status(
        self, status: TaskStatus, progress: float = 0.0, error_msg: str | None = None
    ) -> None:
        """Update evaluation task status.

        Args:
            status: New task status
            progress: Task progress (0.0 to 1.0)
            error_msg: Optional error message
        """
        await update_evaluation_task_status_db(
            self.db, self.task_id, status, error_msg=error_msg, progress=progress
        )

    async def save_simple_evaluation_results(
        self,
        evaluation_metrics: str,
        evaluation_summary: str,
    ) -> None:
        """Save simple evaluation results to database.

        Args:
            evaluation_metrics: Evaluation metrics as JSON string
            evaluation_summary: Summary of evaluation results
        """
        await self.update_task_status(TaskStatus.RUNNING, progress=0.9)

        await update_evaluation_task_results_db(
            self.db,
            self.task_id,
            evaluation_metrics=evaluation_metrics,
            evaluation_summary=evaluation_summary,
        )

    async def save_comparison_evaluation_results(
        self,
        evaluation_metrics: str,
        baseline_metrics: str,
        evaluation_summary: str,
    ) -> None:
        """Save comparison evaluation results to database.

        Args:
            evaluation_metrics: Trained model metrics as JSON string
            baseline_metrics: Baseline model metrics as JSON string
            evaluation_summary: Summary of comparison results
        """
        await self.update_task_status(TaskStatus.RUNNING, progress=0.9)

        await update_evaluation_task_results_db(
            self.db,
            self.task_id,
            evaluation_metrics=evaluation_metrics,
            baseline_metrics=baseline_metrics,
            evaluation_summary=evaluation_summary,
        )

    async def mark_evaluation_complete(self) -> None:
        """Mark the evaluation task as completed."""
        await self.update_task_status(TaskStatus.DONE, progress=1.0)

        logger.info(
            "Evaluation task completed successfully",
            task_id=str(self.task_id),
        )

    async def handle_evaluation_error(self, exc: Exception) -> None:
        """Handle evaluation errors.

        Args:
            exc: The exception that occurred
        """
        logger.error(
            "Evaluation failed",
            task_id=str(self.task_id),
            exc=str(exc),
        )

        try:
            await self.update_task_status(
                TaskStatus.FAILED, progress=0.0, error_msg=str(exc)
            )
        except Exception:
            logger.error(
                "Failed to update task status to FAILED", task_id=str(self.task_id)
            )

    async def validate_baseline_model(self, baseline_model_tag: str) -> str:
        """Validate and get baseline model path.

        Args:
            baseline_model_tag: Baseline model tag to validate

        Returns:
            Path to the baseline model

        Raises:
            EvaluationModelNotFoundError: If baseline model not found
        """
        try:
            baseline_model = await get_ai_model_db(self.db, baseline_model_tag)
        except Exception as exc:
            raise EvaluationModelNotFoundError(baseline_model_tag) from exc
        if not baseline_model:
            raise EvaluationModelNotFoundError(baseline_model_tag)
        return resolve_model_path(baseline_model.model_tag)
