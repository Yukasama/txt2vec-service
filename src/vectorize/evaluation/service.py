"""Service layer for model evaluation."""

from pathlib import Path

from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db

from .schemas import EvaluationRequest, EvaluationResponse
from .utils import EvaluationDatasetResolver, resolve_model_path
from .utils.evaluation_engine import EvaluationEngine

__all__ = ["evaluate_model_task_svc", "resolve_evaluation_dataset_svc"]


async def evaluate_model_task_svc(
    db: AsyncSession,
    evaluation_request: EvaluationRequest,
) -> EvaluationResponse:
    """Service entry point for model evaluation.

    Args:
        db (AsyncSession): Database session.
        evaluation_request (EvaluationRequest): Evaluation configuration.

    Returns:
        EvaluationResponse: Evaluation results.

    Raises:
        ValueError: If neither dataset_id nor training_task_id is provided.
        InvalidDatasetIdError: If dataset ID is invalid.
        TrainingDatasetNotFoundError: If dataset file not found.
        ModelNotFoundError: If model tag not found.
    """
    model = await get_ai_model_db(db, evaluation_request.model_tag)
    if not model:
        raise ModelNotFoundError(evaluation_request.model_tag)

    model_path = resolve_model_path(model.model_tag)

    dataset_path = await EvaluationDatasetResolver.resolve_evaluation_dataset(
        db, evaluation_request
    )

    engine = EvaluationEngine(model_path)

    if evaluation_request.baseline_model_tag:
        baseline_model = await get_ai_model_db(
            db, evaluation_request.baseline_model_tag
        )
        if not baseline_model:
            raise ModelNotFoundError(evaluation_request.baseline_model_tag)

        baseline_model_path = resolve_model_path(baseline_model.model_tag)

        return engine.run_comparison_evaluation(
            evaluation_request, dataset_path, baseline_model_path
        )

    return engine.run_simple_evaluation(evaluation_request, dataset_path)


async def resolve_evaluation_dataset_svc(
    db: AsyncSession, evaluation_request: EvaluationRequest
) -> Path:
    """Resolve the dataset path for evaluation - wrapper for backward compatibility."""
    return await EvaluationDatasetResolver.resolve_evaluation_dataset(
        db, evaluation_request
    )
