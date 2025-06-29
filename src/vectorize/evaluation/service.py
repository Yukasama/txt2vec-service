"""Service layer for model evaluation."""

import json
import traceback
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from mteb import MTEB, get_tasks
from sentence_transformers import SentenceTransformer
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


class EvaluationService:
    def __init__(self, db: object) -> None:
        self.db = db

    def run_benchmark(self, model_tag: str) -> dict[str, object]:
        logger.warning("Starting MTEB benchmark for model {model_tag}", model_tag=model_tag)
        logger.warning("Model tag received {model_tag}", model_tag=model_tag)

        model_path_str = resolve_model_path(model_tag)
        model_path = Path(model_path_str)
        logger.warning("Resolved model path {path}", path=model_path_str)

        logger.warning("Loading model from path {path}", path=model_path)
        model = self._load_model(model_path)
        logger.warning("Model loaded successfully from {path}", path=model_path)

        dataset_name = "mteb/stsbenchmark-sts"
        logger.warning("Loading dataset {dataset}", dataset=dataset_name)
        try:
            load_dataset(dataset_name)
            logger.warning("Dataset loaded and cached successfully: {dataset}", dataset=dataset_name)
        except Exception as e:
            logger.error("Failed to load dataset {dataset}: {error}", dataset=dataset_name, error=str(e))
            raise

        logger.warning("Configuring MTEB benchmark with tasks {tasks}", tasks=["STSBenchmark"])
        benchmark = MTEB(tasks=get_tasks(tasks=["STSBenchmark"]))
        logger.warning("Benchmark tasks initialized {task_names}", task_names=[task.metadata.name for task in benchmark.tasks])

        logger.info("Executing MTEB benchmark for STSBenchmark task")
        try:
            results = benchmark.run(model, output_folder=None)
            logger.warning("Raw MTEB benchmark results type {type}", type=type(results))
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(
                "Error during MTEB benchmark run.\nException type: {etype}\nException message: {msg}\nStack trace:\n{trace}",
                etype=type(e).__name__,
                msg=str(e),
                trace=tb_str,
            )
            raise

        logger.warning("Processing benchmark results for serialization")
        serialized_results = self._serialize_results(results)
        logger.warning("Benchmark completed for model {model_tag}", model_tag=model_tag)

        logger.warning("Unloading model")
        self._unload_model(model)
        logger.warning("Model unloaded successfully")

        return serialized_results

    @staticmethod
    def _load_model(model_path: Path) -> SentenceTransformer:
        logger.info("Loading model from: {path}", path=model_path)
        try:
            model = SentenceTransformer(str(model_path))
            if torch.cuda.is_available():
                model = model.to("cuda")
                logger.info("Model loaded on CUDA device: {device}", device=torch.cuda.get_device_name(0))
            else:
                logger.warning("CUDA not available. Using CPU.")
            return model
        except Exception as exc:
            logger.error("Failed to load model: {error}", error=exc)
            raise

    @staticmethod
    def _unload_model(model: SentenceTransformer) -> None:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Freed CUDA cache")
        except Exception as exc:
            logger.warning("Failed to free CUDA cache: {error}", error=exc)

    @staticmethod
    def _serialize_results(results: Sequence[object]) -> dict[str, object]:
        try:
            return json.loads(json.dumps(results))
        except Exception as e:
            logger.error("Failed to serialize results: {error}", error=e)
            return {}
