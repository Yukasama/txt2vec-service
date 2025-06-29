"""Evaluation tasks using Dramatiq for background processing."""

import json
import traceback
from pathlib import Path
from uuid import UUID

import dramatiq
import pkg_resources
import torch
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import engine
from vectorize.evaluation.exceptions import EvaluationModelNotFoundError
from vectorize.task.task_status import TaskStatus
from vectorize.training.exceptions import TrainingDatasetNotFoundError

from .repository import (
    update_evaluation_task_results_db,
    update_evaluation_task_status_db,
)
from .schemas import EvaluationRequest
from .service import EvaluationService
from .utils import (
    EvaluationDatabaseManager,
    EvaluationDatasetResolver,
)
from .utils.evaluation_engine import EvaluationEngine

__all__ = ["run_evaluation_bg", "run_mteb_benchmark_bg"]


async def _run_baseline_evaluation(
    engine: EvaluationEngine,
    db_manager: EvaluationDatabaseManager,
    evaluation_request: EvaluationRequest,
    dataset_path: Path,
) -> None:
    """Run evaluation with baseline comparison."""
    if not evaluation_request.baseline_model_tag:
        raise ValueError("Baseline model tag is required for comparison evaluation")

    baseline_model_path = await db_manager.validate_baseline_model(
        evaluation_request.baseline_model_tag
    )

    await db_manager.update_task_status(TaskStatus.RUNNING, progress=0.5)

    trained_metrics_dict, baseline_metrics_dict = engine.get_comparison_metrics_dict(
        dataset_path=dataset_path,
        baseline_model_path=baseline_model_path,
        max_samples=evaluation_request.max_samples,
    )

    summary = engine.calculate_improvement_summary(
        trained_metrics_dict, baseline_metrics_dict
    )

    await db_manager.save_comparison_evaluation_results(
        evaluation_metrics=json.dumps(trained_metrics_dict),
        baseline_metrics=json.dumps(baseline_metrics_dict),
        evaluation_summary=summary,
    )


async def _run_simple_evaluation(
    engine: EvaluationEngine,
    db_manager: EvaluationDatabaseManager,
    evaluation_request: EvaluationRequest,
    dataset_path: Path,
) -> None:
    """Run simple evaluation without baseline."""
    await db_manager.update_task_status(TaskStatus.RUNNING, progress=0.5)

    metrics_dict = engine.get_simple_metrics_dict(
        dataset_path, evaluation_request.max_samples
    )

    summary = engine.calculate_simple_summary(metrics_dict)

    await db_manager.save_simple_evaluation_results(
        evaluation_metrics=json.dumps(metrics_dict),
        evaluation_summary=summary,
    )


@dramatiq.actor(max_retries=3)
async def run_evaluation_bg(
    evaluation_request_dict: dict,
    task_id: str,
) -> None:
    """Run model evaluation in the background using Dramatiq.

    Args:
        evaluation_request_dict: Evaluation configuration as dict (JSON serializable)
        task_id: Evaluation task ID as string
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        db_manager = None
        try:
            evaluation_request = EvaluationRequest.model_validate(
                evaluation_request_dict
            )
            task_uuid = UUID(task_id)

            logger.debug(
                "Starting background evaluation task",
                task_id=task_id,
                model_tag=evaluation_request.model_tag,
                dataset_id=evaluation_request.dataset_id,
                training_task_id=evaluation_request.training_task_id,
                baseline_model_tag=evaluation_request.baseline_model_tag,
                max_samples=evaluation_request.max_samples,
            )

            db_manager = EvaluationDatabaseManager(db, task_uuid)

            model_path = await db_manager.setup_evaluation_task(evaluation_request)

            dataset_path = await EvaluationDatasetResolver.resolve_evaluation_dataset(
                db, evaluation_request
            )

            dataset_info = await EvaluationDatasetResolver.get_dataset_info(
                db, evaluation_request
            )

            await db_manager.update_task_metadata(
                model_tag=evaluation_request.model_tag,
                dataset_info=dataset_info,
                baseline_model_tag=evaluation_request.baseline_model_tag,
            )

            engine_eval = EvaluationEngine(model_path)

            if evaluation_request.baseline_model_tag:
                await _run_baseline_evaluation(
                    engine_eval, db_manager, evaluation_request, dataset_path
                )
            else:
                await _run_simple_evaluation(
                    engine_eval, db_manager, evaluation_request, dataset_path
                )

            await db_manager.mark_evaluation_complete()

            logger.info(
                "Evaluation task completed successfully",
                task_id=task_id,
                model_tag=evaluation_request.model_tag,
            )

        except (EvaluationModelNotFoundError, TrainingDatasetNotFoundError) as e:
            logger.error(
                f"Evaluation failed: {e}",
                task_id=task_id,
                model_tag=evaluation_request_dict.get("model_tag", "unknown"),
                error=str(e),
                exc_info=False,
            )
            try:
                if db_manager is not None:
                    await db_manager.handle_evaluation_error(e)
                else:
                    db_manager = EvaluationDatabaseManager(db, UUID(task_id))
                    await db_manager.handle_evaluation_error(e)
            except Exception:
                logger.error("Failed to update task status to FAILED", task_id=task_id)
            return
        except Exception as e:
            logger.error(
                "Error in evaluation background task",
                task_id=task_id,
                model_tag=evaluation_request_dict.get("model_tag", "unknown"),
                error=str(e),
                exc_info=True,
            )
            try:
                if db_manager is not None:
                    await db_manager.handle_evaluation_error(e)
                else:
                    db_manager = EvaluationDatabaseManager(db, UUID(task_id))
                    await db_manager.handle_evaluation_error(e)
            except Exception:
                logger.error("Failed to update task status to FAILED", task_id=task_id)
            raise


@dramatiq.actor(max_retries=3)
async def run_mteb_benchmark_bg(model_tag: str, task_id: str) -> None:
    """Run MTEB benchmark in the background using Dramatiq."""
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            logger.info(
                "Starting MTEB benchmark task {task_id} for model {model_tag}",
                task_id=task_id,
                model_tag=model_tag,
            )
            logger.info("PyTorch version {version}", version=torch.__version__)
            logger.info("MTEB version {version}", version=pkg_resources.get_distribution("mteb").version)
            logger.info("Sentence-Transformers version {version}", version=pkg_resources.get_distribution("sentence-transformers").version)
            logger.info("Datasets version {version}", version=pkg_resources.get_distribution("datasets").version)
            cuda_available = torch.cuda.is_available()
            logger.info("CUDA available {cuda}", cuda=cuda_available)
            if cuda_available:
                logger.info("CUDA device {device}", device=torch.cuda.get_device_name(0))

            await update_evaluation_task_status_db(db, UUID(task_id), TaskStatus.QUEUED, progress=0.0)
            service = EvaluationService(db)
            logger.debug("Calling run_benchmark for model_tag {model_tag}", model_tag=model_tag)
            results = service.run_benchmark(model_tag)
            logger.debug("Received results from run_benchmark of type {type}", type=type(results))

            await update_evaluation_task_results_db(
                db,
                UUID(task_id),
                evaluation_metrics=json.dumps(results),
                evaluation_summary=f"MTEB benchmark completed for model {model_tag}",
            )
            await update_evaluation_task_status_db(db, UUID(task_id), TaskStatus.DONE, progress=1.0)
            logger.info("MTEB benchmark completed successfully for task {task_id}", task_id=task_id)
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(
                "MTEB benchmark failed for task {task_id}: {error}\nStack trace:\n{trace}",
                task_id=task_id,
                error=str(e),
                trace=tb_str,
            )
            await update_evaluation_task_status_db(
                db, UUID(task_id), TaskStatus.FAILED, error_msg=str(e), progress=1.0
            )
            raise
