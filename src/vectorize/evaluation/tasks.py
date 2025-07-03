"""Evaluation tasks using Dramatiq for background processing."""

import asyncio
import concurrent.futures
import json
import time
from pathlib import Path
from uuid import UUID

import dramatiq
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import engine
from vectorize.evaluation.exceptions import EvaluationModelNotFoundError
from vectorize.task.task_status import TaskStatus
from vectorize.training.exceptions import TrainingDatasetNotFoundError

from .schemas import EvaluationRequest
from .utils import (
    EvaluationDatabaseManager,
    EvaluationDatasetResolver,
)
from .utils.evaluation_engine import EvaluationEngine

__all__ = ["run_evaluation_bg"]


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

    def evaluate_in_thread() -> tuple[dict, dict]:
        """Run baseline evaluation in thread."""
        return engine.get_comparison_metrics_dict(
            dataset_path=dataset_path,
            baseline_model_path=baseline_model_path,
            max_samples=evaluation_request.max_samples,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(evaluate_in_thread)

        start_time = time.time()
        while not future.done():
            await asyncio.sleep(3)  # Yield every 3 seconds
            elapsed = time.time() - start_time
            logger.debug(
                "Baseline evaluation in progress...",
                model_path=engine.model_path,
                baseline_model=evaluation_request.baseline_model_tag,
                elapsed_minutes=round(elapsed / 60, 1)
            )

        # Get result
        trained_metrics_dict, baseline_metrics_dict = future.result()

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

    def evaluate_in_thread() -> dict:
        """Run simple evaluation in thread."""
        return engine.get_simple_metrics_dict(
            dataset_path, evaluation_request.max_samples
        )

    # Run evaluation in thread pool with yielding
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(evaluate_in_thread)

        start_time = time.time()
        while not future.done():
            await asyncio.sleep(3)  # Yield every 3 seconds
            elapsed = time.time() - start_time
            logger.debug(
                "Simple evaluation in progress...",
                model_path=engine.model_path,
                elapsed_minutes=round(elapsed / 60, 1)
            )

        # Get result
        metrics_dict = future.result()

    summary = engine.calculate_simple_summary(metrics_dict)

    await db_manager.save_simple_evaluation_results(
        evaluation_metrics=json.dumps(metrics_dict),
        evaluation_summary=summary,
    )


@dramatiq.actor(max_retries=3, queue_name="evaluation")
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

            from vectorize.evaluation.repository import get_evaluation_task_by_id_db
            eval_task = await get_evaluation_task_by_id_db(db, task_uuid)
            evaluation_dataset_ids: list[str] = []

            if evaluation_request.training_task_id:
                from vectorize.training.repository import get_train_task_by_id_db
                train_task = await get_train_task_by_id_db(db, UUID(evaluation_request.training_task_id))
                if train_task and hasattr(train_task, "train_dataset_ids"):
                    evaluation_dataset_ids = [str(x) for x in getattr(train_task, "train_dataset_ids", [])]
            elif evaluation_request.dataset_id:
                try:
                    evaluation_dataset_ids = [str(UUID(evaluation_request.dataset_id))]
                except Exception:
                    evaluation_dataset_ids = []

            if eval_task:
                eval_task.evaluation_dataset_ids = evaluation_dataset_ids
                db.add(eval_task)
                await db.commit()

            await db_manager.update_task_metadata(
                model_tag=evaluation_request.model_tag,
                baseline_model_tag=evaluation_request.baseline_model_tag,
            )

            loop = asyncio.get_running_loop()

            def create_engine() -> EvaluationEngine:
                return EvaluationEngine(model_path)

            engine_eval = await loop.run_in_executor(None, create_engine)

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
