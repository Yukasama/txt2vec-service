"""Tasks module for handling background processes.

This module contains functions for managing background tasks related to
model uploads and processing, such as handling Hugging Face models.
"""

from uuid import UUID

import dramatiq
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model import AIModel
from vectorize.ai_model.models import ModelSource
from vectorize.ai_model.repository import save_ai_model_db
from vectorize.config.db import engine
from vectorize.task.task_status import TaskStatus

from .exceptions import ModelAlreadyExistsError
from .github_service import load_github_model_and_cache_only_svc
from .huggingface_service import load_huggingface_model_and_cache_only_svc
from .repository import update_upload_task_status_db

__all__ = ["process_github_model_bg", "process_huggingface_model_bg"]


@dramatiq.actor(max_retries=3)
async def process_huggingface_model_bg(
    model_tag: str, revision: str, task_id: str
) -> None:
    """Processes a Hugging Face model upload in the background.

    This function handles the background processing of a Hugging Face model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        model_tag (str): The tag of the Hugging Face model repository.
        revision (str): The specific revision or version of the model to download.
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        ModelAlreadyExistsError: If the model already exists in the database.
        IntegrityError: If a database integrity error occurs.
        Exception: If an error occurs during model processing or database operations.
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        task_uid = UUID(task_id)

        try:
            logger.info("[BG] Starting model upload for task", taskId=task_uid)
            await load_huggingface_model_and_cache_only_svc(model_tag, revision)

            ai_model = AIModel(
                model_tag=model_tag.replace("/", "_"),
                name=model_tag,
                source=ModelSource.HUGGINGFACE,
            )

            try:
                await save_ai_model_db(db, ai_model)
            except IntegrityError as integrity_error:
                # Check if it's a unique constraint violation for model_tag
                constraint_msg = "UNIQUE constraint failed: ai_model.model_tag"
                if constraint_msg in str(integrity_error):
                    # Model was already created by another request, this is fine
                    logger.info(
                        "[BG] Model already exists, skipping creation",
                        modelTag=ai_model.model_tag,
                        taskId=task_uid,
                    )
                else:
                    # Re-raise other integrity errors
                    raise

            await update_upload_task_status_db(db, task_uid, TaskStatus.DONE)
            logger.info("[BG] Task completed successfully", taskId=task_uid)

        except ModelAlreadyExistsError as e:
            logger.error(f"[BG] Model already exists for task {task_uid}: {e}")
            await db.rollback()
            await update_upload_task_status_db(
                db, task_uid, TaskStatus.FAILED, error_msg=str(e)
            )
        except Exception as e:
            logger.error(f"[BG] Error in task {task_uid}: {e}")
            await db.rollback()
            await update_upload_task_status_db(
                db, task_uid, TaskStatus.FAILED, error_msg=str(e)
            )


@dramatiq.actor(max_retries=3)
async def process_github_model_bg(  # noqa: D417
    owner: str, repo: str, branch: str, task_id: str
) -> None:
    """Processes a GitHub model upload in the background.

    This function handles the background processing of a GitHub model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        db (AsyncSession): The database session for database operations.
        owner (str):
        repo (str):
        branch (str):
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        Exception: If an error occurs during model processing or database operations.
    """
    key = f"{owner}/{repo}@{branch}"
    normalized_key = key.replace("/", "_")
    async with AsyncSession(engine, expire_on_commit=False) as db:
        task_uid = UUID(task_id)
        try:
            logger.info("[BG] Starting model upload for task", taskId=task_id)
            await load_github_model_and_cache_only_svc(owner, repo, branch)

            ai_model = AIModel(
                model_tag=normalized_key,
                name=repo,
                source=ModelSource.GITHUB,
            )
            await save_ai_model_db(db, ai_model)
            await update_upload_task_status_db(db, task_uid, TaskStatus.DONE)

            logger.info("[BG] Task completed successfully", taskId=task_uid)

        except ModelAlreadyExistsError as e:
            logger.error(f"[BG] Model already exists for task {task_uid}: {e}")
            await db.rollback()
            await update_upload_task_status_db(
                db, task_uid, TaskStatus.FAILED, error_msg=str(e)
            )
        # pylint: disable=broad-except
        except Exception as e:
            logger.error(f"[BG] Error in task {task_uid}: {e}")
            await db.rollback()
            await update_upload_task_status_db(
                db, task_uid, TaskStatus.FAILED, error_msg=str(e)
            )
