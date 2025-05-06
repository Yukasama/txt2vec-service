"""Tasks module for handling background processes.

This module contains functions for managing background tasks related to
model uploads and processing, such as handling Hugging Face models.
"""

from uuid import UUID

from loguru import logger

from txt2vec.ai_model import AIModel
from txt2vec.ai_model.models import ModelSource
from txt2vec.ai_model.repository import save_ai_model
from txt2vec.common.status import TaskStatus
from txt2vec.config.db import get_session
from txt2vec.upload.huggingface_service import load_model_and_cache_only
from txt2vec.upload.repository import update_upload_task_status


async def process_huggingface_model_background(
    model_id: str,
    tag: str,
    task_id: UUID
) -> None:
    """Processes a Hugging Face model upload in the background.

    This function handles the background processing of a Hugging Face model
    upload task. It loads the model, saves it to the database, and updates
    the task status.

    Args:
        model_id (str): The ID of the Hugging Face model repository.
        tag (str): The specific revision or tag of the model to download.
        task_id (UUID): The unique identifier of the upload task.

    Raises:
        Exception: If an error occurs during model processing or database
        operations.
    """
    async for db in get_session():
        key = f"{model_id}@{tag}"
        try:
            logger.info(f"[BG] Starting model upload for task {task_id}")
            await load_model_and_cache_only(model_id, tag)  # No DB insert!

            ai_model = AIModel(
                model_tag=key,
                name=model_id,
                source=ModelSource.HUGGINGFACE,
            )
            await save_ai_model(db, ai_model)
            await update_upload_task_status(db, task_id, TaskStatus.DONE)

            logger.info(f"[BG] Task {task_id} completed successfully.")

        except Exception as e:
            logger.error(f"[BG] Error in task {task_id}: {e}")
            await update_upload_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(e)
            )
