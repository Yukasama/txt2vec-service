from uuid import UUID

from loguru import logger

from txt2vec.ai_model import AIModel
from txt2vec.ai_model.models import ModelSource
from txt2vec.ai_model.repository import save_ai_model
from txt2vec.common.status import TaskStatus
from txt2vec.config.db import get_session
from txt2vec.upload.huggingface_service import load_model_and_cache_only
from txt2vec.upload.repository import update_upload_task_status


async def process_huggingface_model_background(model_id: str, tag: str, task_id: UUID):
    async for db in get_session():
        key = f"{model_id}@{tag}"
        try:
            logger.info(f"[BG] Beginne Modell-Upload für Task {task_id}")
            await load_model_and_cache_only(model_id, tag)  # kein DB-Insert!

            ai_model = AIModel(
                model_tag=key,
                name=model_id,
                source=ModelSource.HUGGINGFACE,
            )
            await save_ai_model(db, ai_model)
            await update_upload_task_status(db, task_id, TaskStatus.DONE)

            logger.info(f"[BG] Task {task_id} erfolgreich abgeschlossen.")

        except Exception as e:
            logger.error(f"[BG] Fehler bei Task {task_id}: {e}")
            await update_upload_task_status(
                db, task_id, TaskStatus.FAILED, error_msg=str(e)
            )
