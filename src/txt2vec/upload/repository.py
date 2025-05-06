from datetime import datetime
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.common.status import TaskStatus
from txt2vec.upload.models import UploadTask


async def save_upload_task(db: AsyncSession, task: UploadTask) -> None:
    db.add(task)
    await db.commit()
    await db.refresh(task)


async def update_upload_task_status(
    db: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    error_msg: str | None = None,
) -> None:
    result = await db.exec(select(UploadTask).where(UploadTask.id == task_id))
    task = result.one()
    task.task_status = status
    task.end_date = datetime.utcnow()
    task.error_msg = error_msg
    await db.commit()
