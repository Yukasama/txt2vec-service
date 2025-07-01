"""Task to upload Hugging Face datasets to the database."""

import asyncio
from uuid import UUID

import dramatiq
from datasets import DatasetInfo, load_dataset_builder
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.db import engine

from .utils.process_hf_model import process_dataset

__all__ = ["upload_hf_dataset_bg"]


@dramatiq.actor(max_retries=3, time_limit=1800000)
async def upload_hf_dataset_bg(
    dataset_tag: str, task_id: str, subsets: list[str]
) -> None:
    """Upload a Hugging Face dataset in the background.

    Processes all subsets and splits of a Hugging Face dataset, downloading
    and converting them to JSONL format for storage in the database.
    Updates the task status upon completion.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset (e.g., 'squad').
        task_id: UUID of the upload task to track progress.
        subsets: List of dataset subsets to process (e.g., ['easy', 'hard']).

    Raises:
        Exception: If dataset loading or processing fails.
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        try:
            loop = asyncio.get_running_loop()

            for subset in subsets:
                info = await loop.run_in_executor(
                    None, _load_dataset_info, dataset_tag, subset
                )

                logger.debug(
                    "Processing Hugging Face dataset",
                    dataset_tag=dataset_tag,
                    subset=subset,
                    splits=list(info.splits.keys()) if info.splits else None,
                    features=list(info.features.keys()) if info.features else None,
                )

                await process_dataset(db, dataset_tag, UUID(task_id), subset, info)
                await db.commit()

            logger.info(
                "HF Dataset upload complete", dataset_tag=dataset_tag, task_id=task_id
            )
        except Exception as e:
            await db.rollback()
            logger.error(
                "Error in HF dataset background task",
                dataset_tag=dataset_tag,
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )
            raise


def _load_dataset_info(dataset_tag: str, subset: str) -> DatasetInfo:
    """Synchronous helper to load Hugging Face dataset info.

    Args:
        dataset_tag: The Hugging Face dataset identifier.
        subset: The subset name of the dataset to load.

    Returns:
        DatasetInfo: The metadata/info object for the specified dataset and subset.
    """
    return load_dataset_builder(dataset_tag, name=subset).info
