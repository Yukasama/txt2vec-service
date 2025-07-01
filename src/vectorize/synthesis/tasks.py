"""Background tasks for synthesis processing."""

import base64
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
import dramatiq
import pandas as pd
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config import settings
from vectorize.config.db import engine
from vectorize.dataset.classification import Classification
from vectorize.dataset.dataset_source import DatasetSource
from vectorize.dataset.exceptions import DatasetNotFoundError
from vectorize.dataset.models import Dataset
from vectorize.dataset.repository import get_dataset_db, upload_dataset_db
from vectorize.dataset.schemas import DatasetUploadOptions
from vectorize.dataset.utils.dataset_fs import _save_dataframe_to_fs
from vectorize.task.task_status import TaskStatus

from .repository import update_synthesis_task_status
from .text_extractor import extract_text_from_media

__all__ = [
    "process_existing_dataset_background_bg",
    "process_file_contents_background_bg",
]


@dramatiq.actor(max_retries=3)
async def process_file_contents_background_bg(
    task_id: str,
    file_contents: list[tuple[str, str]],
    options_dict: dict | None = None,
) -> None:
    """Process file contents extracted from uploaded files.

    Args:
        task_id: ID of the synthesis task
        file_contents: List of tuples containing (filename, base64_encoded_content)
        options_dict: Optional dataset upload options
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        task_uuid = UUID(task_id)
        options = DatasetUploadOptions(**options_dict) if options_dict else None

        try:
            decoded_contents = [
                (filename, base64.b64decode(base64_content))
                for filename, base64_content in file_contents
            ]
            logger.info(
                "Starting processing of file contents",
                taskId=task_uuid,
                fileCount=len(decoded_contents),
            )

            dataset_ids = []

            for filename, content in decoded_contents:
                try:
                    dataset_id = await _process_single_file(
                        db, task_uuid, filename, content, options
                    )
                    if dataset_id:
                        dataset_ids.append(dataset_id)

                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    continue

            await _finalize_task_status(db, task_uuid, dataset_ids)

        except Exception as e:
            logger.error(f"Error in task {task_uuid}: {e}")
            await update_synthesis_task_status(
                db, task_uuid, TaskStatus.FAILED, error_msg=str(e)
            )
        finally:
            logger.debug("Database session closed", taskId=task_uuid)


async def _load_and_validate_source_dataset(
    db: AsyncSession, dataset_uuid: UUID
) -> tuple[Dataset, Path]:
    """Load and validate source dataset for synthesis.

    Args:
        db: Database session
        dataset_uuid: UUID of source dataset

    Returns:
        Tuple of (dataset object, file path)

    Raises:
        DatasetNotFoundError: If dataset not found in database
        FileNotFoundError: If dataset file not found on filesystem
    """
    source_dataset = await get_dataset_db(db, dataset_uuid)
    if not source_dataset:
        raise DatasetNotFoundError(
            dataset_id=dataset_uuid, message=f"Dataset {dataset_uuid} not found"
        )

    dataset_file_path = settings.dataset_upload_dir / source_dataset.file_name
    if not dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")

    return source_dataset, dataset_file_path


async def _create_synthetic_dataset_from_existing(
    db: AsyncSession,
    task_uuid: UUID,
    source_dataset: Dataset,
    df: "pd.DataFrame",
    classification: Classification,
) -> UUID:
    """Create synthetic dataset from processed existing dataset.

    Args:
        db: Database session
        task_uuid: Synthesis task UUID
        source_dataset: Source dataset object
        df: Processed dataframe
        classification: Dataset classification

    Returns:
        ID of created synthetic dataset
    """
    unique_name = f"{source_dataset.name}_{uuid4()}.jsonl"
    _save_dataframe_to_fs(df, unique_name)

    new_dataset = Dataset(
        name=f"{source_dataset.name}",
        file_name=unique_name,
        classification=classification,
        rows=len(df),
        source=DatasetSource.SYNTHETIC,
        synthesis_id=task_uuid,
    )

    return await upload_dataset_db(db, new_dataset)


@dramatiq.actor(max_retries=3)
async def process_existing_dataset_background_bg(
    task_id: str, dataset_id: str, options_dict: dict | None = None
) -> None:
    """Process an existing dataset through text extractor to create new synthetic data.

    Args:
        task_id: The synthesis task ID
        dataset_id: ID of existing dataset to use as input
        options_dict: Optional dataset upload options
    """
    async with AsyncSession(engine, expire_on_commit=False) as db:
        task_uuid = UUID(task_id)
        dataset_uuid = UUID(dataset_id)
        options = DatasetUploadOptions(**options_dict) if options_dict else None

        try:
            logger.info(
                "Processing existing dataset through text extractor",
                taskId=task_uuid,
                sourceDatasetId=dataset_uuid,
            )

            source_dataset, dataset_file_path = await _load_and_validate_source_dataset(
                db, dataset_uuid
            )

            df = extract_text_from_media(dataset_file_path, "dataset", options)

            classification = _determine_classification(df)

            new_dataset_id = await _create_synthetic_dataset_from_existing(
                db, task_uuid, source_dataset, df, classification
            )

            await update_synthesis_task_status(db, task_uuid, TaskStatus.DONE)

            logger.info(
                "Synthetic dataset created successfully",
                taskId=task_uuid,
                sourceDatasetId=dataset_uuid,
                newDatasetId=new_dataset_id,
                syntheticRows=len(df),
            )

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_uuid}: {e}")
            await update_synthesis_task_status(
                db, task_uuid, TaskStatus.FAILED, error_msg=str(e)
            )
        finally:
            logger.debug("Database session closed", taskId=task_uuid)


def _validate_file_for_processing(filename: str, content: bytes) -> str | None:
    """Validate file format and size for synthesis processing.

    Args:
        filename: Name of the file to validate
        content: File content bytes

    Returns:
        File extension if valid, None if invalid

    Logs warnings for invalid files.
    """
    file_path = Path(filename)
    ext = file_path.suffix.lower().lstrip(".")

    if ext not in {"png", "jpg", "jpeg", "pdf"}:
        logger.warning(
            f"Unsupported file format: {ext}",
            filename=filename,
        )
        return None

    file_size = len(content)
    if file_size > settings.dataset_max_upload_size:
        logger.warning(
            f"File too large: {file_size} bytes",
            filename=filename,
            maxSize=settings.dataset_max_upload_size,
        )
        return None

    return ext


async def _create_temp_file(content: bytes, ext: str) -> Path:
    """Create temporary file with content.

    Args:
        content: File content bytes
        ext: File extension

    Returns:
        Path to created temporary file
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"{uuid4()}.{ext}"
    temp_path = temp_dir / temp_filename

    async with aiofiles.open(temp_path, "wb") as temp_file:
        await temp_file.write(content)

    return temp_path


def _determine_classification(df: "pd.DataFrame") -> Classification:
    """Determine dataset classification based on dataframe columns.

    Args:
        df: Processed dataframe

    Returns:
        Classification type for the dataset
    """
    return (
        Classification.SENTENCE_TRIPLES
        if "rejected" in df.columns
        else Classification.SENTENCE_DUPLES
    )


async def _create_synthetic_dataset(
    db: AsyncSession,
    task_id: UUID,
    filename: str,
    df: "pd.DataFrame",
    classification: Classification,
) -> UUID:
    """Create and save synthetic dataset from processed data.

    Args:
        db: Database session
        task_id: Synthesis task ID
        filename: Original filename
        df: Processed dataframe
        classification: Dataset classification

    Returns:
        ID of created dataset
    """
    file_path = Path(filename)
    unique_name = f"{file_path.stem}_{uuid4()}.jsonl"

    _save_dataframe_to_fs(df, unique_name)

    dataset = Dataset(
        name=file_path.stem,
        file_name=unique_name,
        classification=classification,
        source=DatasetSource.SYNTHETIC,
        rows=len(df),
        synthesis_id=task_id,
    )

    return await upload_dataset_db(db, dataset)


def _cleanup_temp_file(temp_path: Path) -> None:
    """Clean up temporary file safely.

    Args:
        temp_path: Path to temporary file to clean up
    """
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


async def _process_single_file(
    db: AsyncSession,
    task_id: UUID,
    filename: str,
    content: bytes,
    options: DatasetUploadOptions | None,
) -> UUID | None:
    """Process a single file and create dataset.

    Args:
        db: Database session
        task_id: Synthesis task ID
        filename: Name of the file
        content: File content bytes
        options: Upload options

    Returns:
        Dataset ID if successful, None otherwise
    """
    ext = _validate_file_for_processing(filename, content)
    if ext is None:
        return None

    temp_path = await _create_temp_file(content, ext)

    try:
        df = extract_text_from_media(temp_path, ext, options)

        classification = _determine_classification(df)

        dataset_id = await _create_synthetic_dataset(
            db, task_id, filename, df, classification
        )

        logger.debug(
            "Processed file successfully",
            filename=filename,
            datasetId=dataset_id,
        )

        return dataset_id

    finally:
        _cleanup_temp_file(temp_path)


async def _finalize_task_status(
    db: AsyncSession, task_id: UUID, dataset_ids: list[UUID]
) -> None:
    """Finalize the task status based on processing results.

    Args:
        db: Database session
        task_id: Synthesis task ID
        dataset_ids: List of successfully created dataset IDs
    """
    if not dataset_ids:
        await update_synthesis_task_status(
            db,
            task_id,
            TaskStatus.FAILED,
            error_msg="No valid files could be processed",
        )
        logger.error(
            "Task failed: No valid files could be processed",
            taskId=task_id,
        )
    else:
        await update_synthesis_task_status(db, task_id, TaskStatus.DONE)

        logger.info(
            "Task completed successfully",
            taskId=task_id,
            datasetCount=len(dataset_ids),
            datasetIds=dataset_ids,
        )
