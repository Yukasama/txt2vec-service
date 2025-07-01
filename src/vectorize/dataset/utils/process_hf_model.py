"""Process Hugging Face dataset splits and subsets."""

import asyncio
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import orjson
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.info import DatasetInfo
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from tqdm.auto import tqdm

from vectorize.config.config import settings
from vectorize.task.task_status import TaskStatus

from ..classification import Classification
from ..dataset_source import DatasetSource
from ..models import Dataset
from ..repository import update_upload_task_status_db, upload_dataset_db
from .match_hf_schema import match_schema

__all__ = ["process_dataset"]


async def process_dataset(
    db: AsyncSession,
    dataset_tag: str,
    task_id: UUID,
    subset: str,
    info: DatasetInfo,
) -> None:
    """Process splits for a single dataset subset.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset.
        task_id: UUID of the upload task for status tracking.
        subset: Name of the dataset subset/configuration.
        info: Dataset metadata containing available splits.

    Raises:
        Exception: If dataset processing fails, updates task status to FAILED.
    """
    try:
        if subset == "default":
            if not info.splits:
                await _process_single_dataset(db, dataset_tag, task_id)
            else:
                for split in info.splits:
                    await _process_single_dataset(db, dataset_tag, task_id, split)
        elif not info.splits:
            await _process_single_dataset(db, dataset_tag, task_id, subset=subset)
        else:
            for split in info.splits:
                await _process_single_dataset(db, dataset_tag, task_id, split, subset)

        await update_upload_task_status_db(db, task_id, TaskStatus.DONE)
    except Exception as e:
        await update_upload_task_status_db(
            db,
            task_id,
            TaskStatus.FAILED,
            error_msg=f"Dataset processing failed: {e!s}",
        )
        raise


# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


async def _process_single_dataset(
    db: AsyncSession,
    dataset_tag: str,
    task_id: UUID,
    split: str | None = None,
    subset: str | None = "default",
) -> None:
    """Process a single dataset split/subset and save to database.

    Downloads a specific split and subset of a Hugging Face dataset,
    converts it to JSONL format, and creates a database record.

    Args:
        db: Database session for persistence operations.
        dataset_tag: Tag identifier for the Hugging Face dataset.
        task_id: UUID of the upload task for status tracking.
        split: Name of the dataset split (e.g., 'train', 'validation').
        subset: Name of the dataset subset/configuration. Defaults to "default".

    Raises:
        Exception: If dataset loading, file writing, or database operations fail.
    """

    def _blocking_dataset_work() -> tuple[
        DatasetDict | HFDataset | IterableDatasetDict | IterableDataset, str, Path, int
    ]:
        """All the blocking operations in one function."""
        ds = load_dataset(dataset_tag, name=subset, split=split, streaming=False)
        logger.debug(
            "Loaded HF dataset", dataset_tag=dataset_tag, split=split, subset=subset
        )

        parts = [dataset_tag.replace("/", "_")]
        if split:
            parts.append(split)
        if subset and subset != "default":
            parts.append(subset)

        file_name = f"{'_'.join(parts)}.jsonl"
        file_path = settings.dataset_upload_dir / file_name
        rows = _write_jsonl(ds, file_path)

        return ds, file_name, file_path, rows

    file_path = None
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            _, file_name, file_path, rows = await loop.run_in_executor(
                executor, _blocking_dataset_work
            )

        logger.debug(
            "Loaded HF dataset", dataset_tag=dataset_tag, split=split, subset=subset
        )

        dataset_name = dataset_tag
        if split:
            dataset_name += f"_{split}"
        if subset and subset != "default":
            dataset_name += f"_{subset}"

        dataset = Dataset(
            name=dataset_name,
            classification=Classification.SENTENCE_TRIPLES,
            file_name=file_name,
            source=DatasetSource.HUGGINGFACE,
            rows=rows,
        )

        dataset_id = await upload_dataset_db(db, dataset)
        await update_upload_task_status_db(db, task_id, TaskStatus.DONE)
        logger.debug("HF Dataset saved", dataset_tag=dataset_tag, dataset_id=dataset_id)

    except SQLAlchemyError:
        if file_path and file_path.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, file_path.unlink)
            logger.debug("Cleaned up file after database error", file_path=file_path)
        raise

    except Exception as e:
        logger.error(
            "Error processing Hugging Face dataset",
            dataset_tag=dataset_tag,
            split=split,
            subset=subset,
            error=str(e),
        )
        raise


_LINE_SEP = b"\xe2\x80\xa8"
_PARA_SEP = b"\xe2\x80\xa9"


def _schema_mapping(feature_names: set[str]) -> dict[str, str]:
    """Return column mapping from feature names to canonical names.

    Args:
        feature_names: Set of column names from the dataset.

    Returns:
        Mapping original columns to canonical names (question, positive, negative).

    Raises:
        ValueError: If no valid mapping found for the given feature set.
    """
    for schema in settings.dataset_hf_allowed_schemas:
        required = set(schema) if isinstance(schema, (list, tuple)) else {schema}
        if required.issubset(feature_names):
            cols = list(schema)
            return {
                cols[0]: "question",
                cols[1]: "positive",
                cols[2]: "negative",
            }

    raise ValueError("No mapping found for feature set:", feature_names)


def _write_jsonl(
    ds: HFDataset | Iterable[dict[str, Any]],
    path: Path,
    *,
    batch_size: int = 2048,
) -> int:
    """Write dataset to JSONL with column mapping and filtering.

    Args:
        ds: HuggingFace dataset or iterable of dictionaries.
        path: Output file path for JSONL.
        batch_size: Number of records to buffer before writing to disk.

    Returns:
        Number of records successfully written.

    Raises:
        ValueError: If dataset is empty or columns don't match allowed schemas.
    """
    if isinstance(ds, HFDataset):
        ds = ds.with_format("python")
        iterator: Iterator[Mapping[str, Any]] = cast(
            Iterator[Mapping[str, Any]], iter(ds)
        )
        feature_names = set(ds.column_names)
        total_rows = ds.num_rows
    else:
        iterator = (row for row in ds if isinstance(row, Mapping))
        iterator = iter(iterator)
        try:
            first_row = next(iterator)
        except StopIteration as e:
            raise ValueError("Dataset appears to be empty") from e
        feature_names = set(first_row.keys())
        iterator = chain([first_row], iterator)
        total_rows = None

    if not match_schema(feature_names):
        raise ValueError(
            f"Columns {sorted(feature_names)} do not match any allowed schema "
            "defined in settings.dataset_hf_allowed_schemas"
        )

    mapping = _schema_mapping(feature_names)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    tmp_path = path.parent / tmp_name

    row_count, buf = 0, bytearray()
    with (
        tmp_path.open("wb") as fh,
        tqdm(total=total_rows, desc=path.name, unit="ex") as bar,
    ):
        for example in iterator:
            try:
                rec = orjson.dumps(_canon(mapping, example))
            except (KeyError, orjson.JSONEncodeError):
                logger.warning("Skipped invalid row #%d", row_count)
                continue

            if (_LINE_SEP in rec) or (_PARA_SEP in rec):
                rec = rec.replace(_LINE_SEP, b"\\u2028").replace(_PARA_SEP, b"\\u2029")

            buf += rec + b"\n"
            row_count += 1
            if row_count % batch_size == 0:
                fh.write(buf)
                buf.clear()
                bar.update(batch_size)

        if buf:
            fh.write(buf)
            bar.update(len(buf))

    tmp_path.replace(path)
    return row_count


def _canon(mapping: dict[str, str], row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a concrete dict with canonical keys only."""
    return {canon: row[src] for src, canon in mapping.items()}
