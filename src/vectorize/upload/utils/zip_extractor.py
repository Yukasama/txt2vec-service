"""Extraction utilities for model ZIP files."""

import contextlib
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model import AIModel, ModelSource
from vectorize.ai_model.exceptions import ModelNotFoundError
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.config.config import settings

from ..exceptions import (
    ModelAlreadyExistsError,
    ModelTooLargeError,
    NoValidModelsFoundError,
)
from .zip_validator import validate_model_files

__all__ = ["process_model_directory", "process_single_model", "save_zip_to_temp"]

ALLOWED_EXTENSIONS = {".pt", ".pth", ".bin", ".model", ".safetensors", ".json"}


async def save_zip_to_temp(file: UploadFile) -> Path:
    """Save ZIP file to a temporary location.

    Args:
        file: The uploaded ZIP file

    Returns:
        Path to the saved temporary file

    Raises:
        ModelTooLargeError: If the file exceeds maximum upload size
    """
    # Create temporary file path
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"{uuid4()}.zip"
    temp_path = temp_dir / temp_filename

    size = 0
    chunk_size = 1024 * 1024

    async with aiofiles.open(temp_path, "wb") as dest_file:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > settings.model_max_upload_size:
                # Clean up partial file
                with contextlib.suppress(Exception):
                    temp_path.unlink(missing_ok=True)
                raise ModelTooLargeError(size)
            await dest_file.write(chunk)

    await file.seek(0)

    return temp_path


async def _extract_file_from_zip(
    zip_ref: zipfile.ZipFile,
    file_path: str,
    target_path: Path,
    common_prefix: str | None = None,
) -> bool:
    """Extract a single file from ZIP to target path."""
    try:
        if file_path.endswith("/"):
            return False

        file_extension = Path(file_path).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.debug("Skipping file with non-allowed extension: {}", file_path)
            return False

        if common_prefix and file_path.startswith(common_prefix + "/"):
            relative_path = file_path[len(common_prefix) + 1 :]
        else:
            relative_path = Path(file_path).name

        final_target_path = target_path / relative_path
        final_target_path.parent.mkdir(parents=True, exist_ok=True)

        with zip_ref.open(file_path) as source:
            async with aiofiles.open(final_target_path, "wb") as target:
                # Read and write in chunks for large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := source.read(chunk_size):
                    await target.write(chunk)

        logger.debug("Extracted {} to {}", file_path, final_target_path)
        return True
    except Exception as e:
        logger.error("Error extracting {}: {}", file_path, e)
        return False


def _find_common_prefix(file_paths: list[str]) -> str | None:
    """Find common prefix for file paths."""
    common_prefix = None
    files = 2
    for path in file_paths:
        if path.endswith("/"):
            continue

        parts = path.split("/")
        if len(parts) >= files:
            prefix = "/".join(parts[:-1])
            if common_prefix is None or len(prefix) < len(common_prefix):
                common_prefix = prefix

    return common_prefix


async def process_model_directory(
    zip_ref: zipfile.ZipFile,
    model_name: str,
    file_paths: list[str],
    base_dir: Path,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Process a single model directory from the ZIP file.

    Args:
        zip_ref: Open ZIP file reference
        model_name: Name for this specific model
        file_paths: List of paths within this model's directory
        base_dir: Base directory where to extract files
        db: Database session for persistence

    Returns:
        Tuple of (model directory path, model database ID)

    Raises:
        NoValidModelsFoundError: When no valid model files were found
    """
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    try:
        await get_ai_model_db(db, safe_model_name)
        raise ModelAlreadyExistsError(
            f"Model with tag '{safe_model_name}' already exists"
        )
    except ModelNotFoundError:
        pass

    model_dir = base_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    common_prefix = _find_common_prefix(file_paths)

    for file_path in file_paths:
        if await _extract_file_from_zip(zip_ref, file_path, model_dir, common_prefix):
            if common_prefix and file_path.startswith(common_prefix + "/"):
                relative_path = file_path[len(common_prefix) + 1 :]
            else:
                relative_path = Path(file_path).name
            extracted_files.append(model_dir / relative_path)

    if not extracted_files:
        raise NoValidModelsFoundError(
            f"No files could be extracted for model {model_name}"
        )

    if not validate_model_files(extracted_files):
        raise NoValidModelsFoundError(
            f"No valid PyTorch model found in directory {model_name}"
        )

    ai_model = AIModel(
        model_tag=safe_model_name,
        name=model_name,
        source=ModelSource.LOCAL,
    )
    model_id = await save_ai_model_db(db, ai_model)

    return (model_dir, str(model_id))


async def process_single_model(
    zip_ref: zipfile.ZipFile,
    model_name: str,
    file_list: list,
    base_dir: Path,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Process the ZIP contents as a single model.

    Args:
        zip_ref: Open ZIP file reference
        model_name: Name for the model
        file_list: List of all files in the ZIP
        base_dir: Base directory where to extract files
        db: Database session for persistence

    Returns:
        Tuple of (model directory path, model database ID)

    Raises:
        NoValidModelsFoundError: When no valid model files were found
    """
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)

    try:
        await get_ai_model_db(db, safe_model_name)
        raise ModelAlreadyExistsError(
            f"Model with tag '{safe_model_name}' already exists"
        )
    except ModelNotFoundError:
        pass

    model_dir = base_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    for file_info in file_list:
        if file_info.is_dir() or file_info.file_size <= 0:
            continue

        file_extension = Path(file_info.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.debug(
                "Skipping file with non-allowed extension: {}", file_info.filename
            )
            continue

        target_name = Path(file_info.filename).name
        target_path = model_dir / target_name

        with zip_ref.open(file_info) as source:
            async with aiofiles.open(target_path, "wb") as target:
                # Read and write in chunks for large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := source.read(chunk_size):
                    await target.write(chunk)

        extracted_files.append(target_path)

    if not extracted_files:
        raise NoValidModelsFoundError("No valid files found in the archive")

    if not validate_model_files(extracted_files):
        raise NoValidModelsFoundError("No valid PyTorch model found in the archive")

    ai_model = AIModel(
        model_tag=safe_model_name,
        name=model_name,
        source=ModelSource.LOCAL,
    )
    model_id = await save_ai_model_db(db, ai_model)

    return (model_dir, str(model_id))
