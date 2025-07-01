"""Extraction utilities for model ZIP files."""

import asyncio
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
MAX_FILENAME_LENGTH = 255
INVALID_FILENAME_CHARS = {"<", ">", "|", "*", "?"}

MSG_INVALID_EXTENSION = "Skipping file with non-allowed extension: {}"
MSG_UNSAFE_PATH = "Skipping file with unsafe path: '{}'"
MSG_PATH_TRAVERSAL = "Path traversal attempt detected: '{}'"


def _is_safe_path(target_path: Path, base_path: Path) -> bool:
    """Check if target path is safe and within base directory.

    Args:
        target_path: The target path to validate
        base_path: The base directory that should contain the target

    Returns:
        True if path is safe, False otherwise
    """
    try:
        resolved_target = target_path.resolve()
        resolved_base = base_path.resolve()

        return resolved_target.is_relative_to(resolved_base)
    except (OSError, ValueError):
        return False


def _sanitize_path(file_path: str) -> str | None:
    """Sanitize file path to prevent path traversal attacks.

    Args:
        file_path: Raw file path from ZIP archive

    Returns:
        Sanitized filename or None if invalid
    """
    if not file_path or file_path.endswith("/"):
        return None

    normalized_path = Path(file_path).as_posix()

    if (
        ".." in normalized_path
        or normalized_path.startswith("/")
        or ":" in normalized_path
        or any(char in normalized_path for char in INVALID_FILENAME_CHARS)
    ):
        logger.warning("Rejected unsafe path: '{}'", file_path)
        return None

    sanitized_name = Path(file_path).name

    if (
        not sanitized_name
        or sanitized_name in {".", ".."}
        or sanitized_name.startswith(".")
        or len(sanitized_name) > MAX_FILENAME_LENGTH
    ):
        logger.warning("Rejected unsafe filename: '{}'", file_path)
        return None

    return sanitized_name


async def save_zip_to_temp(file: UploadFile) -> Path:
    """Save ZIP file to a temporary location.

    Args:
        file: The uploaded ZIP file

    Returns:
        Path to the saved temporary file

    Raises:
        ModelTooLargeError: If the file exceeds maximum upload size
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"{uuid4()}.zip"
    temp_path = temp_dir / temp_filename

    size = 0
    chunk_size = 1024 * 1024

    async with aiofiles.open(temp_path, "wb") as dest_file:
        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > settings.model_max_upload_size:
                with contextlib.suppress(Exception):
                    temp_path.unlink(missing_ok=True)
                raise ModelTooLargeError(size)
            await dest_file.write(chunk)

    await file.seek(0)

    return temp_path


def _extract_zip_file_sync(zip_ref: zipfile.ZipFile, file_path: str) -> bytes:
    """Extract file data from ZIP synchronously.

    Args:
        zip_ref: Open ZIP file reference
        file_path: Path to file within ZIP (must be pre-validated)

    Returns:
        File content as bytes

    Raises:
        ValueError: If file_path contains unsafe characters
    """
    # Additional safety check for path traversal in ZIP file paths
    if ".." in file_path or file_path.startswith("/"):
        raise ValueError(f"Unsafe file path in ZIP: {file_path}")

    with zip_ref.open(file_path) as source:
        return source.read()


async def _extract_file_from_zip(
    zip_ref: zipfile.ZipFile,
    file_path: str,
    target_path: Path,
    common_prefix: str | None = None,
) -> bool:
    """Extract a single file from ZIP to target path.

    Args:
        zip_ref: Open ZIP file reference
        file_path: Path to file within ZIP
        target_path: Target directory for extraction
        common_prefix: Common prefix to strip from paths

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        # Check if file should be extracted
        if not _should_extract_file(file_path):
            return False

        # Prepare and validate target path
        final_target_path = _prepare_target_path(file_path, target_path, common_prefix)
        if not final_target_path:
            return False

        # Extract file content
        final_target_path.parent.mkdir(parents=True, exist_ok=True)
        success = await _extract_file_content(zip_ref, file_path, final_target_path)

        if success:
            logger.debug("Extracted {} to {}", file_path, final_target_path)

        return success
    except Exception as e:
        logger.error("Error extracting {}: {}", file_path, e)
        return False


def _find_common_prefix(file_paths: list[str]) -> str | None:
    """Find common prefix for file paths.

    Args:
        file_paths: List of file paths to analyze

    Returns:
        Common prefix string or None if no common prefix found
    """
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


def _should_extract_file(file_path: str) -> bool:
    """Check if file should be extracted.

    Args:
        file_path: Path to file within ZIP

    Returns:
        True if file should be extracted, False otherwise
    """
    if file_path.endswith("/"):
        return False

    has_valid_extension = Path(file_path).suffix.lower() in ALLOWED_EXTENSIONS
    if not has_valid_extension:
        logger.debug(MSG_INVALID_EXTENSION, file_path)

    return has_valid_extension


def _prepare_target_path(
    file_path: str, target_path: Path, common_prefix: str | None
) -> Path | None:
    """Prepare and validate target path for extraction.

    Args:
        file_path: Path to file within ZIP
        target_path: Target directory for extraction
        common_prefix: Common prefix to strip from paths

    Returns:
        Final target path or None if invalid
    """
    # Determine relative path
    relative_file_path = (
        file_path[len(common_prefix) + 1 :]
        if common_prefix and file_path.startswith(common_prefix + "/")
        else file_path
    )

    # Sanitize and validate path
    sanitized_name = _sanitize_path(relative_file_path)
    if not sanitized_name:
        logger.warning(MSG_UNSAFE_PATH, file_path)
        return None

    final_target_path = target_path / sanitized_name

    # Final path validation
    if not _is_safe_path(final_target_path, target_path):
        logger.error(MSG_PATH_TRAVERSAL, file_path)
        return None

    return final_target_path


def _validate_and_extract_file(
    file_info: zipfile.ZipInfo, model_dir: Path
) -> Path | None:
    """Validate and extract a single file from ZIP.

    Args:
        file_info: ZIP file info object
        model_dir: Target model directory

    Returns:
        Path to extracted file or None if validation/extraction failed
    """
    if file_info.is_dir() or file_info.file_size <= 0:
        return None

    file_extension = Path(file_info.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.debug(MSG_INVALID_EXTENSION, file_info.filename)
        return None

    sanitized_name = _sanitize_path(file_info.filename)
    if not sanitized_name:
        logger.warning(MSG_UNSAFE_PATH, file_info.filename)
        return None

    target_path = model_dir / sanitized_name

    # Validate that the final path is safe and within model directory
    if not _is_safe_path(target_path, model_dir):
        logger.error(MSG_PATH_TRAVERSAL, file_info.filename)
        return None

    return target_path


async def _extract_file_content(
    zip_ref: zipfile.ZipFile, filename: str, target_path: Path
) -> bool:
    """Extract file content from ZIP to target path.

    Args:
        zip_ref: ZIP file reference
        filename: File name in ZIP
        target_path: Target path for extraction

    Returns:
        True if successful, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        file_content = await loop.run_in_executor(
            None, _extract_zip_file_sync, zip_ref, filename
        )

        async with aiofiles.open(target_path, "wb") as target:
            await target.write(file_content)
        return True
    except (ValueError, OSError) as e:
        logger.error("Error extracting file '{}': {}", filename, e)
        return False


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
        ModelAlreadyExistsError: If model with same tag already exists
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
                relative_file_path = file_path[len(common_prefix) + 1 :]
            else:
                relative_file_path = file_path

            sanitized_name = _sanitize_path(relative_file_path)
            if sanitized_name:
                extracted_files.append(model_dir / sanitized_name)

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
        ModelAlreadyExistsError: If model with same tag already exists
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
        target_path = _validate_and_extract_file(file_info, model_dir)
        if target_path and await _extract_file_content(
            zip_ref, file_info.filename, target_path
        ):
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
