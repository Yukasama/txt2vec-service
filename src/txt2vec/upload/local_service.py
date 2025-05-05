"""Service for handling PyTorch model uploads including ZIP files."""

import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Final

import torch
from fastapi import UploadFile
from loguru import logger

from txt2vec.config.config import settings
from txt2vec.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidZipError,
    ModelTooLargeError,
    NoValidModelsFoundError,
    UnsupportedModelFormatError,
)

# Valid PyTorch model file extensions
PYTORCH_EXTENSIONS: Final[set[str]] = {
    ".pt",
    ".pth",
    ".bin",
    ".model",
    ".weights",
    ".json",
    ".safetensors",
}


async def upload_embedding_model(
    files: list[UploadFile],
    model_name: str,
    extract_zip: bool = True,
) -> dict[str, Any]:
    """Process PyTorch model uploads with ZIP support.

    Args:
        files: List of model files to be uploaded
        model_name: Name for the model (used as directory name)
        extract_zip: Whether ZIP files should be extracted (default: True)

    Returns:
        Dictionary with information about the uploaded model

    Raises:
        InvalidModelError: When no or invalid files are provided
        EmptyModelError: When a file is empty
        ModelTooLargeError: When a file is too large
        UnsupportedModelFormatError: When the file format is not supported
        InvalidZipError: When a ZIP file is invalid
        NoValidModelsFoundError: When no valid PyTorch models were found
    """
    if not files:
        raise InvalidModelError("No files provided for upload")

    # Sanitize model name using simple regex for safety
    safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)

    model_id = uuid.uuid4()
    model_dir = Path(settings.model_upload_dir) / f"{safe_model_name}_{model_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        pytorch_file_count = await _process_uploaded_files(
            files, model_dir, extract_zip
        )

        if pytorch_file_count == 0:
            raise NoValidModelsFoundError(
                "No valid PyTorch model files found in the upload"
            )

        return {
            "model_id": str(model_id),
            "model_name": safe_model_name,
            "model_dir": str(model_dir),
            "file_count": pytorch_file_count,
        }

    except (
        ModelTooLargeError,
        InvalidModelError,
        EmptyModelError,
        UnsupportedModelFormatError,
        InvalidZipError,
        NoValidModelsFoundError,
    ):
        _cleanup_model_dir(model_dir)
        raise
    except Exception as e:
        _cleanup_model_dir(model_dir)
        logger.exception("Unexpected error during model upload")
        raise InvalidModelError("Error processing upload") from e


async def _process_uploaded_files(
    files: list[UploadFile],
    model_dir: Path,
    extract_zip: bool,
) -> int:
    """Process each uploaded file and count valid PyTorch models.

    Args:
        files: List of files to process
        model_dir: Directory to save models to
        extract_zip: Whether to extract ZIP files

    Returns:
        int: Number of valid PyTorch models processed

    Raises:
        Various exceptions based on file validation errors
    """
    pytorch_file_count = 0

    for file in files:
        if not file.filename:
            logger.warning("File without filename skipped")
            continue

        temp_path = None
        try:
            file_size, temp_path = await _process_single_file(file)
            if file_size <= 1:
                raise EmptyModelError(f"File '{file.filename}' is empty")

            filename = Path(file.filename).name
            file_ext = Path(filename).suffix.lower()

            if extract_zip and filename.lower().endswith(".zip"):
                pytorch_file_count += _handle_zip_file(temp_path, model_dir)
            elif file_ext in PYTORCH_EXTENSIONS:
                pytorch_file_count += _handle_pytorch_file(
                    temp_path, model_dir, filename
                )
            else:
                raise UnsupportedModelFormatError(
                    f"File '{file.filename}' has unsupported format. "
                    f"Supported: {', '.join(PYTORCH_EXTENSIONS)}"
                )
        except (ModelTooLargeError, EmptyModelError, UnsupportedModelFormatError) as e:
            # Log the error, but explicitly pass the original exception
            logger.error("Error processing file {}: {}", file.filename, str(e))
            # Important: Cleanup before raising
            if temp_path and Path(temp_path).exists():
                try:
                    Path.unlink(temp_path)
                except Exception:
                    logger.warning("Failed to delete temporary file: {}", temp_path)
            raise  # Explicit re-raise of the original exception
        finally:
            # Only cleanup the temporary file if it still exists and wasn't handled
            if temp_path and Path(temp_path).exists() and "e" not in locals():
                try:
                    Path.unlink(temp_path)
                except Exception:
                    logger.warning("Failed to delete temporary file: {}", temp_path)
            await file.seek(0)

    return pytorch_file_count


async def _process_single_file(file: UploadFile) -> tuple[int, str]:
    """Process a single uploaded file into a temporary file.

    Args:
        file: The file to process

    Returns:
        tuple: File size and path to temporary file

    Raises:
        ModelTooLargeError: If file exceeds size limit
        InvalidModelError: For other processing errors
    """
    logger.debug("Starting to process file: {}", file.filename)
    file_size = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                logger.debug(
                    "Read chunk, total size now: {}/{}",
                    file_size,
                    settings.max_upload_size,
                )
                if file_size > settings.max_upload_size:
                    logger.warning(
                        "File size ({}) exceeds limit ({})",
                        file_size,
                        settings.max_upload_size,
                    )
                    raise ModelTooLargeError(file_size)
                temp.write(chunk)
            logger.debug(
                "Completed processing file: {}, size: {}", file.filename, file_size
            )
            return file_size, temp_path
    except ModelTooLargeError:
        # For size errors, delete the temp file and pass the error
        if temp_path and Path(temp_path).exists():
            Path.unlink(temp_path)
        raise  # Important: pass the original ModelTooLargeError
    except Exception as e:
        # For other errors, delete the temp file and raise a general error
        if temp_path and Path(temp_path).exists():
            Path.unlink(temp_path)
        logger.exception("Error processing file {}", file.filename)
        raise InvalidModelError(f"Error processing file {file.filename}: {e!s}") from e


def _handle_zip_file(zip_path: str, model_dir: Path) -> int:
    """Handle ZIP file extraction and return count of valid models.

    Args:
        zip_path: Path to the ZIP file
        model_dir: Directory to extract models to

    Returns:
        int: Number of valid PyTorch models extracted

    Raises:
        InvalidZipError: If the ZIP file is invalid
    """
    if not zipfile.is_zipfile(zip_path):
        logger.warning("Invalid ZIP file: {}", zip_path)
        raise InvalidZipError("File is not a valid ZIP file")

    extracted = _extract_pytorch_models(zip_path, model_dir)
    logger.debug("Extracted {} PyTorch models from ZIP", extracted)
    return extracted


def _handle_pytorch_file(file_path: str, model_dir: Path, filename: str) -> int:
    """Handle PyTorch model file and return 1 if valid, otherwise raise.

    Args:
        file_path: Path to the PyTorch file
        model_dir: Directory to save the model to
        filename: Filename to use for the model

    Returns:
        int: 1 if model is valid, 0 otherwise

    Raises:
        InvalidModelError: If the file is not a valid PyTorch model
    """
    if _is_valid_pytorch_model(file_path):
        dest_path = model_dir / filename
        shutil.move(file_path, dest_path)
        logger.debug("Saved valid PyTorch model: {}", filename)
        return 1
    logger.warning("Invalid PyTorch model: {}", filename)
    raise InvalidModelError("File is not a valid PyTorch model")


def _cleanup_model_dir(model_dir: Path) -> None:
    """Clean up the model directory if it exists.

    Args:
        model_dir: Directory to clean up
    """
    if Path(model_dir).exists():
        shutil.rmtree(model_dir)


def _is_valid_pytorch_model(file_path: str) -> bool:
    """Check if a file is a valid PyTorch model.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if file is a valid PyTorch model, False otherwise
    """
    try:
        torch.load(file_path, map_location="cpu")
        return True
    except Exception as e:
        logger.debug("Invalid PyTorch model: {}", str(e))
        return False


def _extract_pytorch_models(zip_path: str, extract_to: Path) -> int:
    """Extract only valid PyTorch model files from a ZIP file.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract models to

    Returns:
        int: Number of valid PyTorch models extracted
    """
    count = 0

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir() or file_info.file_size == 0:
                continue

            file_ext = Path(file_info.filename).suffix.lower()
            if file_ext in PYTORCH_EXTENSIONS:
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp_path = temp.name

                    with (
                        zip_ref.open(file_info) as source,
                        Path.open(temp_path, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)

                    if _is_valid_pytorch_model(temp_path):
                        target_path = extract_to / Path(file_info.filename).name
                        shutil.move(temp_path, target_path)
                        count += 1
                        logger.debug("Extracted valid model: {}", file_info.filename)
                finally:
                    if temp_path and Path(temp_path).exists():
                        Path.unlink(temp_path)

    return count
