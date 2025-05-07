"""Service for extracting and processing ZIP model archives."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model import AIModel, ModelSource
from txt2vec.ai_model.repository import save_ai_model
from txt2vec.config.config import settings
from txt2vec.upload.exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidZipError,
    ModelTooLargeError,
    NoValidModelsFoundError,
)

__all__ = ["upload_zip_model"]


async def _save_zip_to_temp(file: UploadFile) -> Path:
    """Save ZIP file to a temporary location and validate it."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    # Write file in chunks to avoid memory issues
    size = 0
    with Path.open(temp_path, "wb") as dest_file:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > settings.model_max_upload_size:
                Path.unlink(temp_path)
                raise ModelTooLargeError(size)
            dest_file.write(chunk)

    if not zipfile.is_zipfile(temp_path):
        Path.unlink(temp_path)
        raise InvalidZipError("File is not a valid ZIP archive")

    return temp_path


async def upload_zip_model(
    file: UploadFile,
    model_name: str,
    db: AsyncSession,
) -> dict:
    """Extract a ZIP archive and save the model in the database.

    Args:
        file: The uploaded ZIP file
        model_name: Name for the model (used as directory name)
        db: Database session for persistence

    Returns:
        Dictionary with information about the uploaded model

    Raises:
        InvalidModelError: When the file is not a valid ZIP archive
        EmptyModelError: When the ZIP archive is empty
        ModelTooLargeError: When the file is too large
        InvalidZipError: When the ZIP file is corrupted
        NoValidModelsFoundError: When no valid files were found in the archive
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise InvalidModelError("Only ZIP archives are supported")

    if model_name is None:
        model_name = Path(file.filename).stem

    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    model_dir = Path(settings.model_upload_dir) / safe_model_name
    temp_path = None
    extracted_files = []

    try:
        # Save and validate the ZIP file
        temp_path = await _save_zip_to_temp(file)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Extract the ZIP contents
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            if not file_list:
                raise EmptyModelError("ZIP archive is empty")

            # Extract valid files
            for file_info in file_list:
                if not file_info.is_dir() and file_info.file_size > 0:
                    target_name = Path(file_info.filename).name
                    target_path = model_dir / target_name

                    with (
                        zip_ref.open(file_info) as source,
                        Path.open(target_path, "wb") as target,
                    ):
                        shutil.copyfileobj(source, target)

                    extracted_files.append(target_path)

        if not extracted_files:
            raise NoValidModelsFoundError("No valid files found in the archive")

        # Register model in database
        ai_model = AIModel(
            model_tag=safe_model_name,
            name=model_name,
            source=ModelSource.LOCAL,
        )
        db_model_id = await save_ai_model(db, ai_model)

        return {
            "model_id": str(db_model_id),
            "model_name": safe_model_name,
            "model_dir": str(model_dir),
            "file_count": len(extracted_files),
        }

    except (
        ModelTooLargeError,
        InvalidModelError,
        EmptyModelError,
        InvalidZipError,
        NoValidModelsFoundError,
    ):
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise

    except Exception as e:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise InvalidModelError(f"Error processing upload: {e!s}") from e

    finally:
        # Clean up temporary file
        if temp_path and Path(temp_path).exists():
            Path.unlink(temp_path)
