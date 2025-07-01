"""Service for extracting and processing ZIP model archives."""

import asyncio
import shutil
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import UploadFile
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.config.config import settings

from .exceptions import (
    EmptyModelError,
    InvalidModelError,
    InvalidZipError,
    ModelAlreadyExistsError,
    NoValidModelsFoundError,
)
from .utils.zip_extractor import (
    _is_safe_path,
    process_model_directory,
    process_single_model,
    save_zip_to_temp,
)
from .utils.zip_validator import get_toplevel_directories, is_valid_zip

__all__ = ["upload_zip_model"]


class ProcessingConfig(NamedTuple):
    """Configuration for ZIP model processing."""

    base_name: str
    base_dir: Path
    multi_model: bool


async def _process_directory(
    zip_ref: zipfile.ZipFile,
    dir_path: str,
    dir_files: list,
    base_dir: Path,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Process a single model directory from a ZIP archive."""
    model_folder_name = Path(dir_path).name
    logger.debug("Processing model directory: {} as {}", dir_path, model_folder_name)

    return await process_model_directory(
        zip_ref, model_folder_name, dir_files, base_dir, db
    )


MODEL_ALREADY_EXISTS_MSG = "Model '{}' already exists"


def _handle_model_already_exists(dir_path: str, existing_models: list[str]) -> None:
    """Handle ModelAlreadyExistsError for a directory."""
    model_folder_name = dir_path.split("/", 1)[-1] if "/" in dir_path else dir_path
    existing_models.append(model_folder_name)
    logger.warning(MODEL_ALREADY_EXISTS_MSG, model_folder_name)


def _handle_invalid_model_directory(
    dir_path: str, base_dir: Path, e: NoValidModelsFoundError
) -> None:
    """Handle NoValidModelsFoundError for a directory."""
    model_folder_name = dir_path.split("/", 1)[-1] if "/" in dir_path else dir_path
    logger.warning("Skipping invalid model directory '{}': {}", dir_path, e)

    sanitized_name = Path(model_folder_name).name
    if not sanitized_name or sanitized_name in {".", ".."}:
        logger.warning("Invalid model folder name: '{}'", model_folder_name)
        return

    model_path = base_dir / sanitized_name
    if model_path.exists():
        shutil.rmtree(model_path, ignore_errors=True)


async def _process_directories(
    zip_ref: zipfile.ZipFile,
    model_dirs: dict[str, list],
    base_dir: Path,
    db: AsyncSession,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Process multiple directories from a ZIP archive.

    Args:
        zip_ref: Open ZIP file reference
        model_dirs: Dictionary of directory paths to their files
        base_dir: Base directory for model storage
        db: Database session

    Returns:
        Tuple containing (processed_models, existing_models)
    """
    processed_models = []
    existing_models = []

    for dir_path, dir_files in model_dirs.items():
        try:
            model_dir, model_id = await _process_directory(
                zip_ref, dir_path, dir_files, base_dir, db
            )
            processed_models.append((model_dir, model_id))

        except ModelAlreadyExistsError:
            _handle_model_already_exists(dir_path, existing_models)

        except NoValidModelsFoundError as e:
            _handle_invalid_model_directory(dir_path, base_dir, e)

    return processed_models, existing_models


async def _process_multi_model(
    zip_ref: zipfile.ZipFile,
    file_list: list,
    base_name: str,
    base_dir: Path,
    db: AsyncSession,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Process ZIP as multiple models."""
    model_dirs = get_toplevel_directories(zip_ref)
    processed_models = []
    existing_models = []

    if not model_dirs:
        logger.info("No directories found, processing as single model")
        try:
            result = await process_single_model(
                zip_ref, base_name, file_list, base_dir, db
            )
            processed_models.append(result)
        except ModelAlreadyExistsError:
            existing_models.append(base_name)
            logger.warning(MODEL_ALREADY_EXISTS_MSG, base_name)

        return processed_models, existing_models

    processed, existing = await _process_directories(zip_ref, model_dirs, base_dir, db)
    return processed, existing


async def _cleanup_models(processed_models: list[tuple[Path, str]]) -> None:
    """Clean up model directories in case of failure."""
    for model_dir, _ in processed_models:
        if Path(model_dir).exists():
            await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=True)


def _handle_results(
    processed_models: list[tuple[Path, str]],
    existing_models: list[str],
    base_name: str,
    multi_model: bool,
) -> dict[str, Any]:
    """Handle processing results and generate response.

    Args:
        processed_models: List of processed model directories and IDs
        existing_models: List of models that already exist
        base_name: Base name for the model
        multi_model: Whether multi-model mode was used

    Returns:
        Result dictionary with model information

    Raises:
        ModelAlreadyExistsError: If no models were processed but some already exist
        NoValidModelsFoundError: If no models were processed or exist
    """
    if not processed_models:
        if existing_models:
            msg = (
                MODEL_ALREADY_EXISTS_MSG.format(base_name)
                if not multi_model
                else f"All models already exist: {', '.join(existing_models)}"
            )
            raise ModelAlreadyExistsError(msg) from None
        raise NoValidModelsFoundError(
            "No valid model directories found in the archive"
        ) from None

    result = {
        "models": [
            {
                "model_tag": Path(model_dir).name,
                "model_name": Path(model_dir).name,
                "model_dir": str(model_dir),
            }
            for model_dir, model_id in processed_models
        ],
        "total_models": len(processed_models),
    }

    if existing_models:
        result["existing_models"] = existing_models

    return result


def _validate_upload_params(file: UploadFile, model_name: str | None) -> str:
    """Validate upload parameters and return base name."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise InvalidModelError("Only ZIP archives are supported")

    base_name = "".join(
        c if c.isalnum() else "_" for c in (model_name or Path(file.filename).stem)
    ).strip("_")
    base_dir = Path(settings.model_upload_dir).resolve()
    if not base_name or not _is_safe_path(base_dir / base_name, base_dir):
        raise InvalidModelError("Unsafe model name or path detected")

    return base_name


async def _process_zip_models(
    zip_ref: zipfile.ZipFile,
    file_list: list,
    config: ProcessingConfig,
    db: AsyncSession,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Process models from ZIP file."""
    existing_models = []
    processed_models = []

    if config.multi_model:
        processed_models, existing_models = await _process_multi_model(
            zip_ref, file_list, config.base_name, config.base_dir, db
        )
    else:
        try:
            result = await process_single_model(
                zip_ref, config.base_name, file_list, config.base_dir, db
            )
            processed_models.append(result)
        except ModelAlreadyExistsError as err:
            existing_models.append(config.base_name)
            raise ModelAlreadyExistsError(
                MODEL_ALREADY_EXISTS_MSG.format(config.base_name)
            ) from err

    return processed_models, existing_models


async def upload_zip_model(
    file: UploadFile,
    model_name: str | None,
    db: AsyncSession,
    multi_model: bool = False,
) -> dict[str, Any]:
    """Extract a ZIP archive and save the contained model(s) in the database.

    Args:
        file: The uploaded ZIP file
        model_name: Base name for the model(s)
        db: Database session for persistence
        multi_model: Whether to treat top-level directories as separate models

    Returns:
        Dictionary with information about the uploaded model(s)

    Raises:
        InvalidModelError: When the file is not a valid ZIP archive
        EmptyModelError: When the ZIP archive is empty
        InvalidZipError: When the ZIP file is corrupted
        NoValidModelsFoundError: When no valid files were found
    """
    base_name = _validate_upload_params(file, model_name)
    base_dir = Path(settings.model_upload_dir).resolve()
    temp_path = None
    processed_models = []

    try:
        logger.debug("Processing ZIP model upload: {}", file.filename)
        temp_path = await save_zip_to_temp(file)

        if not is_valid_zip(temp_path):
            raise InvalidZipError("File is not a valid ZIP archive")

        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            file_list = zip_ref.infolist()

            if not file_list:
                raise EmptyModelError("ZIP archive is empty")

            config = ProcessingConfig(base_name, base_dir, multi_model)
            processed_models, existing_models = await _process_zip_models(
                zip_ref, file_list, config, db
            )

        return _handle_results(
            processed_models, existing_models, base_name, multi_model
        )

    except (
        InvalidModelError,
        EmptyModelError,
        InvalidZipError,
        NoValidModelsFoundError,
        ModelAlreadyExistsError,
    ) as e:
        logger.error("Error processing ZIP model: {}", str(e))
        await _cleanup_models(processed_models)
        raise

    except Exception as e:
        logger.exception("Unexpected error during ZIP model upload")
        await _cleanup_models(processed_models)
        raise InvalidModelError(f"Error processing upload: {e!s}") from e

    finally:
        if temp_path and Path(temp_path).exists():
            Path.unlink(temp_path)
