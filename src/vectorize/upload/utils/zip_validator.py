"""Validation utilities for model ZIP files."""

import zipfile
from collections import defaultdict
from pathlib import Path

from loguru import logger

__all__ = ["get_toplevel_directories", "is_valid_zip", "validate_model_files"]

_EXT_PT = ".pt"
_EXT_PTH = ".pth"
_EXT_BIN = ".bin"
_EXT_MODEL = ".model"
_EXT_SAFETENSORS = ".safetensors"

_VALID_EXTENSIONS = {_EXT_PT, _EXT_PTH, _EXT_BIN, _EXT_MODEL, _EXT_SAFETENSORS}
_MODEL_EXTS = (_EXT_PT, _EXT_PTH, _EXT_BIN, _EXT_MODEL, _EXT_SAFETENSORS)

_MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06")
_PICKLE_PROTOCOL_MARKER = 0x80
_PICKLE_PROTOCOLS = {2, 3, 4, 5}
_MIN_HEADER_LENGTH = 2
_SAFETENSORS_MIN_HEADER = 8


def validate_model_files(extracted_files: list[Path]) -> bool:
    """Check if at least one valid PyTorch model exists in the list of files.

    Args:
        extracted_files: List of paths to extracted files

    Returns:
        bool: True if at least one valid PyTorch model was found, False otherwise
    """
    for file_path in extracted_files:
        if file_path.suffix.lower() in _VALID_EXTENSIONS and _is_valid_model_file(
            file_path
        ):
            return True
    return False


def _is_valid_model_file(file_path: Path) -> bool:
    """Check if a single file is a valid model file.

    Args:
        file_path: Path to the file to validate

    Returns:
        bool: True if file is a valid model, False otherwise
    """
    try:
        stat = file_path.stat()
        if not _is_valid_file_size(stat.st_size, file_path):
            return False

        suffix = file_path.suffix.lower()
        return _validate_by_extension(file_path, suffix)

    except Exception as e:
        logger.debug("Error validating model file: {}, error: {}", file_path, e)
        return False


def _is_valid_file_size(size: int, file_path: Path) -> bool:
    """Check if file size is valid."""
    if size <= 0:
        return False

    if size > _MAX_FILE_SIZE:
        logger.debug("Model file too large: {}, size: {} bytes", file_path, size)
        return False

    return True


def _validate_by_extension(file_path: Path, suffix: str) -> bool:
    """Validate file based on its extension."""
    if suffix in {_EXT_PT, _EXT_PTH}:
        return _validate_pytorch_file(file_path)
    if suffix == _EXT_SAFETENSORS:
        return _validate_safetensors_file(file_path)
    return suffix in {_EXT_BIN, _EXT_MODEL}


def _validate_pytorch_file(file_path: Path) -> bool:
    """Validate PyTorch file format without deserialization."""
    try:
        with file_path.open("rb") as f:
            header = f.read(8)
            if header.startswith(_ZIP_SIGNATURES):
                return True
            if (
                len(header) >= _MIN_HEADER_LENGTH
                and header[0] == _PICKLE_PROTOCOL_MARKER
                and header[1] in _PICKLE_PROTOCOLS
            ):
                return True
    except Exception as e:
        logger.debug("Error reading PyTorch file header: {}, error: {}", file_path, e)
    return False


def _validate_safetensors_file(file_path: Path) -> bool:
    """Validate SafeTensors file format."""
    try:
        with file_path.open("rb") as f:
            header = f.read(16)
            return len(header) >= _SAFETENSORS_MIN_HEADER
    except Exception as e:
        logger.debug(
            "Error reading SafeTensors file header: {}, error: {}", file_path, e
        )
    return False


def is_valid_zip(file_path: Path) -> bool:
    """Check if a file is a valid ZIP archive.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if file is a valid ZIP archive, False otherwise
    """
    return zipfile.is_zipfile(file_path)


def get_toplevel_directories(zip_file: zipfile.ZipFile) -> dict[str, list[str]]:
    """Group ZIP entries by top-level model directory or first path segment.

    Args:
        zip_file: An opened ``zipfile.ZipFile`` instance.

    Returns:
        Dictionary mapping directory names to lists of file paths inside each
        directory.
    """
    all_files = zip_file.namelist()

    model_dirs = _collect_model_dirs(all_files)
    if model_dirs:
        return {
            md: [f for f in all_files if f.startswith(f"{md}/")] for md in model_dirs
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for path in all_files:
        if "/" in path:
            grouped[path.split("/", 1)[0]].append(path)
    return dict(grouped)


def _collect_model_dirs(all_files: list[str]) -> set[str]:
    """Return top-level directories that contain at least one model file.

    Args:
        all_files: List of paths obtained from ``zipfile.ZipFile.namelist``.

    Returns:
        Set of directory paths containing model files.
    """
    candidates = {
        "/".join(p.split("/")[:-1])
        for p in all_files
        if "/" in p and p.lower().endswith(_MODEL_EXTS)
    }

    return {
        d
        for d in candidates
        if not any(d != other and d.startswith(f"{other}/") for other in candidates)
    }
