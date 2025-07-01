"""Exception classes for model upload handling."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode
from vectorize.utils.file_size_formatter import format_file_size


class InvalidModelError(AppError):
    """Exception raised when the model file is not a valid PyTorch model."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid PyTorch model file"
    status_code = status.HTTP_400_BAD_REQUEST


class EmptyModelError(AppError):
    """Exception raised when the model file is empty."""

    error_code = ErrorCode.EMPTY_FILE
    message = "Empty model file"
    status_code = status.HTTP_400_BAD_REQUEST


class ModelTooLargeError(AppError):
    """Exception raised when the model file is too large."""

    error_code = ErrorCode.INVALID_FILE
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, size: int) -> None:
        """Initialize with the size of the file."""
        formatted_size = format_file_size(size)
        super().__init__(f"Model file is too large: {formatted_size}")


class InvalidZipError(AppError):
    """Exception raised when a ZIP file is corrupted or invalid."""

    error_code = ErrorCode.INVALID_FILE
    message = "Invalid or corrupted ZIP file"
    status_code = status.HTTP_400_BAD_REQUEST


class NoValidModelsFoundError(AppError):
    """Exception raised when no valid PyTorch models were found in the upload."""

    error_code = ErrorCode.INVALID_FILE
    message = "No valid PyTorch model files found"
    status_code = status.HTTP_400_BAD_REQUEST


class ModelAlreadyExistsError(AppError):
    """Exception raised when the model already exists in the database."""

    error_code = ErrorCode.MODEL_ALREADY_EXISTS
    status_code = status.HTTP_409_CONFLICT

    def __init__(self, model_tag: str) -> None:
        """Initialize with the model tag."""
        super().__init__(f"Model with tag '{model_tag}' already exists.")


class InvalidUrlError(AppError):
    """Exception raised when the url is invalid."""

    error_code = ErrorCode.INVALID_URL
    message = "Invalid GitHub URL"
    status_code = status.HTTP_400_BAD_REQUEST


class ModelNotFoundError(AppError):
    """Exception raised when no model is found."""

    error_code = ErrorCode.NOT_FOUND
    message = "Model not found"
    status_code = status.HTTP_404_NOT_FOUND
