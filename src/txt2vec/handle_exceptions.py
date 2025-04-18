"""Central exception handling for the application."""

from collections.abc import Callable
from enum import StrEnum
from functools import wraps
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

__all__ = ["AppError", "ErrorCode", "handle_exceptions"]


class ErrorCode(StrEnum):
    """Error codes for standardized error handling."""

    # General errors
    SERVER_ERROR = "SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"

    # Dataset errors
    INVALID_FILE = "INVALID_FILE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_CSV_FORMAT = "INVALID_CSV_FORMAT"
    EMPTY_CSV = "EMPTY_CSV"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class AppError(Exception):
    """Base exception for application errors."""

    error_code = ErrorCode.SERVER_ERROR
    message = "An unexpected error occurred"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None) -> None:
        """Initialize with optional custom message."""
        if message:
            self.message = message
        super().__init__(self.message)


def handle_exception(exception: Exception) -> HTTPException:
    """Convert any exception to appropriate HTTPException.

    This function processes exceptions and transforms them into standardized
    HTTPException objects with appropriate status codes and error details.

    The function handles three types of exceptions:
    1. HTTPException: Passed through unchanged
    2. AppError: Converted to HTTPException with the app error details
    3. Other exceptions: Converted to 500 Internal Server Error

    :param exception: The exception to convert
    :return: An HTTPException with standardized error format
    """
    if isinstance(exception, HTTPException):
        logger.error("HTTPException: {} - {}", exception.status_code, exception.detail)
        return exception

    if isinstance(exception, AppError):
        logger.warning(
            "Application exception: {} - {}", exception.error_code, exception.message
        )
        return HTTPException(
            status_code=exception.status_code,
            detail={"code": exception.error_code, "message": exception.message},
        )

    # Handle unexpected exceptions
    logger.error(
        "Unhandled exception: {} - {}", type(exception).__name__, str(exception)
    )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "code": ErrorCode.SERVER_ERROR,
            "message": "An unexpected server error occurred",
        },
    )


def handle_exceptions(func: Callable) -> Callable:
    """Handle exceptions standardly across API endpoints.

    This decorator catches all exceptions from the endpoint function,
    logs them, and converts them to proper HTTPExceptions with
    standardized error formats.

    :param func: The API endpoint function to wrap
    :return: A wrapped function that handles exceptions
    :raises HTTPException: With appropriate status code and error details
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("{}", str(e))
            http_exception = handle_exception(e)
            raise http_exception from e

    return wrapper
