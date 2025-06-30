"""Evaluation exceptions."""

from fastapi import status

from vectorize.common.app_error import AppError
from vectorize.config.errors import ErrorCode

__all__ = ["EvaluationModelNotFoundError", "EvaluationTaskNotFoundError"]


class EvaluationModelNotFoundError(AppError):
    """Raised when a baseline or evaluation model is not found for evaluation."""

    def __init__(self, model_tag: str) -> None:
        """Initialize exception.

        Args:
            model_tag: The model tag that was not found
        """
        super().__init__(f"Evaluation model not found: {model_tag}")
        self.error_code = ErrorCode.NOT_FOUND
        self.status_code = status.HTTP_404_NOT_FOUND


class EvaluationTaskNotFoundError(AppError):
    """Raised when evaluation task is not found."""

    def __init__(self, task_id: str) -> None:
        """Initialize exception.

        Args:
            task_id: The task ID that was not found
        """
        super().__init__(f"Evaluation task not found: {task_id}")
        self.error_code = ErrorCode.NOT_FOUND
        self.status_code = status.HTTP_404_NOT_FOUND
