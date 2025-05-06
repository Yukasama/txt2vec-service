"""Router for model upload and management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model.model_source import ModelSource
from txt2vec.common.status import TaskStatus
from txt2vec.config.db import get_session
from txt2vec.datasets.exceptions import InvalidFileError
from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.local_service import upload_embedding_model
from txt2vec.upload.models import UploadTask
from txt2vec.upload.repository import save_upload_task  # muss angelegt werden
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest
from txt2vec.upload.tasks import process_huggingface_model_background

router = APIRouter(tags=["Model Upload"])


@router.post("/v1/upload/huggingface", status_code=status.HTTP_201_CREATED)
async def load_model_huggingface(
    data: HuggingFaceModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    key = f"{data.model_id}@{data.tag}"

    # Task anlegen
    upload_task = UploadTask(
        model_tag=key,
        task_status=TaskStatus.PENDING,
        source=ModelSource.HUGGINGFACE,
    )
    await save_upload_task(db, upload_task)

    # Hintergrundprozess starten
    background_tasks.add_task(
        process_huggingface_model_background, data.model_id, data.tag, upload_task.id
    )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.base_url}v1/upload/tasks/{upload_task.id}"},
    )


@router.post("/add_model")
async def load_model_github(request: GitHubModelRequest) -> Response:
    """Download and register a model from a specified GitHub repository.

    This endpoint accepts a GitHub repository URL and attempts to download
    and prepare the model files for use. If successful, a JSON response is returned.

    Args:
        request: Contains the GitHub repository URL.

    Returns:
        A response indicating success or error details.

    Raises:
        HTTPException:
            - 400 if the GitHub URL is invalid.
            - 500 if an unexpected error occurs during model processing.

    """
    logger.info(
        "Received request to add model from GitHub URL: {}",
        request.github_url,
    )

    result = await handle_model_download(request.github_url)
    logger.info(
        "Model handled successfully for: {}",
        request.github_url,
    )
    return result


@router.post("/models")
async def load_model_local(
    files: list[UploadFile],
    request: Request,
    model_name: Annotated[str, Query(description="Name for the uploaded model")],
    extract_zip: Annotated[
        bool,
        Query(description="Whether to extract ZIP files"),
    ] = True,
) -> Response:
    """Upload PyTorch model files to the server.

    Args:
        request: The HTTP request object.
        model_name: Name to assign to the uploaded model.
        extract_zip: Whether to extract ZIP files (default: True).
        files: The uploaded model files.

    Returns:
        A 201 Created response with a Location header.

    Raises:
        HTTPException: If an error occurs during file upload or processing.

    """
    if not files:
        raise InvalidFileError

    logger.debug(
        "Uploading model '{}' with {} files",
        model_name,
        len(files),
    )

    result = await upload_embedding_model(files, model_name, extract_zip)
    logger.info(
        "Successfully uploaded model: {}",
        result["model_dir"],
    )

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{result['model_id']}"},
    )
