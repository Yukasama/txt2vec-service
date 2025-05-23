"""Router for model upload and management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from huggingface_hub import model_info
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from txt2vec.ai_model.exceptions import ModelNotFoundError
from txt2vec.ai_model.model_source import ModelSource
from txt2vec.ai_model.service import get_ai_model_svc
from txt2vec.common.exceptions import InternalServerError
from txt2vec.common.task_status import TaskStatus
from txt2vec.config.db import get_session
from txt2vec.datasets.exceptions import InvalidFileError
from txt2vec.upload.exceptions import ModelAlreadyExistsError
from txt2vec.upload.github_service import handle_model_download
from txt2vec.upload.models import UploadTask
from txt2vec.upload.repository import save_upload_task_db
from txt2vec.upload.schemas import GitHubModelRequest, HuggingFaceModelRequest
from txt2vec.upload.tasks import process_huggingface_model_background
from txt2vec.upload.zip_service import upload_zip_model

router = APIRouter(tags=["Model Upload"])


@router.post("/huggingface")
async def load_model_huggingface(
    data: HuggingFaceModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Upload a Hugging Face model by model_tag and revision.

    Checks if the model already exists, verifies its presence on
    Hugging Face, creates an upload task, and starts background
    processing. Returns a 201 response with a Location header.

    Args:
        data: Model tag and revision for Hugging Face.
        request: FastAPI request object.
        background_tasks: FastAPI background task manager.
        db: Async database session.

    Returns:
        Response with 201 status and Location header.

    Raises:
        ModelAlreadyExistsError: If the model already exists in the database.
        ModelNotFoundError: If the model is not found on Hugging Face.
        InternalServerError: If an internal error occurs while checking the model.
    """
    key = f"{data.model_tag}@{data.revision}"

    try:
        await get_ai_model_svc(db, key)
        raise ModelAlreadyExistsError(key)
    except ModelNotFoundError:
        pass

    try:
        model_info(repo_id=data.model_tag, revision=data.revision)
    except (EntryNotFoundError, HfHubHTTPError) as e:
        raise ModelNotFoundError(data.model_tag, data.revision) from e
    except Exception as e:
        raise InternalServerError(
            "Internal server error while checking model on Hugging Face.",
        ) from e

    upload_task = UploadTask(
        model_tag=key,
        task_status=TaskStatus.PENDING,
        source=ModelSource.HUGGINGFACE,
    )
    await save_upload_task_db(db, upload_task)

    background_tasks.add_task(
        process_huggingface_model_background,
        db,
        data.model_tag,
        data.revision,
        upload_task.id,
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


@router.post("/local_models", summary="Upload multiple models from a ZIP archive")
async def load_multiple_models(
    file: Annotated[UploadFile, File()],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    model_name: Annotated[
        str | None, Query(description="Base name for models (optional)")
    ] = None,
) -> JSONResponse:
    """Upload a ZIP archive containing multiple model directories.

    Each top-level directory in the ZIP will be treated as a separate model.
    All extracted models will be registered in the database.

    Args:
        file: ZIP archive containing model directories
        request: HTTP request object
        model_name: Optional base name for models (used as fallback)
        db: Database session

    Returns:
        JSON response with metadata about uploaded models

    Raises:
        InvalidFileError: If no file is provided or format is invalid
        HTTPException: If an error occurs during processing
    """
    if not file:
        raise InvalidFileError("No file uploaded")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise InvalidFileError("Only ZIP archives are supported")

    logger.debug(
        "Multi-model ZIP upload: '{}' with {} bytes",
        file.filename,
        file.size if hasattr(file, "size") else "unknown size",
    )

    result = await upload_zip_model(file, model_name, db, multi_model=True)

    model_count = result["total_models"]
    logger.info(f"Successfully uploaded {model_count} models from ZIP archive")

    models_info = [
        {
            "id": model["model_id"],
            "name": model["model_name"],
            "directory": model["model_dir"],
            "url": f"{request.url.scheme}://{request.url.netloc}{request.url.path}/{model['model_id']}",
        }
        for model in result["models"]
    ]

    headers = {}
    if result["models"]:
        first_model = result["models"][0]
        headers["Location"] = f"{request.url}/{first_model['model_id']}"

    return JSONResponse(
        content={
            "message": f"Successfully uploaded {model_count} models",
            "models": models_info,
        },
        status_code=status.HTTP_201_CREATED,
        headers=headers,
    )
