"""Dataset router."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.common.exceptions import InvalidFileError
from vectorize.config.db import get_session

from ..utils.pagination import Page
from .models import DatasetAll, DatasetPublic, DatasetUpdate
from .schemas import DatasetUploadOptions, HuggingFaceDatasetRequest
from .service import (
    delete_dataset_svc,
    get_dataset_svc,
    get_datasets_svc,
    get_hf_upload_status_svc,
    update_dataset_svc,
    upload_hf_dataset_svc,
)
from .task_model import UploadDatasetTask
from .utils.process_upload import _process_uploads
from .utils.validate_zip import _handle_zip_upload

__all__ = ["router"]


router = APIRouter(tags=["Dataset"])


@router.get("", summary="Get paginated datasets")
async def get_datasets(
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[DatasetAll]:
    """Retrieve all datasets with limited fields.

    Args:
        db: Database session for persistence operations
        limit: Maximum number of datasets to return (default 10, max 100)
        offset: Number of datasets to skip (default 0)

    Returns:
        List of datasets with limited fields (DatasetAll model)
    """
    items, total = await get_datasets_svc(db, limit=limit, offset=offset)
    logger.debug("Datasets retrieved", length=len(items), total=total)
    return Page[DatasetAll](items=items, total=total, limit=limit, offset=offset)


@router.get("/{dataset_id}", response_model=None, summary="Get dataset by ID")
async def get_dataset(
    dataset_id: UUID,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> DatasetPublic | Response:
    """Retrieve a single dataset by its ID.

    Args:
        dataset_id: The UUID of the dataset to retrieve
        request: The HTTP request object
        response: FastAPI response object for setting headers
        db: Database session for persistence operations

    Returns:
        The dataset object, or 304 Not Modified if the ETag matches the current version

    Raises:
        DatasetNotFoundError: If the dataset with the specified ID doesn't exist
    """
    dataset, version = await get_dataset_svc(db, dataset_id)
    response.headers["ETag"] = f'"{version}"'
    etag = f'"{version}"'

    client_match = request.headers.get("If-None-Match")
    if client_match and client_match.strip('"') == str(version):
        logger.debug("Dataset not modified", dataset_id=dataset_id, version=version)
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag}
        )

    logger.debug("Dataset retrieved", dataset_id=dataset_id, version=version)
    return dataset


@router.post("", summary="Upload dataset files")
async def upload_dataset(
    files: Annotated[list[UploadFile], File(description="One or many files")],
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    options: Annotated[DatasetUploadOptions, Depends()],
) -> Response:
    """Upload one or more dataset files and convert them to CSV format.

    Processes each file individually, allowing partial success when uploading
    multiple files. Any files that fail to upload will be listed in the response
    but won't cause the entire request to fail.

    Args:
        files: The files to upload (CSV, JSON, XML, Excel, or ZIP)
        request: The HTTP request object
        db: Database session for persistence operations
        options: Options for dataset upload, including column names and sheet index

    Returns:
        Response with status code 201 Created and the dataset ID in the Location header.
        If any files failed to upload, they will be listed in the response body.
    """
    if not files:
        raise InvalidFileError("No files provided")

    first = files[0]
    if len(files) == 1 and first.filename and first.filename.lower().endswith(".zip"):
        files_for_upload = await _handle_zip_upload(first)
    else:
        if any(f.filename and f.filename.lower().endswith(".zip") for f in files):
            raise InvalidFileError("Cannot mix ZIP and individual files")
        files_for_upload = files

    dataset_ids, failed_uploads = await _process_uploads(files_for_upload, db, options)

    if not dataset_ids:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"successful_uploads": 0, "failed": failed_uploads},
        )

    response_body = {}
    response_body["successful_uploads"] = len(dataset_ids)
    if failed_uploads:
        response_body["failed"] = failed_uploads

    last_id = dataset_ids[-1]

    if failed_uploads or len(files_for_upload) > 1:
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response_body,
        )
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{last_id}"},
    )


@router.post("/huggingface", summary="Upload datasets from Hugging Face")
async def upload_hf_dataset(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_session)],
    data: HuggingFaceDatasetRequest,
) -> Response:
    """Upload dataset files from Hugging Face.

    This endpoint allows uploading datasets directly from Hugging Face repositories.
    It processes the files and returns the dataset ID in the response.

    Args:
        request: The HTTP request object
        db: Database session for persistence operations
        data: The request body containing the Hugging Face dataset tag

    Returns:
        Response with status code 201 Created and the task ID in the Location header.
    """
    logger.debug("Hugging Face dataset upload initiated", dataset_tag=data.dataset_tag)
    task_id = await upload_hf_dataset_svc(db, data.dataset_tag)

    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/status/{task_id}"},
    )


@router.get("/huggingface/status/{task_id}", summary="Get HF upload dataset status")
async def get_hf_upload_status(
    task_id: UUID, db: Annotated[AsyncSession, Depends(get_session)]
) -> UploadDatasetTask:
    """Get the status of a Hugging Face dataset upload task.

    Args:
        task_id: The UUID of the upload task
        db: Database session for persistence operations

    Returns:
        Response with the status of the upload task.
    """
    return await get_hf_upload_status_svc(db, task_id)


@router.put("/{dataset_id}", summary="Update dataset by ID")
async def update_dataset(
    dataset_id: UUID,
    request: Request,
    dataset: DatasetUpdate,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Update a dataset with version control using ETags.

    Updates a dataset by its ID, requiring an If-Match header with the current
    version to prevent concurrent modification issues.

    Args:
        dataset_id: The UUID of the dataset to update
        request: The HTTP request object containing If-Match header
        dataset: The updated dataset object
        db: Database session for persistence operations

    Returns:
        204 No Content response with Location header

    Raises:
        VersionMismatchError: If the ETag doesn't match current version
        VersionMissingError: If the If-Match header is missing
        DatasetNotFoundError: If the dataset doesn't exist
    """
    new_version = await update_dataset_svc(db, request, dataset_id, dataset)
    logger.debug("Dataset updated", dataset_id=dataset_id)

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"Location": f"{request.url.path}", "ETag": f'"{new_version}"'},
    )


@router.delete("/{dataset_id}", summary="Delete dataset by ID")
async def delete_dataset(
    dataset_id: UUID, db: Annotated[AsyncSession, Depends(get_session)]
) -> Response:
    """Delete a dataset by its ID.

    Permanently removes the dataset and any related records (if cascading deletes
    are configured) from the database.

    Args:
        dataset_id: The UUID of the dataset to delete
        db: Database session for persistence operations

    Returns:
        204 No Content response

    Raises:
        DatasetNotFoundError: If the dataset with the specified ID doesn't exist
    """
    await delete_dataset_svc(db, dataset_id)
    logger.debug("Dataset deleted", dataset_id=dataset_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
