"""Upload router."""

from typing import Annotated, Final, List

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from loguru import logger

from txt2vec.handle_exceptions import handle_exceptions
from txt2vec.upload.service import upload_embedding_model

router = APIRouter(tags=["Model Upload"])


@router.get("/")
def helloworld() -> dict[str, str]:
    """Root endpoint for the FastAPI application.

    :return: A simple greeting message
    :rtype: dict[str, str]
    """
    return {"message": "Hello World!"}


@router.post("/models")
@handle_exceptions
async def upload_model(
    files: List[UploadFile],
    request: Request,
    model_name: Annotated[str, Query(description="Name for the uploaded model")],
    description: Annotated[
        str, Query(description="Description of the model")
    ] = "",
    extract_zip: Annotated[
        bool, Query(description="Whether to extract ZIP files")
    ] = True,
) -> Response:
    """Upload embedding model files to the server.
    
    This endpoint accepts multiple files that constitute an embedding model
    and saves them to the model storage directory. If a ZIP file is uploaded
    and extract_zip is True, its contents will be extracted to the model directory.
    
    :param files: List of files comprising the embedding model
    :param request: The HTTP request object
    :param model_name: Name to give to the uploaded model
    :param description: Optional description of the model's purpose or capabilities
    :param extract_zip: Whether to extract contents of ZIP files (default: True)
    :return: HTTP 201 response with location header pointing to the created model
    """
    logger.debug("Uploading model '{}' with {} files", model_name, len(files))
    
    result = await upload_embedding_model(files, model_name, description, extract_zip)
    
    logger.info("Successfully uploaded model: {}", result["model_dir"])
    
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={"Location": f"{request.url}/{result['model_id']}"},
    )