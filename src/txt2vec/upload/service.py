"""Service for handling model uploads including ZIP files."""

import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Final, Optional

from fastapi import UploadFile
from loguru import logger

from txt2vec.config import UPLOAD_DIR


async def upload_embedding_model(
    files: List[UploadFile], 
    model_name: str, 
    description: str = "", 
    extract_zip: bool = True
) -> Dict[str, Any]:
    """Process embedding model upload with ZIP support.

    This function handles temporary file storage and creation of the model directory
    structure in the upload folder. If a ZIP file is provided and extract_zip is True,
    the contents will be extracted preserving the directory structure.

    :param files: List of files comprising the model to upload
    :param model_name: Name for the model (will be used as directory name)
    :param description: Optional description of the model
    :param extract_zip: Whether to extract ZIP files (default: True)
    :return: Dictionary containing information about the uploaded model
    """
    if not files:
        raise ValueError("No files provided for upload")
    
    # Sanitize model name for filesystem usage
    safe_model_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_name)
    
    # Generate a unique model ID
    model_id = uuid.uuid4()
    
    # Create a model directory with the sanitized name and ID
    model_dir = Path(UPLOAD_DIR) / "models" / f"{safe_model_name}_{model_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Write a description file if provided
    if description:
        with open(model_dir / "description.txt", "w", encoding="utf-8") as f:
            f.write(description)
    
    # Counter for extracted files from ZIP
    extracted_file_count = 0
    
    # Store all uploaded files
    for file in files:
        if not file.filename:
            continue
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Read content in chunks to handle large files
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            filename = Path(file.filename).name
            dest_path = model_dir / filename
            
            # Check if this is a ZIP file that should be extracted
            if extract_zip and filename.lower().endswith('.zip'):
                extracted_file_count = _extract_zip_file(temp_path, model_dir)
                logger.debug(f"Extracted {extracted_file_count} files from {filename} to {model_dir}")
                # We still save the original ZIP file
                shutil.move(temp_path, dest_path)
                logger.debug(f"Also saved original ZIP file to {dest_path}")
            else:
                # Move from temp location to final destination
                shutil.move(temp_path, dest_path)
                logger.debug(f"Saved {filename} to {dest_path}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            # Cleanup on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        finally:
            # Ensure the temporary file is deleted
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Reset file position for potential reuse
            await file.seek(0)
    
    # Get the relative path from UPLOAD_DIR
    relative_path = model_dir.relative_to(UPLOAD_DIR)
    
    # In a future version, we'll add database persistence here
    
    return {
        "model_id": str(model_id),
        "model_name": safe_model_name,
        "model_dir": str(model_dir),
        "file_count": len(files) + extracted_file_count
    }


def _extract_zip_file(zip_path: str, extract_to: Path) -> int:
    """Extract a ZIP file to the specified directory.
    
    :param zip_path: Path to the ZIP file
    :param extract_to: Directory to extract contents to
    :return: Number of files extracted
    """
    file_count = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_count = len(zip_ref.namelist())
        zip_ref.extractall(extract_to)
    return file_count