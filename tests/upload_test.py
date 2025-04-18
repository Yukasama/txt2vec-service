# ruff: noqa: S101

"""Test the model upload functionality."""

import os
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from fastapi import status
from httpx import post

from txt2vec.config.config import BASE_URL, UPLOAD_DIR

# Ensure the models directory exists
MODELS_DIR = UPLOAD_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Test data
TEST_MODEL_NAME = "test-model"
TEST_MODEL_DESCRIPTION = "Test model for upload functionality"


def test_upload_single_files():
    """Test uploading individual model files."""
    # Create temporary test files
    file_contents = {"config.json": b'{"model_type": "test"}', "model.bin": b"dummy binary content"}
    temp_files = []
    file_objects = []
    
    try:
        # Create temporary files with test content
        files = []
        for filename, content in file_contents.items():
            temp_file = NamedTemporaryFile(delete=False)
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Open the file for reading and track the object
            file_obj = open(temp_file.name, "rb")
            file_objects.append(file_obj)
            
            # Add to files list for the request
            files.append(("files", (filename, file_obj, "application/octet-stream")))
        
        # Make the request
        response = post(
            f"{BASE_URL}uploads/models",
            params={
                "model_name": TEST_MODEL_NAME,
                "description": TEST_MODEL_DESCRIPTION,
            },
            files=files,
        )
        
        # Check response
        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        
        # Extract model ID from Location header
        location = response.headers["Location"]
        model_id = location.split("/")[-1]
        
        # Verify files were saved
        model_dir = list(MODELS_DIR.glob(f"{TEST_MODEL_NAME}_{model_id}*"))
        assert len(model_dir) == 1, "Model directory not found"
        
        model_dir = model_dir[0]
        assert (model_dir / "config.json").exists()
        assert (model_dir / "model.bin").exists()
        assert (model_dir / "description.txt").exists()
        
        # Verify description content
        with open(model_dir / "description.txt", "r") as f:
            description = f.read()
            assert description == TEST_MODEL_DESCRIPTION
            
    finally:
        # First close all file objects
        for file_obj in file_objects:
            try:
                file_obj.close()
            except Exception:
                pass
                
        # Then try to delete temp files with error handling
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except (PermissionError, OSError):
                # Log or print a message, but don't fail the test
                print(f"Could not delete temporary file: {temp_file}")

def test_upload_zip_file():
    """Test uploading and extracting a ZIP file."""
    # Create a temporary ZIP file
    zip_filename = "test_model.zip"
    file_contents = {"config.json": b'{"model_type": "test"}', "model.bin": b"dummy binary content"}
    
    temp_zip = NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip.close()
    temp_files = [temp_zip.name]
    
    try:
        # Create ZIP file with test content
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            for filename, content in file_contents.items():
                # Create temporary file
                temp_file = NamedTemporaryFile(delete=False)
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
                
                # Add to ZIP
                zipf.write(temp_file.name, filename)
        
        # Make the request
        with open(temp_zip.name, "rb") as zip_file:
            files = [("files", (zip_filename, zip_file, "application/zip"))]
            
            response = post(
                f"{BASE_URL}uploads/models",
                params={
                    "model_name": f"{TEST_MODEL_NAME}-zip",
                    "description": f"{TEST_MODEL_DESCRIPTION} (ZIP)",
                    "extract_zip": "true"
                },
                files=files,
            )
        
        # Check response
        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        
        # Extract model ID from Location header
        location = response.headers["Location"]
        model_id = location.split("/")[-1]
        
        # Verify files were saved and extracted
        model_dir = list(MODELS_DIR.glob(f"{TEST_MODEL_NAME}-zip_{model_id}*"))
        assert len(model_dir) == 1, "Model directory not found"
        
        model_dir = model_dir[0]
        assert (model_dir / zip_filename).exists(), "Original ZIP file not saved"
        assert (model_dir / "config.json").exists(), "File not extracted from ZIP"
        assert (model_dir / "model.bin").exists(), "File not extracted from ZIP"
        assert (model_dir / "description.txt").exists()
        
    finally:
        # Clean up the temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except PermissionError:
                    # File might be locked by another process
                    pass