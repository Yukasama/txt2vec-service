"""Utils for ZIP model tests."""

from pathlib import Path

__all__ = ["get_test_zip_file"]


def get_test_zip_file(file_path: Path) -> dict[str, tuple[str, bytes, str]]:
    """Read file content and prepare it for upload testing.

    Args:
        file_path: Path to the file to read

    Returns:
        Dictionary in the format expected by FastAPI's TestClient for file uploads
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    content = file_path.read_bytes()
    filename = file_path.name

    # Determine MIME type based on extension
    mime_type = (
        "application/zip"
        if filename.lower().endswith(".zip")
        else "application/octet-stream"
    )

    return {"file": (filename, content, mime_type)}
