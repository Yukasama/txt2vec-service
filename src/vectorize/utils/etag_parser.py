"""Parse ETag header."""

from fastapi import Request

from vectorize.common.exceptions import VersionMissingError

__all__ = ["parse_etag"]


def parse_etag(resource_id: str, request: Request) -> int:
    """Parse the If-Match ETag header from a request and return the integer version.

    Args:
        resource_id (str): The identifier of the resource for error reporting.
        request (Request): The FastAPI request object containing headers.

    Returns:
        int: The integer version extracted from the ETag header.

    Raises:
        VersionMissingError: If-Match header is missing, malformed, or not an integer.
    """
    value = request.headers.get("If-Match", "").strip()
    if not (value.startswith('"') and value.endswith('"')):
        raise VersionMissingError(resource_id)

    try:
        return int(value.strip('"'))
    except ValueError as exc:
        raise VersionMissingError(resource_id) from exc
