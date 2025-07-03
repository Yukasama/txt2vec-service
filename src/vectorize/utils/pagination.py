"""Pagination model for API responses."""

from typing import TypeVar

from pydantic import BaseModel, Field

__all__ = ["Page"]


T = TypeVar("T")


class Page[T](BaseModel):
    """Generic pagination model for API responses."""

    items: list[T]
    total: int
    limit: int = Field(ge=1, le=1000)
    offset: int = Field(ge=0)
