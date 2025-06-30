"""EmbeddingRequest model cloning OpenAI API Endpoint."""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["EmbeddingRequest"]


class EmbeddingRequest(BaseModel):
    """Request body for Embeddings Endpoint, modelled after OpenAI's spec."""

    input: str | list[str] | list[list[int]] | list[int] = Field(
        description=(
            "Input text or tokens to embed. Accepts a single string, list of strings, "
            "or a list of token arrays."
        ),
    )

    model: str = Field(
        description="ID of the model to use, e.g. `text-embedding-ada-002`.",
        min_length=1,
    )

    dimensions: int | None = Field(
        None,
        description=(
            "Optional number of dimensions for resulting embeddings (text-embedding-3)."
        ),
        ge=1,
    )

    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Either `float` (default) or `base64` for the embedding values.",
    )

    user: str | None = Field(
        None,
        description="An identifier for your end-user (helps OpenAI monitor for abuse).",
    )
