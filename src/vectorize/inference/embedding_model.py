"""Response model for OpenAI's Embedding API."""

from typing import Literal

from pydantic import BaseModel

__all__ = ["EmbeddingData", "EmbeddingUsage", "Embeddings"]


class EmbeddingUsage(BaseModel):
    """Usage information for the embeddings of a model."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """Data structure for each embedding that is generated."""

    object: Literal["embedding"]
    embedding: list[float] | str
    index: int


class Embeddings(BaseModel):
    """Embeddings structure for the generated embeddings of a model."""

    object: Literal["list"]
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
