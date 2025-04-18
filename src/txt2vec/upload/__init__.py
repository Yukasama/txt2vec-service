"""Upload module for txt2vec."""

from txt2vec.upload.router import router
from txt2vec.upload.service import upload_embedding_model

__all__ = ["router", "upload_embedding_model"]