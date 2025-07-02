"""Generate embeddings from a model output."""

import base64
import struct
from collections.abc import Iterable
from typing import Any

import torch
from loguru import logger
from transformers import AutoTokenizer

from vectorize.ai_model.exceptions import ModelLoadError, UnsupportedModelError
from vectorize.config import settings

from ..embedding_model import EmbeddingData
from ..schemas import EmbeddingRequest
from .pool_mean import _mean_pool

__all__ = ["_generate_embeddings"]


_DEVICE = torch.device(settings.inference_device)


def _prepare_input_tensors(
    item: str | list[int], tokenizer: AutoTokenizer | None, model_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare input tensors from text or token array.

    Args:
        item: Input text string or token array
        tokenizer: Tokenizer for text processing
        model_name: Model name for error reporting

    Returns:
        Tuple of (input_ids, attention_mask) tensors

    Raises:
        ModelLoadError: If tokenizer is None for text input
    """
    if isinstance(item, list) and all(isinstance(tok, int) for tok in item):
        ids = torch.tensor([item], device=_DEVICE)
        attn = torch.ones_like(ids)
    else:
        if tokenizer is None:
            raise ModelLoadError(model_name)

        text_input = item if isinstance(item, str) else str(item)
        enc = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        ids, attn = (
            enc["input_ids"].to(_DEVICE),
            enc["attention_mask"].to(_DEVICE),
        )

    return ids, attn


def _format_embedding_output(
    embedding: Iterable[float], encoding_format: str | None, dimensions: int | None
) -> list[float] | str:
    """Format embedding output according to requested format and dimensions.

    Args:
        embedding: Raw embedding values
        encoding_format: Output format ("base64" or None for float list)
        dimensions: Number of dimensions to truncate to (optional)

    Returns:
        Formatted embedding as list of floats or base64 string
    """
    emb = list(embedding)
    if dimensions is not None:
        emb = emb[:dimensions]

    if encoding_format == "base64":
        float_bytes = b"".join(struct.pack("f", x) for x in emb)
        return base64.b64encode(float_bytes).decode("utf-8")

    return emb


def _generate_embeddings(
    data: EmbeddingRequest, model: torch.nn.Module, tokenizer: AutoTokenizer | None
) -> tuple[list[EmbeddingData], int]:
    """Generate embeddings from input text using the provided model.

    Processes each input in the request (text strings or token arrays) and creates
    vector embeddings using the specified AI model. Handles different model output
    formats and applies dimension filtering if requested.

    Args:
        data: The embedding request containing the input text or token arrays
            and configuration options like dimensions.
        model: The PyTorch model to use for generating embeddings.
        tokenizer: The tokenizer for converting text to tokens. Can be None
            if the input consists only of pre-tokenized token arrays.

    Returns:
        A tuple containing:
            - results: List of EmbeddingData objects with the generated embeddings
            - total_toks: Total number of tokens processed across all inputs

    Raises:
        ModelLoadError: If the model output format is unsupported or if no tokenizer
            is provided for text inputs.
    """
    inputs: list[str | list[int]] = (
        data.input if isinstance(data.input, list) else [data.input]
    )
    results: list[EmbeddingData] = []
    total_toks = 0

    # Save resources and don't track gradients
    with torch.no_grad():
        for idx, item in enumerate(inputs):
            ids, attn = _prepare_input_tensors(item, tokenizer, data.model)
            total_toks += ids.size(1)

            out = model(
                ids,
                attention_mask=attn,
                return_dict=True,
                output_hidden_states=True,
            )
            vec = _extract_embedding_vector(out, model, attn, ids)

            emb: Iterable[float] = vec.tolist()
            embedding_value = _format_embedding_output(
                emb, data.encoding_format, data.dimensions
            )

            results.append(
                EmbeddingData(object="embedding", embedding=embedding_value, index=idx)
            )

    return results, total_toks


def _extract_embedding_vector(
    model_output: "Any",  # noqa: ANN401
    model: torch.nn.Module,
    attn_mask: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Extract embedding vector from various model output formats.

    Args:
        model_output: Output from the model's forward pass
        model: The PyTorch model that generated the output
        attn_mask: Attention mask tensor for mean pooling
        ids: Input token IDs that may be needed for model.encode

    Returns:
        torch.Tensor: The extracted embedding vector

    Raises:
        ModelLoadError: If the embedding cannot be extracted from the output format
    """
    vec = None

    match model_output:
        case tensor if hasattr(tensor, "last_hidden_state"):
            vec = _mean_pool(tensor.last_hidden_state, attn_mask).squeeze(0)

        case tensor if hasattr(tensor, "pooler_output"):
            vec = tensor.pooler_output.squeeze(0)

        case tensor if (
            hasattr(tensor, "hidden_states") and tensor.hidden_states is not None
        ):
            last_hidden = (
                tensor.hidden_states[-1]
                if isinstance(tensor.hidden_states, (list, tuple))
                else tensor.hidden_states
            )
            vec = _mean_pool(last_hidden, attn_mask).squeeze(0)

        case tensor if hasattr(tensor, "logits"):
            vec = tensor.logits.mean(dim=1).squeeze(0)

        case tensor if isinstance(tensor, torch.Tensor):
            vec = tensor.squeeze(0) if tensor.dim() > 1 else tensor

        case dict_output if (
            isinstance(dict_output, dict) and "embeddings" in dict_output
        ):
            vec = dict_output["embeddings"]

        case list_output if isinstance(list_output, (list, tuple)) and all(
            isinstance(x, (float, int)) for x in list_output
        ):
            vec = torch.tensor(list_output).to(_DEVICE)

    if vec is None and hasattr(model, "encode") and callable(model.encode):
        encoded = model.encode(ids, attention_mask=attn_mask)
        vec = (
            torch.tensor(encoded).to(_DEVICE)
            if not isinstance(encoded, torch.Tensor)
            else encoded
        )

    if vec is None:
        logger.error(
            "Unable to extract embeddings from model output type: {}",
            type(model_output).__name__,
        )
        if isinstance(model_output, dict):
            logger.debug("Available keys: {}", list(model_output.keys()))
        raise UnsupportedModelError(type(model_output).__name__)

    return vec
