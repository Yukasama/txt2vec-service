"""Service zum Laden und Speichern von Hugging Face Modellen.

Dieses Modul lädt Modelle von Hugging Face, cached sie lokal und speichert sie
in der Datenbank, falls sie noch nicht vorhanden sind.
"""

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.upload.exceptions import InvalidModelError

_models = {}


async def load_model_and_cache_only(model_id: str, tag: str) -> None:  # noqa: RUF029
    """Loads a Hugging Face model and caches it locally.

    This function downloads a model and its tokenizer from the Hugging Face Hub,
    initializes a sentiment-analysis pipeline, and caches it in memory. If the
    model is already cached, it skips the download and initialization.

    Args:
        model_id (str): The ID of the Hugging Face model repository.
        tag (str): The specific revision or tag of the model to download.

    Raises:
        InvalidModelError: If an error occurs during the model download or
        initialization.
    """
    key = f"{model_id}@{tag}"
    if key in _models:
        logger.info(f"Model '{key}' is already in Cache.")
        return

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id, revision=tag, cache_dir="./hf_cache"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)
        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info(f"Model '{key}' successfully loaded and cached.")
    except Exception as e:
        logger.exception(f"Error loading the model '{key}'")
        raise InvalidModelError from e
