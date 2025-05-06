"""Service zum Laden und Speichern von Hugging Face Modellen.

Dieses Modul lädt Modelle von Hugging Face, cached sie lokal und speichert sie
in der Datenbank, falls sie noch nicht vorhanden sind.
"""

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.upload.exceptions import InvalidModelError

_models = {}


async def load_model_and_cache_only(model_id: str, tag: str) -> None:
    key = f"{model_id}@{tag}"
    if key in _models:
        logger.info(f"Modell '{key}' bereits im Cache.")
        return

    try:
        snapshot_path = snapshot_download(
            repo_id=model_id, revision=tag, cache_dir="./hf_cache"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)
        _models[key] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info(f"Modell '{key}' erfolgreich geladen und gecached.")
    except Exception as e:
        logger.exception(f"Fehler beim Laden des Modells '{key}'")
        raise InvalidModelError from e
