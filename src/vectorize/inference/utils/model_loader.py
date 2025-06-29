"""Model loader for saved AI Models."""

from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    T5EncoderModel,
)
from transformers.configuration_utils import PretrainedConfig

from vectorize.ai_model.exceptions import ModelLoadError, ModelNotFoundError
from vectorize.config import settings

__all__ = ["instantiate_from_weights", "load_model"]


# Constants
_DEVICE = torch.device(settings.inference_device)
_IS_TORCH_NEW = torch.__version__ >= "2.7"
_MODEL_NAME = "model.bin"
_CONFIG_FILE = "config.json"
_SNAPSHOTS_DIR = "snapshots"

_OPTIMIZED_MODEL_KWARGS = {
    "trust_remote_code": True,
    "torch_dtype": torch.float16 if _DEVICE.type == "cuda" else "auto",
    "low_cpu_mem_usage": True,
    "use_safetensors": True,
    "device_map": _DEVICE if _DEVICE.type == "cuda" else None,
}


def _has_valid_config(path: Path) -> bool:
    """Check if a path contains a valid config.json file."""
    return path.exists() and (path / _CONFIG_FILE).exists()


def _find_hf_snapshot_path(hf_folder: Path) -> Path | None:
    """Find a valid snapshot path in HuggingFace cache structure."""
    snapshots_dir = hf_folder / _SNAPSHOTS_DIR
    if not snapshots_dir.exists():
        return None

    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    for snapshot_path in snapshot_dirs:
        if _has_valid_config(snapshot_path):
            logger.debug(f"Found model at HF snapshot: {snapshot_path}")
            return snapshot_path

    return None


def _try_standard_path(base_dir: Path, model_tag: str) -> Path | None:
    """Try to find model at standard path."""
    folder = base_dir / model_tag
    if _has_valid_config(folder):
        logger.debug(f"Found model at standard path: {folder}")
        return folder
    return None


def _try_hf_cache_path(base_dir: Path, model_tag: str) -> Path | None:
    """Try to find model in HuggingFace cache structure."""
    hf_folder_name = model_tag.replace("_", "--").replace("/", "--")
    if not hf_folder_name.startswith("models--"):
        hf_folder_name = f"models--{hf_folder_name}"

    hf_folder = base_dir / hf_folder_name
    logger.debug(f"Checking HF cache path: {hf_folder}")

    if not hf_folder.exists():
        return None

    # Try direct config in HF folder
    if _has_valid_config(hf_folder):
        logger.debug(f"Found model at HF cache: {hf_folder}")
        return hf_folder

    # Try snapshot structure
    return _find_hf_snapshot_path(hf_folder)


def _try_alternative_patterns(base_dir: Path, model_tag: str) -> Path | None:
    """Try alternative naming patterns."""
    alt_patterns = [
        f"models--{model_tag.replace('/', '--')}",
        f"models--{model_tag.replace('_', '--')}",
        model_tag.replace("_", "/"),
    ]

    for pattern in alt_patterns:
        alt_folder = base_dir / pattern
        if not alt_folder.exists():
            continue

        # Try direct config
        if _has_valid_config(alt_folder):
            logger.debug(f"Found model at alternative path: {alt_folder}")
            return alt_folder

        # Try snapshot structure
        snapshot_path = _find_hf_snapshot_path(alt_folder)
        if snapshot_path:
            logger.debug(f"Found model at alt snapshot: {snapshot_path}")
            return snapshot_path

    return None


def _find_model_path(model_tag: str) -> Path:
    """Find the actual model path, handling both flat and HuggingFace cache structures.

    Args:
        model_tag: Name of the model directory to find

    Returns:
        Path to the model directory

    Raises:
        ModelNotFoundError: If no valid model path is found
    """
    base_dir = Path(settings.model_inference_dir)

    # Try different search strategies in order
    search_strategies = [
        lambda: _try_standard_path(base_dir, model_tag),
        lambda: _try_hf_cache_path(base_dir, model_tag),
        lambda: _try_alternative_patterns(base_dir, model_tag),
    ]

    for strategy in search_strategies:
        result = strategy()
        if result:
            return result

    # Log all attempted paths for debugging
    _log_search_paths(base_dir, model_tag)
    raise ModelNotFoundError(model_tag)


def _log_search_paths(base_dir: Path, model_tag: str) -> None:
    """Log all paths that were searched for the model."""
    logger.error("Model not found. Searched paths:")
    logger.error(f"  - Standard: {base_dir / model_tag}")

    hf_folder_name = model_tag.replace("_", "--").replace("/", "--")
    if not hf_folder_name.startswith("models--"):
        hf_folder_name = f"models--{hf_folder_name}"
    logger.error(f"  - HF Cache: {base_dir / hf_folder_name}")

    alt_patterns = [
        f"models--{model_tag.replace('/', '--')}",
        f"models--{model_tag.replace('_', '--')}",
        model_tag.replace("_", "/"),
    ]
    for pattern in alt_patterns:
        logger.error(f"  - Alternative: {base_dir / pattern}")


def load_model(model_tag: str) -> tuple[torch.nn.Module, AutoTokenizer | None]:
    """Load a Hugging Face model and its tokenizer from a checkpoint directory.

    Optimized version with config reuse and better loading parameters.
    Supports loading from directories containing either a complete HF model snapshot
    or just the config.json and model weights files.

    Args:
        model_tag: Name of the model directory to load (without file extensions).

    Returns:
        A tuple containing:
            - model: The loaded PyTorch model in evaluation mode on target device
            - tokenizer: The model's tokenizer if available, or None if not found

    Raises:
        ModelNotFoundError: If the model directory doesn't exist
        ModelLoadError: If the model can't be successfully loaded
    """
    folder = _find_model_path(model_tag)
    if not folder.exists():
        logger.error("Model directory not found: {}", folder)
        raise ModelNotFoundError(model_tag)

    logger.info("Loading model '{}' from {}", model_tag, folder)

    try:
        model = _load_model_with_automodel(folder)
    except OSError as ose_exc:
        logger.warning(
            "AutoModel loading failed with OSError: {}. Trying fallback method.",
            str(ose_exc),
        )
        model = _load_model_with_fallback(folder, model_tag)

    model = _prepare_model_for_inference(model)
    tokenizer = _load_tokenizer(folder, model_tag)

    logger.info(
        "Model loading complete - {} on {}, tokenizer: {}",
        model.__class__.__name__,
        _DEVICE,
        "available" if tokenizer is not None else "not available",
    )
    return model, tokenizer


def _load_model_with_automodel(folder: Path) -> torch.nn.Module:
    """Load model using AutoModel.from_pretrained."""
    logger.debug("Loading config from {}", folder / _CONFIG_FILE)
    config = AutoConfig.from_pretrained(folder)

    model_type = getattr(config, "model_type", "unknown")
    architectures = getattr(config, "architectures", [])
    logger.info(
        "Config loaded - model_type: '{}', architectures: {}",
        model_type,
        architectures,
    )

    logger.debug("Attempting to load model with AutoModel.from_pretrained")
    model = AutoModel.from_pretrained(folder, config=config, **_OPTIMIZED_MODEL_KWARGS)
    logger.info(
        "Model loaded successfully with AutoModel: {}", model.__class__.__name__
    )
    return model


def _load_model_with_fallback(folder: Path, model_tag: str) -> torch.nn.Module:
    """Load model using fallback method."""
    try:
        config = AutoConfig.from_pretrained(folder)
        model = instantiate_from_weights(folder, config)
        logger.info("Model loaded successfully with fallback method")
        return model
    except Exception as fallback_exc:
        logger.error("Fallback loading also failed: {}", str(fallback_exc))
        raise ModelLoadError(model_tag) from fallback_exc


def _prepare_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for inference by moving to device and setting eval mode."""
    if (hasattr(model, "device") and model.device != _DEVICE) or not hasattr(
        model, "device"
    ):
        logger.debug("Moving model to device: {}", _DEVICE)
        model = model.to(_DEVICE)
    else:
        logger.debug("Model already on correct device: {}", _DEVICE)

    model = model.eval().requires_grad_(False)
    logger.debug("Model set to eval mode with requires_grad=False")
    return model


def _load_tokenizer(folder: Path, model_tag: str) -> AutoTokenizer | None:
    """Load tokenizer for the model."""
    try:
        logger.debug("Loading tokenizer from {}", folder)
        tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.warning("No tokenizer available for '{}': {}", model_tag, e)
        return None


def _create_model_from_config(cfg: PretrainedConfig) -> torch.nn.Module:
    """Create model instance from configuration."""
    model_type = getattr(cfg, "model_type", None)
    architectures = getattr(cfg, "architectures", None)

    logger.debug("  model_type: {}", model_type)
    logger.debug("  architectures: {}", architectures)

    if model_type == "t5":
        logger.info("T5 model detected - using T5EncoderModel")
        return T5EncoderModel.from_config(cfg)  # type: ignore

    if architectures and len(architectures) > 0 and "MaskedLM" in architectures[0]:
        logger.debug("  Architecture: {}", architectures[0])
        return AutoModelForMaskedLM.from_config(cfg)

    logger.info("Standard model detected - using AutoModel")
    if architectures:
        logger.debug(
            "  Architecture: {}", architectures[0] if architectures else "None"
        )
    return AutoModel.from_config(cfg)


def _load_safetensors_weights(folder: Path, model: torch.nn.Module) -> bool:
    """Try to load weights from safetensors files."""
    safetensors_files = list(folder.glob("*.safetensors"))

    if not safetensors_files:
        return False

    st_path = safetensors_files[0]
    logger.info("Loading weights from safetensors: {}", st_path.name)

    if len(safetensors_files) > 1:
        logger.warning("Multiple safetensors files found, using: {}", st_path.name)

    try:
        state = load_file(st_path, device=str(_DEVICE))
        model.load_state_dict(state, strict=False)
        logger.info(
            "Safetensors weights loaded successfully ({} parameters)", len(state)
        )
        return True
    except Exception as e:
        logger.error("Failed to load safetensors: {}", str(e))
        return False


def _load_pytorch_weights(folder: Path, model: torch.nn.Module) -> bool:
    """Try to load weights from PyTorch binary files."""
    pytorch_file = folder / _MODEL_NAME

    if not pytorch_file.is_file():
        return False

    logger.info("Loading weights from PyTorch binary: {}", pytorch_file.name)
    try:
        state = torch.load(
            pytorch_file,
            mmap=_IS_TORCH_NEW,
            map_location=_DEVICE,
            weights_only=_IS_TORCH_NEW,
        )
        model.load_state_dict(state, strict=False)
        logger.info("PyTorch binary weights loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load PyTorch binary: {}", str(e))
        return False


def instantiate_from_weights(folder: Path, cfg: PretrainedConfig) -> torch.nn.Module:
    """Create a model from config.json and load weights from a state dict file.

    This is a fallback method for loading models that don't follow the standard
    Hugging Face directory structure but contain the necessary components.

    Args:
        folder: Path to the directory containing config.json and weight files.
        cfg: The PretrainedConfig object containing model configuration.

    Returns:
        The loaded PyTorch model in evaluation mode.

    Raises:
        ModelNotFoundError: If neither pytorch_model.bin nor model.bin exist
    """
    logger.info("Using fallback: instantiating model from weights manually")

    model = _create_model_from_config(cfg)
    logger.info("Model architecture created: {}", model.__class__.__name__)

    logger.debug("Moving model to device: {}", _DEVICE)
    model.to(_DEVICE)

    logger.debug("Searching for weight files in {}", folder)

    # Try different weight loading strategies
    weight_loading_strategies = [
        lambda: _load_safetensors_weights(folder, model),
        lambda: _load_pytorch_weights(folder, model),
    ]

    for strategy in weight_loading_strategies:
        if strategy():
            return model.eval()

    # If no weights were loaded successfully
    available_files = [f.name for f in folder.iterdir() if f.is_file()]
    logger.error(
        "No weight files found in {}. Available files: {}", folder, available_files
    )
    raise ModelNotFoundError(str(folder))
