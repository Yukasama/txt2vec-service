"""Module for monitoring VRAM usage and estimating model sizes."""

import sys

import torch
from loguru import logger

__all__ = ["VRAMMonitor"]


class VRAMMonitor:
    """Monitors VRAM usage and estimates model sizes."""

    def __init__(self, safety_margin_gb: float = 1.0) -> None:
        """Initialize VRAM monitor.

        Args:
            safety_margin_gb: Safety margin in GB to keep free
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.safety_margin_bytes = int(safety_margin_gb * 1024**3)
        self.is_cuda = self.device.type == "cuda"

        if self.is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            logger.info(
                "VRAM Monitor initialized",
                device=self.device,
                total_vram_gb=props.total_memory / (1024**3),
                safety_margin_gb=safety_margin_gb,
            )
        else:
            logger.info("VRAM Monitor initialized for CPU - unlimited memory")

    def get_total_vram(self) -> int:
        """Get total VRAM in bytes."""
        if self.is_cuda:
            return torch.cuda.get_device_properties(self.device).total_memory
        return sys.maxsize

    def get_used_vram(self) -> int:
        """Get currently used VRAM in bytes."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device)
        return 0

    def get_available_vram(self) -> int:
        """Get available VRAM minus safety margin in bytes."""
        if self.is_cuda:
            total = self.get_total_vram()
            used = self.get_used_vram()
            available = total - used - self.safety_margin_bytes
            return max(0, available)
        return sys.maxsize

    @staticmethod
    def estimate_model_vram(model: torch.nn.Module) -> int:
        """Estimate VRAM consumption of a model in bytes."""
        total_size = 0

        for param in model.parameters():
            total_size += param.numel() * param.element_size()

        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()

        return int(total_size * 1.2)

    def can_fit_model(self, estimated_vram: int) -> bool:
        """Check if model fits in available VRAM."""
        available = self.get_available_vram()
        can_fit = available >= estimated_vram

        logger.debug(
            "VRAM check",
            estimated_gb=estimated_vram / (1024**3),
            available_gb=available / (1024**3),
            can_fit=can_fit,
        )

        return can_fit

    def get_vram_info(self) -> dict:
        """Get VRAM status for monitoring."""
        if not self.is_cuda:
            return {"device": "cpu", "unlimited": True}

        total = self.get_total_vram()
        used = self.get_used_vram()
        available = self.get_available_vram()

        return {
            "device": str(self.device),
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "available_gb": available / (1024**3),
            "utilization_percent": (used / total) * 100,
            "safety_margin_gb": self.safety_margin_bytes / (1024**3),
        }
