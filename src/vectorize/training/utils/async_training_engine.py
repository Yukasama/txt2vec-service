"""Async SBERT training engine with yielding capability."""

import ast
import asyncio
import builtins
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from ..schemas import TrainRequest


class AsyncSBERTTrainingEngine:
    """Handles SBERT model training with async yielding capability."""

    def __init__(self, model: SentenceTransformer) -> None:
        """Initialize the async training engine.

        Args:
            model: The SBERT model to train
        """
        self.model = model
        self._should_yield = False

    async def train_model_async(
        self,
        train_dataloader: DataLoader,
        train_request: TrainRequest,
        output_dir: str,
        yield_interval_steps: int = 50,
    ) -> dict:
        """Train the SBERT model with async yielding.

        Args:
            train_dataloader: Training data loader
            train_request: Training configuration
            output_dir: Output directory for the trained model
            yield_interval_steps: Yield control every N training steps

        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        orig_output_dir = output_dir
        if "finetuned" in train_request.model_tag:
            new_uuid = uuid.uuid4().hex[:8]
            output_dir = f"{orig_output_dir}-trained-{new_uuid}"

        loss = losses.CosineSimilarityLoss(self.model)
        start_time = time.time()
        captured_metrics = {}

        # Patch the training to add async yields
        original_print = builtins.print
        custom_print = self._create_metrics_capture_function(
            captured_metrics, original_print
        )
        builtins.print = custom_print

        try:
            await self._execute_training_async(
                train_dataloader, train_request, output_dir, loss, yield_interval_steps
            )
        finally:
            builtins.print = original_print

        end_time = time.time()
        train_runtime = end_time - start_time

        training_metrics = self._calculate_metrics(
            captured_metrics, train_runtime, train_dataloader, train_request
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))

        return training_metrics

    async def _execute_training_async(
        self,
        train_dataloader: DataLoader,
        train_request: TrainRequest,
        output_dir: str,
        loss: losses.CosineSimilarityLoss,
        yield_interval_steps: int,
    ) -> None:
        """Execute the actual model training with async yielding."""
        import queue
        import threading

        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        step_counter = [0]  # Use list for mutable reference

        def training_thread():
            """Run training in a separate thread with step counting."""
            try:
                self.model.fit(
                    train_objectives=[(train_dataloader, loss)],
                    epochs=train_request.epochs,
                    warmup_steps=train_request.warmup_steps or 0,
                    show_progress_bar=False,
                    output_path=str(Path(output_dir)),
                    checkpoint_path=str(checkpoint_dir),
                    checkpoint_save_steps=yield_interval_steps,  # Save checkpoints frequently
                )
                result_queue.put("success")

            except Exception as e:
                exception_queue.put(e)

        # Start training in background thread
        training_thread_obj = threading.Thread(target=training_thread)
        training_thread_obj.start()

        # Periodically yield control to the event loop
        while training_thread_obj.is_alive():
            await asyncio.sleep(2)  # Yield every 2 seconds
            logger.debug(
                "Training in progress...",
                steps=step_counter[0],
                target_steps=len(train_dataloader) * train_request.epochs
            )

        # Wait for thread to complete
        training_thread_obj.join()

        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()

        # Ensure we got a successful result
        if result_queue.empty():
            raise RuntimeError("Training completed but no result returned")

    @staticmethod
    def _create_metrics_capture_function(
        captured_metrics: dict, original_print: Callable
    ) -> Callable:
        """Create a custom print function that captures training metrics."""

        def custom_print(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            """Custom print that captures training metrics."""
            text = " ".join(str(arg) for arg in args)

            if (
                "train_runtime" in text
                and "train_loss" in text
                and "train_samples_per_second" in text
            ):
                try:
                    if "{" in text and "}" in text:
                        start_idx = text.find("{")
                        end_idx = text.rfind("}") + 1
                        dict_str = text[start_idx:end_idx]
                        parsed_metrics = ast.literal_eval(dict_str)
                        if isinstance(parsed_metrics, dict):
                            captured_metrics.update(parsed_metrics)
                            logger.debug(
                                "Captured training metrics from print",
                                **parsed_metrics,
                            )
                except (ValueError, SyntaxError) as e:
                    logger.debug(
                        "Failed to parse metrics from print",
                        text=text,
                        error=str(e),
                    )
            original_print(*args, **kwargs)

        return custom_print

    @staticmethod
    def _calculate_metrics(
        captured_metrics: dict,
        train_runtime: float,
        train_dataloader: DataLoader,
        train_request: TrainRequest,
    ) -> dict:
        """Calculate and return training metrics."""
        try:
            total_samples = len(train_dataloader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            total_samples = (
                len(train_dataloader) * train_request.per_device_train_batch_size
            )

        total_steps = len(train_dataloader) * train_request.epochs

        training_metrics = {
            "train_runtime": captured_metrics.get("train_runtime", train_runtime),
            "train_samples_per_second": captured_metrics.get(
                "train_samples_per_second",
                total_samples / train_runtime if train_runtime > 0 else 0.0,
            ),
            "train_steps_per_second": captured_metrics.get(
                "train_steps_per_second",
                total_steps / train_runtime if train_runtime > 0 else 0.0,
            ),
            "train_loss": captured_metrics.get("train_loss", 0.0),
            "epoch": captured_metrics.get("epoch", float(train_request.epochs)),
        }

        if captured_metrics:
            logger.debug("Using captured training metrics", **captured_metrics)
        else:
            logger.debug(
                "No metrics captured, using calculated values",
                calculated_runtime=train_runtime,
            )

        return training_metrics
