"""WeightStore: the global weights, protected by a single threading.Lock.

Holds the authoritative weights, the momentum buffers, and the global step
counter. apply_gradients() averages worker gradients and applies an SGD+momentum
update with optional learning rate scheduling.
"""

import copy
import logging
import pickle
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightStore:
    def __init__(self, initial_weights: dict, lr: float, momentum: float, config=None):
        self.lock = threading.Lock()
        self.weights = {k: v.copy() for k, v in initial_weights.items()}
        self.momentum_buffers = {k: (v * 0.0) for k, v in initial_weights.items()}
        self.base_lr = lr
        self.momentum = momentum
        self.global_step = 0
        self.config = config

    def get_weights(self) -> dict:
        """Deep copy so callers can serialize while other threads update."""
        with self.lock:
            return copy.deepcopy(self.weights)

    def get_step(self) -> int:
        with self.lock:
            return self.global_step

    def apply_gradients(self, grads_list: list[dict]) -> dict:
        """Average worker deltas and apply the global update.

            v = momentum * v + avg_delta
            w = w - server_lr * v

        Workers send (old_weights - locally_updated_weights), so with
        server_lr=1 and momentum=0 this reduces to w <- mean(worker weights).
        The learning rate that shapes training lives in the worker's local
        optimizer, not here.
        """
        n = len(grads_list)
        with self.lock:
            current_lr = self.base_lr

            for name in self.weights:
                avg_grad = grads_list[0][name].copy()
                for g in grads_list[1:]:
                    avg_grad += g[name]
                avg_grad /= n

                v = self.momentum * self.momentum_buffers[name] + avg_grad
                self.momentum_buffers[name] = v
                self.weights[name] = self.weights[name] - current_lr * v

            self.global_step += 1
            return copy.deepcopy(self.weights)

    def save_checkpoint(self, path: Path) -> None:
        """Save weights and state to disk."""
        with self.lock:
            checkpoint = {
                "weights": copy.deepcopy(self.weights),
                "momentum_buffers": copy.deepcopy(self.momentum_buffers),
                "global_step": self.global_step,
                "base_lr": self.base_lr,
                "momentum": self.momentum,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> bool:
        """Load weights and state from disk. Returns True if successful."""
        if not path.exists():
            logger.info(f"No checkpoint found at {path}")
            return False

        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

            with self.lock:
                self.weights = copy.deepcopy(checkpoint["weights"])
                self.momentum_buffers = copy.deepcopy(checkpoint["momentum_buffers"])
                self.global_step = checkpoint["global_step"]

            logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
