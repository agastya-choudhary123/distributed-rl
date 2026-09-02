"""TrainConfig: every hyperparameter in one dataclass.

This is the one place a researcher changes things. train.py applies CLI
overrides on top of these defaults and passes the frozen config to every
process.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from env.swingbot import SwingBot


def scheduled_lr(base_lr: float, step: int, max_steps: int,
                 schedule: str = "constant", warmup_steps: int = 0) -> float:
    """Learning rate at `step` under the given schedule.

    Applied by the worker to its local optimizer: that is where the learning
    rate actually acts, since the server only averages weight deltas.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / warmup_steps)
    if schedule == "constant":
        return base_lr

    denom = max(max_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / denom, 1.0)
    if schedule == "linear":
        return base_lr * (1.0 - progress)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


@dataclass
class TrainConfig:
    # Distributed / rollout.
    num_workers: int = 4
    rollout_length: int = 512
    envs_per_worker: int = 4  # Vectorized environments per worker.

    # PPO.
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Worker-local optimizer. Each worker runs true PPO locally with Adam, so
    # the policy actually moves between epochs and the clipped surrogate
    # objective is live. This is the learning rate that matters.
    lr: float = 3e-4
    lr_schedule: str = "constant"  # "constant", "linear", "cosine"
    lr_warmup_steps: int = 0

    # Parameter-server blend. Workers ship (old_weights - new_weights) as a
    # pseudo-gradient; the server averages them and applies
    #     w <- w - server_lr * (momentum * v + avg_delta)
    # With server_lr=1 and momentum=0 this is exactly FedAvg: the new global
    # weights are the mean of the workers' locally-updated weights.
    server_lr: float = 1.0
    momentum: float = 0.0

    # Async aggregation: apply update if >=min_ready_workers arrive.
    # 0 = synchronous (wait for all); N = async with N-worker quorum.
    min_ready_workers: int = 0  # 0 = sync (all), else = straggler-tolerant
    gradient_timeout_sec: float = 30.0  # Give slow workers this long to push.

    # Training length (global weight updates).
    max_steps: int = 5_000_000

    # Networking.
    server_host: str = "127.0.0.1"
    server_port: int = 9999
    dashboard_port: int = 8000

    # Hardware. CPU is the default on purpose, not as a placeholder: the
    # networks here are 64-unit MLPs and the envs step on the CPU, so a GPU
    # spends more time on per-timestep transfers and kernel launches than it
    # saves. Measured on this box (cartpole, 2 workers x 4 envs, 100 steps):
    # cpu 20,480 env-steps/s vs mps 3,103 env-steps/s — the GPU is 6.6x slower.
    # Set --device cuda|mps explicitly if you scale the networks up enough for
    # that to invert. "auto" picks the fastest available, which today is cpu.
    device: str = "cpu"  # "auto", "cuda", "cpu", "mps" (Apple Silicon)
    use_grad_compression: bool = False  # Quantize gradients for wire.

    # Checkpointing.
    checkpoint_dir: Path = Path("./checkpoints")
    checkpoint_interval: int = 100  # Save every N global steps.
    load_checkpoint: bool = True  # Resume from latest if it exists.

    # Environment (the extension point).
    env_class: type = SwingBot

    # Reproducibility: base seed sent by the server so workers init identically.
    seed: int = 0

    def __post_init__(self):
        """Validate config hyperparameters."""
        assert self.num_workers >= 1, "num_workers must be >= 1"
        assert self.rollout_length >= 32, "rollout_length must be >= 32"
        assert self.envs_per_worker >= 1, "envs_per_worker must be >= 1"
        assert self.ppo_epochs >= 1, "ppo_epochs must be >= 1"
        assert self.minibatch_size >= 1, "minibatch_size must be >= 1"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        assert 0 < self.gae_lambda <= 1, "gae_lambda must be in (0, 1]"
        assert 0 < self.clip_eps < 1, "clip_eps must be in (0, 1)"
        assert self.max_grad_norm > 0, "max_grad_norm must be > 0"
        assert 0 <= self.vf_coef, "vf_coef must be >= 0"
        assert 0 <= self.ent_coef, "ent_coef must be >= 0"
        assert self.lr > 0, "lr must be > 0"
        assert self.server_lr > 0, "server_lr must be > 0"
        assert 0 <= self.momentum < 1, "momentum must be in [0, 1)"
        assert self.lr_schedule in ("constant", "linear", "cosine"), \
            "lr_schedule must be one of: constant, linear, cosine"
        assert self.lr_warmup_steps >= 0, "lr_warmup_steps must be >= 0"
        assert self.min_ready_workers >= 0, "min_ready_workers must be >= 0"
        if self.min_ready_workers > 0:
            assert self.min_ready_workers <= self.num_workers, \
                "min_ready_workers must be <= num_workers"
        assert self.device in ("auto", "cuda", "cpu", "mps"), \
            "device must be one of: auto, cuda, cpu, mps"
        assert self.checkpoint_interval >= 1, "checkpoint_interval must be >= 1"
        assert self.max_steps >= 1, "max_steps must be >= 1"

    def total_parallel_envs(self) -> int:
        """Total number of environments running in parallel."""
        return self.num_workers * self.envs_per_worker

    def resolve_device(self) -> str:
        """Turn device="auto" into a concrete device for this machine.

        "auto" means fastest-for-this-workload, which is CPU: these networks
        are too small for a GPU to pay off (see the `device` field comment for
        the measurement). It is not a stub — if you scale the networks up,
        change the policy here.

        An explicitly named device is returned untouched, so asking for "cuda"
        on a box without it fails loudly rather than silently falling back and
        quietly training somewhere you did not intend.
        """
        if self.device != "auto":
            return self.device
        return "cpu"
