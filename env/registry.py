"""Environment registry: select an env by name from the CLI.

    python train.py --env cartpole

To register a new env, import its class and add one line to ENVS. Nothing else
in the codebase needs to change — the actor, PPO, and rollout storage all read
the env's declared attributes (observation_dim, action_dim, action_type).
"""

from .acrobot import Acrobot
from .cartpole import CartPole
from .mountain_car import MountainCarContinuous
from .swingbot import SwingBot

ENVS: dict[str, type] = {
    "swingbot": SwingBot,                    # continuous, 4-dim  (pendulum swing-up)
    "cartpole": CartPole,                    # discrete,   4-dim  (balance)
    "mountain_car": MountainCarContinuous,   # continuous, 2-dim  (momentum)
    "acrobot": Acrobot,                      # discrete,   6-dim  (two-link swing-up)
}


def get_env(name: str) -> type:
    if name not in ENVS:
        raise ValueError(
            f"unknown env '{name}'. Available: {', '.join(sorted(ENVS))}"
        )
    return ENVS[name]
