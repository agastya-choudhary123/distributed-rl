"""BaseEnv: the extension point for plugging in your own environment.

To use a custom environment, subclass BaseEnv, set the class attributes, and
implement reset() and step(). The worker instantiates it as config.env_class().

Two action-space types are supported, declared by `action_type`:

    "continuous" -> action is a float vector of shape (action_dim,) in
                    [action_low, action_high]; the policy is a Gaussian.
    "discrete"   -> action is an integer in [0, action_dim); the policy is a
                    Categorical. action_dim is the number of choices.

Everything downstream (actor head, PPO log-prob/entropy, rollout storage) reads
these attributes, so a new env needs no changes anywhere else.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEnv(ABC):
    observation_dim: int = 0
    action_dim: int = 0
    action_type: str = "continuous"  # "continuous" | "discrete"
    action_low: float = -1.0         # continuous only
    action_high: float = 1.0         # continuous only

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Return the initial observation as a float32 numpy array."""

    @abstractmethod
    def step(self, action):
        """Advance one step.

        `action` is a float vector (continuous) or an int (discrete).
        Returns (next_obs, reward, done, info).
        """
