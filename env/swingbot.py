"""SwingBot: a 2D continuous-control swing-up-and-balance task, from scratch.

A single rigid rod is pinned at its base. The agent applies torque at the pin.
Goal: swing the rod upright (theta = 0 from vertical) and hold it there.

State (4-dim):  [cos(theta), sin(theta), theta_dot, action_prev]
Action:         scalar torque in [-1, 1]
Dynamics:       Euler integration, dt = 0.05s
Reward:         cos(theta) - 0.1 * theta_dot^2 - 0.001 * action^2
Episode:        200 steps, no early termination
"""

import numpy as np

from .base import BaseEnv


class SwingBot(BaseEnv):
    observation_dim = 4
    action_dim = 1
    action_low = -1.0
    action_high = 1.0

    # Physics parameters.
    max_torque = 2.0
    gravity = 9.8
    mass = 1.0
    length = 1.0
    dt = 0.05
    max_steps = 200

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self.theta_dot = 0.0
        self.action_prev = 0.0
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [np.cos(self.theta), np.sin(self.theta), self.theta_dot, self.action_prev],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        self.theta_dot = float(self.rng.uniform(-1.0, 1.0))
        self.action_prev = 0.0
        self.step_count = 0
        return self._obs()

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

        torque = a * self.max_torque
        # theta measured from vertical; sin(theta) is the gravity moment arm.
        theta_ddot = (
            torque - self.gravity * self.mass * self.length * np.sin(self.theta)
        ) / (self.mass * self.length**2)
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt
        # Wrap angle to [-pi, pi] for numerical stability.
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        reward = (
            np.cos(self.theta)
            - 0.1 * self.theta_dot**2
            - 0.001 * a**2
        )

        self.action_prev = a
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._obs(), float(reward), done, {"angle": self.theta}

    def render_ascii(self) -> str:
        """20-char-wide ASCII bar showing tip horizontal position (debug only)."""
        width = 20
        pos = int((np.sin(self.theta) * 0.5 + 0.5) * (width - 1))
        bar = ["-"] * width
        bar[pos] = "O" if np.cos(self.theta) > 0 else "x"
        return "|" + "".join(bar) + f"|  theta={self.theta:+.2f}"
