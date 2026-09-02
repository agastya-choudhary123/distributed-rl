"""CartPole: balance a pole on a cart by pushing left or right.

A DISCRETE-action classic-control task, implemented from scratch. Proves the
framework's policy path works for Categorical policies, not just Gaussian.

State (4-dim): [x, x_dot, theta, theta_dot]
Action:        discrete {0: push left, 1: push right}
Reward:        +1 per step the pole stays up
Done:          |x| > 2.4  or  |theta| > 12 degrees, or 500 steps
"""

import numpy as np

from .base import BaseEnv


class CartPole(BaseEnv):
    observation_dim = 4
    action_dim = 2
    action_type = "discrete"

    gravity = 9.8
    mass_cart = 1.0
    mass_pole = 0.1
    length = 0.5  # half the pole length
    force_mag = 10.0
    dt = 0.02
    max_steps = 500
    x_threshold = 2.4
    theta_threshold = 12 * np.pi / 180

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if int(action) == 1 else -self.force_mag

        total_mass = self.mass_cart + self.mass_pole
        pole_ml = self.mass_pole * self.length
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        temp = (force + pole_ml * theta_dot**2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * cos_t**2 / total_mass)
        )
        x_acc = temp - pole_ml * theta_acc * cos_t / total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.step_count += 1

        failed = (
            abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
        )
        done = bool(failed or self.step_count >= self.max_steps)
        reward = 1.0
        return self.state.copy(), reward, done, {}
