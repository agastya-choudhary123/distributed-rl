"""MountainCarContinuous: drive an underpowered car up a hill.

A CONTINUOUS-action classic-control task with a 2-dim state (a different obs
size than SwingBot, proving nothing is hard-coded to 4-dim). The car can't climb
directly; it must learn to build momentum by rocking back and forth.

State (2-dim): [position, velocity]
Action:        continuous scalar in [-1, 1] (throttle)
Reward:        -0.1 * action^2 each step, +100 on reaching the goal
Done:          position >= 0.45, or 999 steps
"""

import numpy as np

from .base import BaseEnv


class MountainCarContinuous(BaseEnv):
    observation_dim = 2
    action_dim = 1
    action_type = "continuous"
    action_low = -1.0
    action_high = 1.0

    min_pos = -1.2
    max_pos = 0.6
    max_speed = 0.07
    goal_pos = 0.45
    power = 0.0015
    max_steps = 999

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.pos = 0.0
        self.vel = 0.0
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.pos, self.vel], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.pos = float(self.rng.uniform(-0.6, -0.4))
        self.vel = 0.0
        self.step_count = 0
        return self._obs()

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))

        self.vel += a * self.power - 0.0025 * np.cos(3 * self.pos)
        self.vel = float(np.clip(self.vel, -self.max_speed, self.max_speed))
        self.pos += self.vel
        self.pos = float(np.clip(self.pos, self.min_pos, self.max_pos))
        if self.pos <= self.min_pos and self.vel < 0:
            self.vel = 0.0  # inelastic wall on the left

        self.step_count += 1
        reached = self.pos >= self.goal_pos
        done = bool(reached or self.step_count >= self.max_steps)
        reward = -0.1 * a**2 + (100.0 if reached else 0.0)
        return self._obs(), reward, done, {"reached": reached}
