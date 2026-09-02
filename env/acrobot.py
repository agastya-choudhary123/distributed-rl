"""Acrobot: swing a two-link underactuated arm so its tip reaches a height.

A harder DISCRETE-action task with a 6-dim observation and genuine two-link
rigid-body dynamics (integrated with RK4). Torque is applied only at the second
joint, so the agent must pump energy in — a non-trivial control problem.

State (6-dim): [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]
Action:        discrete {0:-1, 1:0, 2:+1} torque at the elbow
Reward:        -1 per step until the tip is raised
Done:          -cos(t1) - cos(t1+t2) > 1.0, or 500 steps
"""

import numpy as np

from .base import BaseEnv


class Acrobot(BaseEnv):
    observation_dim = 6
    action_dim = 3
    action_type = "discrete"

    dt = 0.2
    LINK_LENGTH_1 = 1.0
    LINK_MASS_1 = LINK_MASS_2 = 1.0
    LINK_COM_1 = LINK_COM_2 = 0.5
    LINK_MOI = 1.0
    g = 9.8
    max_vel_1 = 4 * np.pi
    max_vel_2 = 9 * np.pi
    max_steps = 500
    avail_torque = (-1.0, 0.0, 1.0)

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float64)  # t1, t2, dt1, dt2
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        t1, t2, dt1, dt2 = self.state
        return np.array(
            [np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), dt1, dt2],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.1, 0.1, size=4)
        self.step_count = 0
        return self._obs()

    def _dsdt(self, s, torque):
        m1, m2 = self.LINK_MASS_1, self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1, lc2 = self.LINK_COM_1, self.LINK_COM_2
        I1 = I2 = self.LINK_MOI
        g = self.g
        t1, t2, dt1, dt2 = s

        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(t2))
            + I1 + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(t2)) + I2
        phi2 = m2 * lc2 * g * np.cos(t1 + t2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dt2**2 * np.sin(t2)
            - 2 * m2 * l1 * lc2 * dt2 * dt1 * np.sin(t2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(t1 - np.pi / 2.0)
            + phi2
        )
        ddt2 = (
            torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dt1**2 * np.sin(t2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddt1 = -(d2 * ddt2 + phi1) / d1
        return np.array([dt1, dt2, ddt1, ddt2])

    def step(self, action):
        torque = self.avail_torque[int(action)]
        s = self.state
        # RK4 over one dt.
        k1 = self._dsdt(s, torque)
        k2 = self._dsdt(s + 0.5 * self.dt * k1, torque)
        k3 = self._dsdt(s + 0.5 * self.dt * k2, torque)
        k4 = self._dsdt(s + self.dt * k3, torque)
        s = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        s[0] = _wrap(s[0])
        s[1] = _wrap(s[1])
        s[2] = np.clip(s[2], -self.max_vel_1, self.max_vel_1)
        s[3] = np.clip(s[3], -self.max_vel_2, self.max_vel_2)
        self.state = s
        self.step_count += 1

        raised = bool(-np.cos(s[0]) - np.cos(s[0] + s[1]) > 1.0)
        done = bool(raised or self.step_count >= self.max_steps)
        reward = 0.0 if raised else -1.0
        return self._obs(), reward, done, {"raised": raised}


def _wrap(x: float) -> float:
    return (x + np.pi) % (2 * np.pi) - np.pi
