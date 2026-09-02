"""Critic network: maps state -> V(s), the scalar value estimate.

4 -> 64 -> 64 -> 1, tanh activations.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs).squeeze(-1)
