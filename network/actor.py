"""Actor network: a policy over a continuous OR discrete action space.

The action-space type is chosen from the env's `action_type`:

    "continuous" -> head outputs a mean per action dim; a learned log_std gives a
                    diagonal Gaussian. action_dim = number of action components.
    "discrete"   -> head outputs logits; a Categorical distribution.
                    action_dim = number of choices.

The rest of the codebase never branches on action type — it only calls
distribution(), act(), and evaluate(), which handle both cases internally.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        hidden: int = 64,
    ):
        super().__init__()
        assert action_type in ("continuous", "discrete"), action_type
        self.action_type = action_type
        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        if action_type == "continuous":
            # State-independent learned log std, one per action dim.
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        out = self.trunk(obs)
        if self.action_type == "continuous":
            std = self.log_std.exp().expand_as(out)
            return torch.distributions.Normal(out, std)
        return torch.distributions.Categorical(logits=out)

    def act(self, obs: torch.Tensor):
        """Sample an action for rollout collection (no grad expected).

        Returns (action_for_env, action_to_store, log_prob) where:
          continuous: action_for_env == action_to_store, a float vector.
          discrete:   action_for_env is a python int, action_to_store is a
                      float array of shape (1,) holding that index.
        """
        dist = self.distribution(obs)
        a = dist.sample()
        if self.action_type == "continuous":
            log_prob = dist.log_prob(a).sum(dim=-1)
            a_np = a.squeeze(0).detach().numpy()
            return a_np, a_np, float(log_prob.item())
        log_prob = dist.log_prob(a)
        idx = int(a.item())
        return idx, [float(idx)], float(log_prob.item())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return (log_prob, entropy) for stored actions, with grad. Used by PPO.

        `actions` is shape (batch, action_dim) for continuous, or (batch, 1)
        holding integer indices for discrete.
        """
        dist = self.distribution(obs)
        if self.action_type == "continuous":
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            idx = actions.squeeze(-1).long()
            log_prob = dist.log_prob(idx)
            entropy = dist.entropy()
        return log_prob, entropy
