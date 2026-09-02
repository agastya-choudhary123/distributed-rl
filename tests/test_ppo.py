"""Step 5: rollout buffer + PPO gradients produce non-None, non-zero grads."""

import numpy as np
import torch

from network.actor import Actor
from network.critic import Critic
from worker.ppo import compute_gradients
from worker.rollout_buffer import RolloutBuffer


def test_gradients_nonzero():
    torch.manual_seed(0)
    np.random.seed(0)
    num_steps, num_envs = 64, 2
    buf = RolloutBuffer(num_steps=num_steps, num_envs=num_envs, obs_dim=4, action_dim=1)
    for _ in range(num_steps):
        buf.add(
            states=np.random.randn(num_envs, 4).astype(np.float32),
            actions=np.random.uniform(-1, 1, size=(num_envs, 1)).astype(np.float32),
            rewards=np.random.randn(num_envs),
            values=np.random.randn(num_envs),
            log_probs=np.random.randn(num_envs),
            dones=np.zeros(num_envs),
        )
    buf.compute_gae(last_values=np.zeros(num_envs))

    actor, critic = Actor(4, 1, "continuous"), Critic(4)
    actor.zero_grad(set_to_none=False)
    critic.zero_grad(set_to_none=False)
    for _ in range(4):
        for batch in buf.get_minibatches(64):
            compute_gradients(actor, critic, batch)

    for name, p in list(actor.named_parameters()) + list(critic.named_parameters()):
        assert p.grad is not None, f"{name} grad is None"
        assert float(p.grad.abs().sum()) > 0, f"{name} grad is all zero"
    print("PASS test_ppo: all actor+critic params have non-zero accumulated grads")


if __name__ == "__main__":
    test_gradients_nonzero()
