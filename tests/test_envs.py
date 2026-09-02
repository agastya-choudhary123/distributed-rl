"""Generalization test: every registered env trains one step through the same
vectorized pipeline the worker uses, for both continuous (Gaussian) and
discrete (Categorical) policies.
"""

import numpy as np
import torch

from env.registry import ENVS
from env.vectorized import VectorizedEnv
from network.actor import Actor
from network.critic import Critic
from worker.ppo import compute_gradients
from worker.rollout_buffer import RolloutBuffer

NUM_ENVS = 3   # deliberately not 1, so batch dims can't silently collapse
NUM_STEPS = 32


def test_all_envs_train_one_step():
    for name, env_cls in ENVS.items():
        vec = VectorizedEnv(env_cls, num_envs=NUM_ENVS, seed=0)
        actor = Actor(vec.observation_dim, vec.action_dim, vec.action_type)
        critic = Critic(vec.observation_dim)
        store_dim = vec.action_dim if vec.action_type == "continuous" else 1
        buf = RolloutBuffer(NUM_STEPS, NUM_ENVS, vec.observation_dim, store_dim)

        obs = vec.reset()
        assert obs.shape == (NUM_ENVS, vec.observation_dim), name

        for _ in range(NUM_STEPS):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = actor.distribution(obs_t)
                values = critic(obs_t)
                if vec.action_type == "continuous":
                    acts = dist.sample()
                    logp = dist.log_prob(acts).sum(dim=-1)
                    env_acts = acts.numpy()
                    stored = env_acts
                else:
                    acts = dist.sample()
                    logp = dist.log_prob(acts)
                    env_acts = acts.long().numpy()
                    stored = env_acts.astype(np.float32).reshape(-1, 1)

            assert values.shape == (NUM_ENVS,), name
            assert logp.shape == (NUM_ENVS,), name

            nobs, r, done, _ = vec.step(env_acts)
            buf.add(obs, stored, r, values.numpy(), logp.numpy(), done)
            obs = nobs

        buf.compute_gae(np.zeros(NUM_ENVS))
        actor.zero_grad(set_to_none=False)
        critic.zero_grad(set_to_none=False)
        for batch in buf.get_minibatches(32):
            compute_gradients(actor, critic, batch)

        # Every parameter got a real gradient regardless of action-space type.
        for pname, p in list(actor.named_parameters()) + list(critic.named_parameters()):
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"{name}:{pname}"
        print(f"PASS {name:14s} obs={vec.observation_dim} "
              f"act={vec.action_dim} type={vec.action_type}")


if __name__ == "__main__":
    test_all_envs_train_one_step()
