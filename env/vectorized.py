"""Vectorized environment wrapper: run N environments in sync and batch operations.

Simplistic synchronous vectorizer (no asyncio) that stacks N env instances,
steps them all together, and batches observations for the actor network.
"""

import numpy as np

from env.base import BaseEnv


class VectorizedEnv:
    """Wrapper: N copies of an env, stepped synchronously, observations batched."""

    def __init__(self, env_class: type, num_envs: int, seed: int = 0):
        """Create num_envs instances of env_class.

        Args:
            env_class: A BaseEnv subclass.
            num_envs: Number of parallel environments.
            seed: Base seed; env i gets seed + i.
        """
        self.num_envs = num_envs
        self.envs = [env_class(seed=seed + i) for i in range(num_envs)]

        # Metadata from the first env (all should be identical).
        self.observation_dim = self.envs[0].observation_dim
        self.action_dim = self.envs[0].action_dim
        self.action_type = self.envs[0].action_type
        if self.action_type == "continuous":
            self.action_low = self.envs[0].action_low
            self.action_high = self.envs[0].action_high

    def reset(self) -> np.ndarray:
        """Reset all envs and return stacked observations.

        Returns:
            observations: shape (num_envs, observation_dim)
        """
        obs_list = [env.reset() for env in self.envs]
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        """Step all envs with a batch of actions.

        Args:
            actions: shape (num_envs, action_dim) for continuous,
                     or (num_envs,) of ints for discrete.

        Returns:
            observations: shape (num_envs, observation_dim)
            rewards: shape (num_envs,)
            dones: shape (num_envs,)
            infos: list of dicts
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for i, env in enumerate(self.envs):
            if self.action_type == "continuous":
                action = actions[i]
            else:
                action = int(actions[i])

            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

            # Auto-reset if done.
            if done:
                obs = env.reset()
                obs_list[-1] = obs

        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            info_list,
        )
