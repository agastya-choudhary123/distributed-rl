"""RolloutBuffer: stores transitions and computes GAE advantages.

Storage is (num_steps, num_envs, ...) — time-major, env-minor. This shape is
load-bearing: GAE is a backward recurrence *along the time axis of a single
environment*, so the envs must stay on their own axis. Flattening N envs into
one sequential array makes step t of env i bootstrap off step t of env i+1,
which silently destroys every advantage.

The flatten to (num_steps * num_envs, ...) happens only in get_minibatches(),
after GAE is done, where sample order no longer matters.
"""

import numpy as np


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        T, N = self.num_steps, self.num_envs
        self.states = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.ptr = 0

    def add(self, states, actions, rewards, values, log_probs, dones) -> None:
        """Store one timestep across all envs.

        Every argument is batched over envs: states (num_envs, obs_dim),
        actions (num_envs, action_dim), the rest (num_envs,).
        """
        t = self.ptr
        self.states[t] = states
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.values[t] = values
        self.log_probs[t] = log_probs
        self.dones[t] = dones
        self.ptr += 1

    def compute_gae(self, last_values, gamma: float = 0.99, lam: float = 0.95):
        """Backward GAE recurrence, run independently per environment.

        last_values is (num_envs,): V(s_T) for each env, used to bootstrap the
        final step. Vectorized over the env axis, so the recurrence carries one
        running advantage per env rather than one shared across all of them.
        """
        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.num_envs)
        adv = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.ptr)):
            next_values = last_values if t == self.ptr - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]
        return self.advantages[: self.ptr], self.returns[: self.ptr]

    def get_minibatches(self, minibatch_size: int = 64):
        """Yield shuffled minibatches with advantages normalized over the batch."""
        T = self.ptr
        n = T * self.num_envs

        states = self.states[:T].reshape(n, self.obs_dim)
        actions = self.actions[:T].reshape(n, self.action_dim)
        log_probs = self.log_probs[:T].reshape(n)
        advantages = self.advantages[:T].reshape(n)
        returns = self.returns[:T].reshape(n)

        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            mb = idx[start : start + minibatch_size]
            yield {
                "states": states[mb],
                "actions": actions[mb],
                "log_probs": log_probs[mb],
                "advantages": adv_norm[mb],
                "returns": returns[mb],
            }
