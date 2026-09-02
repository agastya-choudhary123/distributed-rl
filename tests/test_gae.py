"""GAE correctness, including the per-env separation that vectorization needs.

The bug this guards against: flattening N envs into one sequential buffer makes
step t of env i bootstrap off step t of env i+1, which silently corrupts every
advantage while still producing plausible-looking numbers.
"""

import numpy as np

from worker.rollout_buffer import RolloutBuffer

GAMMA, LAM = 0.99, 0.95


def _reference_gae(rewards, values, dones, last_value):
    """Single-trajectory GAE, written the obvious way, as ground truth."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * nonterminal - values[t]
        running = delta + GAMMA * LAM * nonterminal * running
        adv[t] = running
    return adv


def test_gae_matches_reference_per_env():
    rng = np.random.default_rng(0)
    T, N = 16, 3
    buf = RolloutBuffer(T, N, obs_dim=2, action_dim=1)

    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = (rng.random((T, N)) < 0.15).astype(np.float32)
    last_values = rng.normal(size=N)

    for t in range(T):
        buf.add(
            states=rng.normal(size=(N, 2)),
            actions=rng.normal(size=(N, 1)),
            rewards=rewards[t],
            values=values[t],
            log_probs=rng.normal(size=N),
            dones=dones[t],
        )
    buf.compute_gae(last_values, gamma=GAMMA, lam=LAM)

    # Each env's advantages must match the single-trajectory reference computed
    # from that env's column alone — no cross-env leakage.
    for e in range(N):
        expected = _reference_gae(rewards[:, e], values[:, e], dones[:, e], last_values[e])
        np.testing.assert_allclose(buf.advantages[:, e], expected, rtol=1e-5, atol=1e-5)
    print("PASS test_gae: per-env advantages match single-trajectory reference")


def test_envs_are_independent():
    """Changing one env's rewards must not move another env's advantages."""
    T, N = 8, 2

    def run(reward_for_env1):
        buf = RolloutBuffer(T, N, obs_dim=2, action_dim=1)
        for t in range(T):
            buf.add(
                states=np.zeros((N, 2)),
                actions=np.zeros((N, 1)),
                rewards=np.array([1.0, reward_for_env1]),
                values=np.zeros(N),
                log_probs=np.zeros(N),
                dones=np.zeros(N),
            )
        buf.compute_gae(np.zeros(N), gamma=GAMMA, lam=LAM)
        return buf.advantages[:, 0].copy()

    np.testing.assert_allclose(run(5.0), run(-99.0), rtol=1e-6, atol=1e-6)
    print("PASS test_gae: envs are independent")


def test_done_truncates_bootstrap():
    """A terminal step must not bootstrap past the episode boundary."""
    T, N = 3, 1
    buf = RolloutBuffer(T, N, obs_dim=1, action_dim=1)
    dones = [0.0, 1.0, 0.0]
    for t in range(T):
        buf.add(
            states=np.zeros((N, 1)),
            actions=np.zeros((N, 1)),
            rewards=np.array([1.0]),
            values=np.array([10.0]),
            log_probs=np.zeros(N),
            dones=np.array([dones[t]]),
        )
    buf.compute_gae(np.array([10.0]), gamma=GAMMA, lam=LAM)

    # At t=1 (done), delta = r - V = 1 - 10 = -9 with no bootstrap term,
    # and no advantage carried back from t=2.
    assert abs(buf.advantages[1, 0] - (-9.0)) < 1e-5, buf.advantages[1, 0]
    print("PASS test_gae: done truncates bootstrapping")


if __name__ == "__main__":
    test_gae_matches_reference_per_env()
    test_envs_are_independent()
    test_done_truncates_bootstrap()
