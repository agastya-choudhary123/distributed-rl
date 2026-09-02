"""Step 3: sanity-check SwingBot physics and reward under a random policy."""

import numpy as np

from env.swingbot import SwingBot


def test_random_policy():
    env = SwingBot(seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    total = 0.0
    n = 0
    for i in range(1000):
        a = np.random.uniform(-1, 1, size=(1,))
        obs, r, done, info = env.step(a)
        total += r
        n += 1
        if i % 200 == 0:
            print(env.render_ascii())
        if done:
            env.reset()
    mean_r = total / n
    print(f"mean reward under random policy: {mean_r:.3f} (expect roughly [-2, 0.5])")
    assert -3.0 < mean_r < 1.0
    print("PASS test_swingbot")


if __name__ == "__main__":
    test_random_policy()
