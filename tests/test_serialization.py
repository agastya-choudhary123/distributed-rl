"""Step 4: weights survive serialize -> numpy dict -> deserialize identically."""

import torch

from network.actor import Actor
from network.critic import Critic
from network.utils import deserialize_weights, serialize_weights


def test_weight_roundtrip():
    a1, c1 = Actor(4, 1, "continuous"), Critic(4)
    a2, c2 = Actor(4, 1, "continuous"), Critic(4)  # different random init

    obs = torch.randn(8, 4)
    before = a2.distribution(obs).mean
    deserialize_weights(serialize_weights(a1, c1), a2, c2)
    after = a2.distribution(obs).mean

    # a2 now equals a1.
    assert torch.allclose(a1.distribution(obs).mean, after, atol=1e-6)
    assert not torch.allclose(before, after)  # weights actually changed
    print("PASS test_serialization: weights round-trip and match source model")


if __name__ == "__main__":
    test_weight_roundtrip()
