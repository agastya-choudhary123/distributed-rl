"""Weight/gradient serialization between actor+critic models and numpy dicts.

Keys are prefixed with "actor." / "critic." so a single flat dict carries both
networks over the wire. The parameter server never needs to know the network
topology beyond these string keys.
"""

import numpy as np

from comms.serialization import numpy_to_state_dict, state_dict_to_numpy


def serialize_weights(actor, critic) -> dict:
    """actor.state_dict() + critic.state_dict() -> prefixed numpy dict."""
    out = {}
    for k, v in state_dict_to_numpy(actor.state_dict()).items():
        out[f"actor.{k}"] = v
    for k, v in state_dict_to_numpy(critic.state_dict()).items():
        out[f"critic.{k}"] = v
    return out


def deserialize_weights(weights_dict: dict, actor, critic) -> None:
    """Load a prefixed numpy dict back into actor and critic in place."""
    actor_np = {
        k[len("actor.") :]: v
        for k, v in weights_dict.items()
        if k.startswith("actor.")
    }
    critic_np = {
        k[len("critic.") :]: v
        for k, v in weights_dict.items()
        if k.startswith("critic.")
    }
    actor.load_state_dict(numpy_to_state_dict(actor_np))
    critic.load_state_dict(numpy_to_state_dict(critic_np))


def weight_delta(old_weights: dict, actor, critic) -> dict:
    """Return old_weights - current_weights as a prefixed numpy dict.

    This is the "pseudo-gradient" a worker ships after running its local PPO
    update. The server applies `w <- w - server_lr * avg_delta`, so with
    server_lr=1 the global weights become the mean of the workers' locally
    updated weights (FedAvg).
    """
    new_weights = serialize_weights(actor, critic)
    return {k: (old_weights[k] - new_weights[k]) for k in new_weights}


def serialize_gradients(actor, critic) -> dict:
    """Extract .grad from every parameter -> prefixed numpy dict.

    Parameters untouched by the graph have grad=None; substitute zeros of the
    correct shape so the parameter server can average uniformly.
    """
    out = {}
    for prefix, model in (("actor", actor), ("critic", critic)):
        for name, p in model.named_parameters():
            if p.grad is None:
                arr = np.zeros(tuple(p.shape), dtype=np.float32)
            else:
                arr = p.grad.detach().cpu().numpy()
            out[f"{prefix}.{name}"] = arr
    return out
