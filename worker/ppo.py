"""PPO loss and gradient computation. No optimizer step happens here.

compute_gradients() runs one minibatch: clipped surrogate objective + value MSE
+ entropy bonus, then loss.backward() to ACCUMULATE gradients into .grad. The
caller zeros grads once before the epochs and reads .grad after all minibatches.
The parameter server, not the worker, applies the update.
"""

import torch


def compute_gradients(
    actor,
    critic,
    batch: dict,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    device: str = "cpu",
    scale: float = 1.0,
) -> dict:
    """Accumulate gradients for one minibatch.

    `scale` divides the loss so that summing over every minibatch of every epoch
    yields the MEAN gradient rather than the sum. Without it the gradient
    magnitude silently scales with ppo_epochs and rollout_length.
    """
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

    new_log_probs, entropy_per_sample = actor.evaluate(states, actions)
    entropy = entropy_per_sample.mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    values = critic(states)
    critic_loss = ((values - returns) ** 2).mean()

    total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
    (total_loss * scale).backward()

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(entropy.item()),
        "total_loss": float(total_loss.item()),
    }
