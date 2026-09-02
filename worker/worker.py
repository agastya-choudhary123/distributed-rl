"""Worker process: local rollout collection + PPO gradients + comms.

Each worker runs its own full PPO loop but never applies an optimizer step. It
computes gradients, ships them, waits for the ACK, then pulls the freshly
updated global weights. The parameter server owns the update.

Supports:
- Vectorized environments (batch N envs per worker).
- GPU training (tensors on device).
- Loss tracking and reporting.
- Advantage normalization.
"""

import logging
import math
import time

import numpy as np
import torch

from config import scheduled_lr
from env.vectorized import VectorizedEnv
from logging_setup import setup_logging
from network.actor import Actor
from network.critic import Critic
from network.utils import deserialize_weights, weight_delta
from worker.client import ParameterServerClient
from worker.ppo import compute_gradients
from worker.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 5.0  # seconds
PROGRESS_EVERY = 10       # global steps between INFO progress lines


def worker_main(worker_id: int, config) -> None:
    """Main worker training loop."""
    setup_logging()
    logger.info(f"Worker {worker_id} starting (device={config.device})")

    torch.manual_seed(config.seed + worker_id)
    np.random.seed(config.seed + worker_id)

    device = torch.device(config.device)

    vec_env = VectorizedEnv(
        config.env_class,
        num_envs=config.envs_per_worker,
        seed=config.seed + worker_id * 1000,
    )

    actor = Actor(
        vec_env.observation_dim,
        vec_env.action_dim,
        vec_env.action_type,
    ).to(device)
    critic = Critic(vec_env.observation_dim).to(device)

    action_store_dim = (
        vec_env.action_dim
        if vec_env.action_type == "continuous"
        else 1
    )

    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    client = ParameterServerClient(config.server_host, config.server_port)
    _connect_with_retry(client, worker_id)

    base_weights = client.pull_weights(worker_id)
    deserialize_weights(base_weights, actor, critic)

    last_heartbeat = 0.0
    global_step = 0
    loss_history = {"actor": [], "critic": [], "entropy": []}

    obs = vec_env.reset()  # shape: (num_envs, obs_dim)
    ep_returns = np.zeros(config.envs_per_worker)
    step_counts = np.zeros(config.envs_per_worker, dtype=int)
    recent_episode_returns = []

    while global_step < config.max_steps:
        buffer = RolloutBuffer(
            config.rollout_length,
            config.envs_per_worker,
            vec_env.observation_dim,
            action_store_dim,
        )

        # --- Collect a rollout (across all vectorized envs) ----------------
        for step_in_rollout in range(config.rollout_length):
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            )  # (num_envs, obs_dim)

            with torch.no_grad():
                # Batch evaluation: one forward pass covers every env.
                dist = actor.distribution(obs_t)
                values = critic(obs_t)  # (num_envs,)

                if actor.action_type == "continuous":
                    actions_t = dist.sample()                    # (N, action_dim)
                    log_probs = dist.log_prob(actions_t).sum(dim=-1)
                    env_actions = actions_t.cpu().numpy()
                    stored_actions = env_actions
                else:
                    actions_t = dist.sample()                    # (N,) already
                    log_probs = dist.log_prob(actions_t)
                    env_actions = actions_t.long().cpu().numpy()
                    # Stored as a float column so one buffer shape serves both
                    # action types; Actor.evaluate squeezes it back to indices.
                    stored_actions = env_actions.astype(np.float32).reshape(-1, 1)

            next_obs, rewards, dones, _ = vec_env.step(env_actions)

            # One timestep, all envs at once — keeps each env on its own axis
            # so GAE can run per-env.
            buffer.add(
                obs,
                stored_actions,
                rewards,
                values.detach().cpu().numpy(),
                log_probs.detach().cpu().numpy(),
                dones,
            )

            ep_returns += rewards
            step_counts += 1

            for env_idx in range(config.envs_per_worker):
                if dones[env_idx]:
                    recent_episode_returns.append(float(ep_returns[env_idx]))
                    ep_returns[env_idx] = 0.0
                    step_counts[env_idx] = 0

            obs = next_obs

        # --- Bootstrap final value (one per env) ---------------------------
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            last_values = critic(obs_t).detach().cpu().numpy()

        buffer.compute_gae(last_values, gamma=config.gamma, lam=config.gae_lambda)

        # --- Local PPO update ----------------------------------------------
        current_lr = scheduled_lr(
            config.lr, global_step, config.max_steps,
            config.lr_schedule, config.lr_warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # A real optimizer step per minibatch, so the policy moves between
        # epochs and the importance ratio actually departs from 1 — which is
        # what makes the clipped surrogate objective do anything.
        epoch_losses = {"actor": [], "critic": [], "entropy": []}

        for epoch in range(config.ppo_epochs):
            for batch in buffer.get_minibatches(config.minibatch_size):
                optimizer.zero_grad(set_to_none=True)
                loss_stats = compute_gradients(
                    actor,
                    critic,
                    batch,
                    clip_eps=config.clip_eps,
                    vf_coef=config.vf_coef,
                    ent_coef=config.ent_coef,
                    device=device,
                )
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                optimizer.step()

                epoch_losses["actor"].append(loss_stats["actor_loss"])
                epoch_losses["critic"].append(loss_stats["critic_loss"])
                epoch_losses["entropy"].append(loss_stats["entropy"])

        # Track average losses per epoch.
        avg_actor_loss = float(np.mean(epoch_losses["actor"]))
        avg_critic_loss = float(np.mean(epoch_losses["critic"]))
        avg_entropy = float(np.mean(epoch_losses["entropy"]))

        loss_history["actor"].append(avg_actor_loss)
        loss_history["critic"].append(avg_critic_loss)
        loss_history["entropy"].append(avg_entropy)
        loss_history["actor"] = loss_history["actor"][-100:]
        loss_history["critic"] = loss_history["critic"][-100:]
        loss_history["entropy"] = loss_history["entropy"][-100:]

        # --- Ship the local update as a pseudo-gradient, wait for ACK -------
        delta = weight_delta(base_weights, actor, critic)
        client.push_gradients(worker_id, global_step, delta)

        # --- Periodically send heartbeat -----------------------------------
        now = time.time()
        if now - last_heartbeat > HEARTBEAT_INTERVAL:
            mean_return = (
                float(np.mean(recent_episode_returns))
                if recent_episode_returns
                else 0.0
            )
            client.heartbeat(
                worker_id,
                mean_return,
                extra_metrics={
                    "avg_actor_loss": avg_actor_loss,
                    "avg_critic_loss": avg_critic_loss,
                    "avg_entropy": avg_entropy,
                    "recent_episodes": len(recent_episode_returns),
                },
            )
            last_heartbeat = now
            recent_episode_returns = recent_episode_returns[-20:]

        if global_step % PROGRESS_EVERY == 0:
            mean_return = (
                float(np.mean(recent_episode_returns))
                if recent_episode_returns
                else float("nan")
            )
            logger.info(
                f"worker {worker_id} step {global_step}/{config.max_steps} "
                f"return={mean_return:8.2f} "
                f"actor_loss={avg_actor_loss:+.4f} "
                f"critic_loss={avg_critic_loss:.4f} "
                f"entropy={avg_entropy:.3f}"
            )

        # --- Pull the new global weights; they become the next base ---------
        base_weights = client.pull_weights(worker_id)
        deserialize_weights(base_weights, actor, critic)

        global_step += 1

    logger.info(f"Worker {worker_id} finished ({global_step} steps)")


def _connect_with_retry(
    client: ParameterServerClient,
    worker_id: int,
    max_retries: int = 60,
    delay: float = 0.5,
) -> None:
    """Connect to the parameter server, retrying while it is still coming up."""
    for attempt in range(max_retries):
        try:
            client.connect()
            logger.info(f"Worker {worker_id} connected to server")
            return
        except OSError as e:
            logger.debug(
                f"Worker {worker_id} connect attempt {attempt + 1}/{max_retries} "
                f"failed ({e}); retrying in {delay}s"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Worker {worker_id} failed to connect after {max_retries} attempts"
    )
