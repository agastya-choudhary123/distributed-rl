"""Typed shapes for the metrics payloads flowing server -> queue -> WebSocket.

These are documentation-grade TypedDicts; the pipeline passes plain dicts.
"""

from typing import TypedDict


class WeightUpdate(TypedDict):
    type: str  # "weight_update"
    global_step: int
    timestamp: float
    per_worker_grad_norms: dict  # {worker_id: float}
    worker_statuses: dict         # {worker_id: "ALIVE"|"DEAD"}


class Heartbeat(TypedDict):
    type: str  # "heartbeat"
    worker_id: int
    global_step: int
    timestamp: float
    episode_reward: float
    # Optional loss metrics (when available from worker).
    avg_actor_loss: float  # Actor (policy) loss
    avg_critic_loss: float  # Critic (value) loss
    avg_entropy: float  # Policy entropy
    recent_episodes: int  # Number of episodes in recent window
