"""MetricsEmitter: puts metrics dicts onto the (multiprocessing) MetricsQueue.

The queue is the only cross-process channel that is not TCP. It is drained by
the dashboard process. Emitting never blocks training: if the queue is full or
absent, metrics are dropped silently.

Tracks: per-worker rewards, loss statistics (actor/critic/entropy), gradient
norms, and worker liveness.
"""

import time


class MetricsEmitter:
    def __init__(self, queue):
        self.queue = queue
        self.recent_rewards = {}  # worker_id -> list of recent rewards

    def _put(self, payload: dict) -> None:
        if self.queue is None:
            return
        try:
            self.queue.put_nowait(payload)
        except Exception:
            pass

    def weight_update(self, global_step, grad_norms, worker_statuses) -> None:
        """Called after each gradient aggregation and update."""
        self._put(
            {
                "type": "weight_update",
                "global_step": global_step,
                "timestamp": time.time(),
                "per_worker_grad_norms": grad_norms,
                "worker_statuses": worker_statuses,
            }
        )

    def heartbeat(self, worker_id, global_step, episode_reward, extra_metrics=None) -> None:
        """Called when a worker sends a heartbeat."""
        msg = {
            "type": "heartbeat",
            "worker_id": worker_id,
            "global_step": global_step,
            "timestamp": time.time(),
            "episode_reward": episode_reward,
        }
        if extra_metrics:
            msg.update(extra_metrics)
        self._put(msg)
