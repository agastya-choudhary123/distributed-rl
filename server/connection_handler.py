"""Per-worker connection thread + gradient aggregator.

Supports both synchronous (wait for all workers) and async (straggler-tolerant)
aggregation. The GradientAggregator batches gradients and applies updates either
when all alive workers report (sync) or when >=min_ready_workers report (async).

Lock ordering is fixed: aggregator buffer lock first, weight-store lock second.
"""

import logging
import threading

import numpy as np

from comms.protocol import ConnectionClosed, MessageType, recv_message, send_message

logger = logging.getLogger(__name__)


def _grad_norm(grads: dict) -> float:
    total = 0.0
    for arr in grads.values():
        total += float(np.sum(np.square(arr)))
    return float(np.sqrt(total))


class GradientAggregator:
    """Gradient barrier supporting synchronous and async (straggler-tolerant) aggregation."""

    def __init__(self, weight_store, heartbeat_monitor, metrics_emitter, config=None):
        self.weight_store = weight_store
        self.heartbeat = heartbeat_monitor
        self.metrics = metrics_emitter
        self.config = config
        self.lock = threading.Lock()
        self.buffer: dict[int, dict] = {}
        self.norms: dict[int, float] = {}
        self.sockets: dict[int, object] = {}
        self.last_checkpoint_step = weight_store.get_step()

    def register_socket(self, worker_id, sock) -> None:
        with self.lock:
            self.sockets[worker_id] = sock

    def unregister(self, worker_id) -> None:
        with self.lock:
            self.sockets.pop(worker_id, None)
            self.buffer.pop(worker_id, None)
            self.norms.pop(worker_id, None)

    def submit(self, worker_id, grads) -> None:
        """Add gradients; apply update if threshold is met.

        Synchronous (min_ready_workers=0): wait for all alive workers.
        Async (min_ready_workers=N>0): apply when >=N workers ready (straggler tolerance).
        """
        with self.lock:
            self.buffer[worker_id] = grads
            self.norms[worker_id] = _grad_norm(grads)

            alive = set(self.heartbeat.alive_workers())
            have = set(self.buffer.keys())
            ready_count = len(have & alive)

            # Determine if we should apply an update.
            should_apply = False
            min_ready = 0
            if self.config:
                min_ready = self.config.min_ready_workers

            if min_ready <= 0:
                waiting_on = alive.intersection(self.sockets.keys()) - have
                should_apply = not waiting_on
            else:
                should_apply = ready_count >= min_ready

            if not should_apply:
                return

            grads_list = list(self.buffer.values())
            round_norms = dict(self.norms)
            round_sockets = dict(self.sockets)
            self.buffer.clear()
            self.norms.clear()

        new_weights = self.weight_store.apply_gradients(grads_list)
        step = self.weight_store.get_step()

        for sock in round_sockets.values():
            try:
                send_message(sock, {"type": MessageType.ACK, "global_step": step})
            except OSError:
                pass

        self.metrics.weight_update(step, round_norms, self.heartbeat.snapshot())
        self._maybe_checkpoint(step)

    def _maybe_checkpoint(self, step: int) -> None:
        """Save a checkpoint every checkpoint_interval global steps.

        This lives on the update path, not the accept() loop: once every worker
        has connected the server blocks in accept() forever, so checkpointing
        from there would never fire.
        """
        if self.config is None:
            return
        if step - self.last_checkpoint_step < self.config.checkpoint_interval:
            return
        self.last_checkpoint_step = step
        self.weight_store.save_checkpoint(self.config.checkpoint_dir / "latest.pkl")


def handle_worker(sock, addr, weight_store, aggregator,
                  heartbeat_monitor, metrics_emitter):
    """Thread body for a single worker connection.

    The connection is NOT registered as a worker until it sends a message
    carrying its own worker_id. Anything that merely opens a socket and hangs
    up — a health probe, a port scan — is therefore never counted as alive.
    That matters: a registered worker joins the aggregation barrier, so
    treating a healthcheck as a worker lets it stall training.
    """
    worker_id = None

    try:
        while True:
            msg = recv_message(sock)
            mtype = msg["type"]

            if worker_id is None:
                worker_id = msg.get("worker_id")
                if worker_id is None:
                    logger.warning(f"{addr} sent {mtype} without a worker_id; dropping")
                    return
                heartbeat_monitor.register(worker_id)
                aggregator.register_socket(worker_id, sock)
                logger.info(f"Worker {worker_id} connected from {addr}")

            if mtype == MessageType.PULL_WEIGHTS:
                send_message(sock, {
                    "type": MessageType.WEIGHTS_RESPONSE,
                    "weights": weight_store.get_weights(),
                })

            elif mtype == MessageType.PUSH_GRADIENTS:
                aggregator.submit(worker_id, msg["gradients"])

            elif mtype == MessageType.HEARTBEAT:
                heartbeat_monitor.beat(worker_id)
                step = weight_store.get_step()
                reward = msg.get("episode_reward", 0.0)
                extra = {k: v for k, v in msg.items()
                        if k not in ("type", "worker_id", "episode_reward")}
                metrics_emitter.heartbeat(worker_id, step, reward, extra)
                send_message(sock, {"type": MessageType.HEARTBEAT_ACK})

            else:
                logger.warning(f"Worker {worker_id} sent unknown message type: {mtype}")

    except (ConnectionClosed, OSError) as e:
        if worker_id is None:
            logger.debug(f"{addr} closed before identifying itself ({e}); ignored")
        else:
            logger.info(f"Worker {worker_id} disconnected: {e}")
    finally:
        if worker_id is not None:
            heartbeat_monitor.mark_dead(worker_id)
            aggregator.unregister(worker_id)
        try:
            sock.close()
        except OSError:
            pass
