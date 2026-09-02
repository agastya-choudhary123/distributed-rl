"""Parameter server process: TCP server, weight store, gradient aggregation.

Creates initial weights (seeded for reproducibility), opens a TCP listen socket,
and spawns one connection-handler thread per worker. Supports checkpointing and
resumption from saved states. The heartbeat monitor and metrics emitter run
alongside.
"""

import logging
import socket
import threading
import time

import torch

from logging_setup import setup_logging
from network.actor import Actor
from network.critic import Critic
from network.utils import serialize_weights
from server.connection_handler import GradientAggregator, handle_worker
from server.heartbeat_monitor import HeartbeatMonitor
from server.metrics_emitter import MetricsEmitter
from server.weight_store import WeightStore

logger = logging.getLogger(__name__)


def build_initial_weights(config) -> dict:
    """Create fresh initial weights, seeded for reproducibility."""
    torch.manual_seed(config.seed)
    env = config.env_class()
    actor = Actor(env.observation_dim, env.action_dim, env.action_type)
    critic = Critic(env.observation_dim)
    return serialize_weights(actor, critic)


def server_main(config, metrics_queue=None, ready_event=None) -> None:
    """Main parameter server loop."""
    setup_logging()
    weights = build_initial_weights(config)
    weight_store = WeightStore(
        weights, lr=config.server_lr, momentum=config.momentum, config=config
    )

    if config.load_checkpoint:
        latest_checkpoint = config.checkpoint_dir / "latest.pkl"
        weight_store.load_checkpoint(latest_checkpoint)

    heartbeat = HeartbeatMonitor()
    metrics = MetricsEmitter(metrics_queue)
    aggregator = GradientAggregator(weight_store, heartbeat, metrics, config=config)
    heartbeat.start()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((config.server_host, config.server_port))
    listener.listen(config.num_workers + 4)
    logger.info(f"Listening on {config.server_host}:{config.server_port}")

    if ready_event is not None:
        ready_event.set()

    try:
        while True:
            conn, addr = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # No id is assigned here: the connection announces its own
            # worker_id in its first message. Assigning ids by connection order
            # made every health probe look like a new worker.
            threading.Thread(
                target=handle_worker,
                args=(conn, addr, weight_store, aggregator, heartbeat, metrics),
                daemon=True,
            ).start()

    except KeyboardInterrupt:
        logger.info("Server shutting down")
    finally:
        heartbeat.stop()
        listener.close()
        final_checkpoint = config.checkpoint_dir / "latest.pkl"
        weight_store.save_checkpoint(final_checkpoint)
        logger.info("Server stopped")
