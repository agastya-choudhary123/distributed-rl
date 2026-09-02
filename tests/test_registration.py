"""A connection only counts as a worker once it identifies itself.

The bug this guards against surfaced only under Docker: the compose healthcheck
opened a TCP connection every 2 seconds, and because the server assigned worker
ids by connection order, each probe registered as a brand-new live worker (177
of them in one run). A registered worker joins the aggregation barrier, so a
health probe could stall training.
"""

import socket
import threading
import time

from comms.protocol import MessageType, recv_message, send_message
from server.connection_handler import GradientAggregator, handle_worker
from server.heartbeat_monitor import HeartbeatMonitor
from server.metrics_emitter import MetricsEmitter
from server.weight_store import WeightStore


def _serve(sock, weight_store, aggregator, heartbeat):
    t = threading.Thread(
        target=handle_worker,
        args=(sock, ("test", 0), weight_store, aggregator, heartbeat,
              MetricsEmitter(None)),
        daemon=True,
    )
    t.start()
    return t


def _fixtures():
    import numpy as np
    weights = {"actor.w": np.zeros((2, 2), dtype=np.float32)}
    store = WeightStore(weights, lr=1.0, momentum=0.0)
    heartbeat = HeartbeatMonitor()
    aggregator = GradientAggregator(store, heartbeat, MetricsEmitter(None))
    return store, heartbeat, aggregator


def test_probe_that_never_speaks_is_not_a_worker():
    store, heartbeat, aggregator = _fixtures()
    server_sock, client_sock = socket.socketpair()
    t = _serve(server_sock, store, aggregator, heartbeat)

    # Exactly what the Docker healthcheck does: connect, then hang up.
    client_sock.close()
    t.join(timeout=2)

    assert heartbeat.alive_workers() == [], heartbeat.alive_workers()
    assert aggregator.sockets == {}, aggregator.sockets
    print("PASS test_registration: silent probe is not registered")


def test_worker_registers_under_its_own_id():
    store, heartbeat, aggregator = _fixtures()
    server_sock, client_sock = socket.socketpair()
    _serve(server_sock, store, aggregator, heartbeat)

    # A real worker announces its id; 7 must be honored, not a slot counter.
    send_message(client_sock, {"type": MessageType.PULL_WEIGHTS, "worker_id": 7})
    resp = recv_message(client_sock)
    assert resp["type"] == MessageType.WEIGHTS_RESPONSE, resp["type"]

    deadline = time.time() + 2
    while time.time() < deadline and 7 not in heartbeat.alive_workers():
        time.sleep(0.01)

    assert heartbeat.alive_workers() == [7], heartbeat.alive_workers()
    assert set(aggregator.sockets) == {7}, aggregator.sockets
    client_sock.close()
    print("PASS test_registration: worker registers under its announced id")


def test_probes_do_not_accumulate():
    """Many probes must leave the alive set empty, not grow it."""
    store, heartbeat, aggregator = _fixtures()
    threads = []
    for _ in range(25):
        s, c = socket.socketpair()
        threads.append(_serve(s, store, aggregator, heartbeat))
        c.close()
    for t in threads:
        t.join(timeout=2)

    assert heartbeat.alive_workers() == [], heartbeat.alive_workers()
    assert aggregator.sockets == {}, aggregator.sockets
    print("PASS test_registration: 25 probes registered 0 workers")


if __name__ == "__main__":
    test_probe_that_never_speaks_is_not_a_worker()
    test_worker_registers_under_its_own_id()
    test_probes_do_not_accumulate()
