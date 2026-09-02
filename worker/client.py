"""ParameterServerClient: the worker's TCP client to the parameter server.

Thin request/response wrappers over the shared framing helpers. All wire I/O
goes through comms.protocol.send_message / recv_message.
"""

import socket

from comms.protocol import MessageType, recv_message, send_message


class ParameterServerClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def pull_weights(self, worker_id: int) -> dict:
        send_message(self.sock, {"type": MessageType.PULL_WEIGHTS, "worker_id": worker_id})
        resp = recv_message(self.sock)
        assert resp["type"] == MessageType.WEIGHTS_RESPONSE, resp["type"]
        return resp["weights"]

    def push_gradients(self, worker_id: int, step: int, grads: dict) -> dict:
        send_message(
            self.sock,
            {
                "type": MessageType.PUSH_GRADIENTS,
                "worker_id": worker_id,
                "step": step,
                "gradients": grads,
            },
        )
        resp = recv_message(self.sock)
        assert resp["type"] == MessageType.ACK, resp["type"]
        return resp

    def heartbeat(self, worker_id: int, episode_reward: float, extra_metrics: dict | None = None) -> None:
        msg = {
            "type": MessageType.HEARTBEAT,
            "worker_id": worker_id,
            "episode_reward": episode_reward,
        }
        if extra_metrics:
            msg.update(extra_metrics)
        send_message(self.sock, msg)
        resp = recv_message(self.sock)
        assert resp["type"] == MessageType.HEARTBEAT_ACK, resp["type"]
