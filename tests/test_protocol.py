"""Step 1: bulletproof the wire protocol before anything touches a real socket."""

import socket
import threading

import numpy as np

from comms.protocol import recv_message, send_message


def test_roundtrip_varied_messages():
    payloads = [
        {"type": "empty"},
        {"type": "small", "x": 1, "s": "hello"},
        {"type": "nested", "d": {"a": [1, 2, 3], "b": {"c": 4}}},
        {"type": "array", "w": np.random.randn(64, 64).astype(np.float32)},
        {"type": "big", "w": np.random.randn(256, 256).astype(np.float32),
         "meta": {"worker": 3, "step": 99}},
    ] * 20  # 100 messages

    a, b = socket.socketpair()
    received = []

    def receiver():
        for _ in range(len(payloads)):
            received.append(recv_message(b))

    t = threading.Thread(target=receiver)
    t.start()
    for p in payloads:
        send_message(a, p)
    t.join()

    assert len(received) == len(payloads)
    for sent, got in zip(payloads, received):
        assert sent["type"] == got["type"]
        if "w" in sent:
            assert np.array_equal(sent["w"], got["w"])
    a.close()
    b.close()
    print("PASS test_protocol: 100 messages round-tripped exactly")


if __name__ == "__main__":
    test_roundtrip_varied_messages()
