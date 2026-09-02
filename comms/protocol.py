"""Lowest-level wire protocol: length-prefixed message framing over raw TCP.

Every message on the wire is:

    [4-byte big-endian uint32 length][pickled payload bytes]

TCP is a byte stream with no message boundaries, so both the header and the
payload must be read in a loop until exactly the expected number of bytes have
been accumulated. These two functions are the ONLY place in the codebase where
raw socket send/recv is called.
"""

import pickle
import socket
import struct


class ConnectionClosed(Exception):
    """Raised when the peer closes the socket mid-message."""


class MessageType:
    """String message-type tags. Every payload dict has a "type" key."""

    PUSH_GRADIENTS = "PUSH_GRADIENTS"
    PULL_WEIGHTS = "PULL_WEIGHTS"
    WEIGHTS_RESPONSE = "WEIGHTS_RESPONSE"
    ACK = "ACK"
    HEARTBEAT = "HEARTBEAT"
    HEARTBEAT_ACK = "HEARTBEAT_ACK"


_HEADER = struct.Struct(">I")  # 4-byte big-endian unsigned int


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock, looping over recv (TCP may fragment)."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionClosed("peer closed the connection")
        data.extend(chunk)
    return bytes(data)


def send_message(sock: socket.socket, payload_dict: dict) -> None:
    """Pickle the payload, prefix its length, and send it in full."""
    data = pickle.dumps(payload_dict, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_message(sock: socket.socket) -> dict:
    """Read one framed message and return the unpickled payload dict."""
    header = _recv_exactly(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    payload = _recv_exactly(sock, length)
    return pickle.loads(payload)
