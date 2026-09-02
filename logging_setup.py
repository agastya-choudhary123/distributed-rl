"""Logging configuration shared by every process.

With the "spawn" start method each child process gets a fresh interpreter, so
logging must be configured again inside the child. Every process entry point
(server_main, worker_main, dashboard_main) calls setup_logging() first;
otherwise child-process log records are silently dropped and failures look like
a process that "just exited".
"""

import logging
import os


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging. Safe to call once per process."""
    if level is None:
        level = os.environ.get("DRL_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(processName)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # child processes may inherit a half-configured root logger
    )
