"""HeartbeatMonitor: tracks worker liveness in a background thread.

Every check_interval seconds it scans last_seen and marks any worker silent for
longer than timeout as DEAD. Synchronous aggregation waits only for ALIVE
workers, so a crashed worker cannot block training forever.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    ALIVE = "ALIVE"
    DEAD = "DEAD"

    def __init__(self, check_interval: float = 10.0, timeout: float = 30.0):
        self.check_interval = check_interval
        self.timeout = timeout
        self.lock = threading.Lock()
        self.last_seen: dict[int, float] = {}
        self.status: dict[int, str] = {}
        self._stop = threading.Event()

    def register(self, worker_id: int) -> None:
        with self.lock:
            self.last_seen[worker_id] = time.time()
            self.status[worker_id] = self.ALIVE

    def beat(self, worker_id: int) -> None:
        with self.lock:
            self.last_seen[worker_id] = time.time()
            self.status[worker_id] = self.ALIVE

    def mark_dead(self, worker_id: int) -> None:
        with self.lock:
            self.status[worker_id] = self.DEAD

    def alive_workers(self) -> list[int]:
        with self.lock:
            return [w for w, s in self.status.items() if s == self.ALIVE]

    def snapshot(self) -> dict[int, str]:
        with self.lock:
            return dict(self.status)

    def _loop(self) -> None:
        while not self._stop.wait(self.check_interval):
            now = time.time()
            with self.lock:
                for w, t in self.last_seen.items():
                    if now - t > self.timeout and self.status.get(w) == self.ALIVE:
                        self.status[w] = self.DEAD
                        logger.warning(f"Worker {w} marked DEAD (silent {now - t:.0f}s)")

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
