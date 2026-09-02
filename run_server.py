#!/usr/bin/env python3
"""Parameter-server container entry point.

Runs the parameter server, and optionally the dashboard alongside it. The
dashboard lives here rather than in its own container because it reads the
metrics through a multiprocessing.Queue, which is shared memory — it cannot
cross a container boundary. Keeping both in one container keeps the queue in
one process tree; the workers stay in their own containers and talk TCP.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

from config import TrainConfig
from env.registry import ENVS, get_env
from logging_setup import setup_logging
from server.parameter_server import server_main


def parse_args(defaults: TrainConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter server")
    p.add_argument("--env", type=str, default="swingbot", choices=sorted(ENVS),
                   help="must match what the workers run — it sets the network shapes")
    p.add_argument("--num_workers", type=int, default=defaults.num_workers)
    p.add_argument("--server_host", type=str, default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=defaults.server_port)
    p.add_argument("--checkpoint_dir", type=str, default=str(defaults.checkpoint_dir))
    p.add_argument("--checkpoint_interval", type=int, default=defaults.checkpoint_interval)
    p.add_argument("--no_checkpoint", action="store_true", dest="no_load_checkpoint")
    p.add_argument("--max_steps", type=int, default=defaults.max_steps)
    p.add_argument("--server_lr", type=float, default=defaults.server_lr)
    p.add_argument("--momentum", type=float, default=defaults.momentum)
    p.add_argument("--min_ready_workers", type=int, default=defaults.min_ready_workers)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--dashboard", action="store_true",
                   help="also serve the live dashboard from this container")
    p.add_argument("--dashboard_port", type=int, default=defaults.dashboard_port)
    return p.parse_args()


def main() -> None:
    setup_logging()
    defaults = TrainConfig()
    args = parse_args(defaults)

    config = TrainConfig()
    config.env_class = get_env(args.env)
    config.num_workers = args.num_workers
    config.server_host = args.server_host
    config.server_port = args.server_port
    config.checkpoint_dir = Path(args.checkpoint_dir)
    config.checkpoint_interval = args.checkpoint_interval
    config.load_checkpoint = not args.no_load_checkpoint
    config.max_steps = args.max_steps
    config.server_lr = args.server_lr
    config.momentum = args.momentum
    config.min_ready_workers = args.min_ready_workers
    config.dashboard_port = args.dashboard_port
    config.seed = args.seed
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_queue = None
    if args.dashboard:
        from dashboard.server import dashboard_main
        metrics_queue = mp.Queue(maxsize=10000)
        mp.Process(target=dashboard_main, args=(config, metrics_queue),
                   name="dashboard", daemon=True).start()

    server_main(config, metrics_queue)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
