#!/usr/bin/env python3
"""Worker container entry point: one worker process, dialing a remote server."""

import argparse

from config import TrainConfig
from env.registry import ENVS, get_env
from logging_setup import setup_logging
from worker.worker import worker_main


def parse_args(defaults: TrainConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training worker")
    p.add_argument("worker_id", type=int)
    p.add_argument("--env", type=str, default="swingbot", choices=sorted(ENVS))
    p.add_argument("--server_host", type=str, default="parameter-server")
    p.add_argument("--server_port", type=int, default=defaults.server_port)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu", "mps"],
                   help="auto picks cuda > mps > cpu")
    p.add_argument("--envs_per_worker", type=int, default=defaults.envs_per_worker)
    p.add_argument("--rollout_length", type=int, default=defaults.rollout_length)
    p.add_argument("--ppo_epochs", type=int, default=defaults.ppo_epochs)
    p.add_argument("--minibatch_size", type=int, default=defaults.minibatch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--lr_schedule", type=str, default=defaults.lr_schedule,
                   choices=["constant", "linear", "cosine"])
    p.add_argument("--max_steps", type=int, default=defaults.max_steps)
    p.add_argument("--seed", type=int, default=defaults.seed)
    return p.parse_args()


def main() -> None:
    setup_logging()
    defaults = TrainConfig()
    args = parse_args(defaults)

    config = TrainConfig()
    config.env_class = get_env(args.env)
    config.server_host = args.server_host
    config.server_port = args.server_port
    config.device = args.device
    config.device = config.resolve_device()
    config.envs_per_worker = args.envs_per_worker
    config.rollout_length = args.rollout_length
    config.ppo_epochs = args.ppo_epochs
    config.minibatch_size = args.minibatch_size
    config.lr = args.lr
    config.lr_schedule = args.lr_schedule
    config.max_steps = args.max_steps
    config.seed = args.seed

    worker_main(args.worker_id, config)


if __name__ == "__main__":
    main()
