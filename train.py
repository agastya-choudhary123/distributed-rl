"""Entry point: parse config, launch the parameter server, workers, and dashboard.

    python train.py                 # 4 workers + dashboard on defaults
    python train.py --num_workers 8 --device cuda
    python train.py --no_dashboard  # training only

All processes are children of this one. The MetricsQueue is the single non-TCP
cross-process channel, shared with the server and dashboard.
"""

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

from config import TrainConfig
from dashboard.server import dashboard_main
from env.registry import ENVS, get_env
from server.parameter_server import server_main
from worker.worker import worker_main


logger = logging.getLogger(__name__)


def parse_args(config: TrainConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed RL training framework")
    p.add_argument("--env", type=str, default="swingbot", choices=sorted(ENVS),
                   help="which physics task to train on")
    p.add_argument("--num_workers", type=int, default=config.num_workers)
    p.add_argument("--envs_per_worker", type=int, default=config.envs_per_worker,
                   help="vectorized environments per worker")
    p.add_argument("--rollout_length", type=int, default=config.rollout_length)
    p.add_argument("--ppo_epochs", type=int, default=config.ppo_epochs)
    p.add_argument("--lr", type=float, default=config.lr)
    p.add_argument("--lr_schedule", type=str, default=config.lr_schedule,
                   choices=["constant", "linear", "cosine"])
    p.add_argument("--max_steps", type=int, default=config.max_steps)
    p.add_argument("--server_port", type=int, default=config.server_port)
    p.add_argument("--dashboard_port", type=int, default=config.dashboard_port)
    p.add_argument("--device", type=str, default=config.device,
                   choices=["auto", "cuda", "cpu", "mps"],
                   help="auto picks cuda > mps > cpu")
    p.add_argument("--seed", type=int, default=config.seed)
    p.add_argument("--checkpoint_dir", type=str, default=str(config.checkpoint_dir))
    p.add_argument("--checkpoint_interval", type=int, default=config.checkpoint_interval,
                   help="save checkpoint every N global steps")
    p.add_argument("--no_checkpoint", action="store_true", dest="no_load_checkpoint")
    p.add_argument("--no_dashboard", action="store_true")
    p.add_argument("--min_ready_workers", type=int, default=config.min_ready_workers,
                   help="async aggregation: apply update if >= N workers ready (0=sync)")
    return p.parse_args()


def setup_logging():
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> None:
    setup_logging()
    config = TrainConfig()
    args = parse_args(config)

    config.env_class = get_env(args.env)
    config.num_workers = args.num_workers
    config.envs_per_worker = args.envs_per_worker
    config.rollout_length = args.rollout_length
    config.ppo_epochs = args.ppo_epochs
    config.lr = args.lr
    config.lr_schedule = args.lr_schedule
    config.max_steps = args.max_steps
    config.server_port = args.server_port
    config.dashboard_port = args.dashboard_port
    config.device = args.device
    config.seed = args.seed
    # Resolve once in the parent so every worker gets the same concrete device.
    config.device = config.resolve_device()
    config.checkpoint_dir = Path(args.checkpoint_dir)
    config.checkpoint_interval = args.checkpoint_interval
    config.load_checkpoint = not args.no_load_checkpoint
    config.min_ready_workers = args.min_ready_workers

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting training: env={args.env}, workers={config.num_workers}, "
        f"envs_per_worker={config.envs_per_worker}, device={config.device}, "
        f"total_envs={config.total_parallel_envs()}, lr_schedule={config.lr_schedule}"
    )

    metrics_queue = mp.Queue(maxsize=10000)
    ready = mp.Event()
    procs: list[mp.Process] = []
    workers: list[mp.Process] = []

    server = mp.Process(target=server_main, args=(config, metrics_queue, ready),
                        name="param-server")
    server.start()
    procs.append(server)
    ready.wait(timeout=30)
    logger.info("Parameter server ready")

    if not args.no_dashboard:
        dash = mp.Process(target=dashboard_main, args=(config, metrics_queue),
                          name="dashboard")
        dash.start()
        procs.append(dash)
        logger.info(f"Dashboard at http://127.0.0.1:{config.dashboard_port}")

    for wid in range(config.num_workers):
        w = mp.Process(target=worker_main, args=(wid, config), name=f"worker-{wid}")
        w.start()
        procs.append(w)
        workers.append(w)

    logger.info(f"Launched {config.num_workers} workers")

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1.0)
        logger.info("All workers finished; shutting down")
    except KeyboardInterrupt:
        logger.info("Interrupted; terminating children")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        logger.info("Cleanup complete")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
