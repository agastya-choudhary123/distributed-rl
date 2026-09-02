# Distributed RL Training Framework — From Scratch

A transparent, from-scratch distributed reinforcement-learning trainer you can
read in a weekend and fully understand from **socket to weight update**. No Ray,
no gRPC, no ZeroMQ, no `torch.distributed`, no Gym. Just raw TCP sockets, a
hand-written wire protocol, manual gradient serialization, explicit
synchronization, a custom physics environment, and a live dashboard that exposes
the internals that black-box frameworks hide.

It trains a non-trivial continuous-control task (**SwingBot**, a pendulum
swing-up-and-balance) to convergence with distributed **PPO**, and anyone can
plug in their own environment in under 30 minutes.

## What makes it different

- **Raw TCP sockets** with a custom length-prefixed framing protocol — you see
  exactly what goes over the wire (`comms/protocol.py`).
- **Manual gradient serialization** — `loss.backward()` populates `.grad`, then
  those tensors are converted to numpy and shipped as bytes.
- **A parameter server you can actually read** — TCP server, per-worker threads,
  a lock-protected weight store, synchronous gradient aggregation with SGD +
  momentum applied *on the server*.
- **Heartbeat-based liveness** — a crashed worker is detected and excluded from
  the aggregation barrier so it can't stall training forever.
- **A live dashboard** — per-worker reward curves, gradient norms, weight-update
  frequency, and worker health, streamed over WebSockets.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Single Machine                               │
│  ┌──────────────────────── Parameter Server Process ─────────────┐   │
│  │   TCP Server Socket :9999                                     │   │
│  │   [worker 0 thread] [worker 1 thread] ... [worker N thread]   │   │
│  │                    │  (aggregation barrier)                   │   │
│  │              ┌─────▼──────┐   ┌──────────────┐                │   │
│  │              │ WeightStore│   │ Heartbeat    │                │   │
│  │              │ + Lock     │   │ Monitor      │                │   │
│  │              └─────┬──────┘   └──────────────┘                │   │
│  │              ┌─────▼──────┐                                    │   │
│  │              │MetricsQueue│───────────────────────┐           │   │
│  └──────────────────────────────────────────────────┼───────────┘   │
│                                                      │               │
│   [Worker 0 Process] [Worker 1 Process] ... [Worker N Process]       │
│    SwingBot + PPO      SwingBot + PPO         SwingBot + PPO          │
│    TCP client ─────────────── TCP :9999 ─────────────┘               │
│                                                      │               │
│   ┌────────── Dashboard Process (FastAPI) ───────────▼──────────┐    │
│   │   drains MetricsQueue → broadcasts JSON over WebSocket :8000 │    │
│   └──────────────────────────┬──────────────────────────────────┘    │
│                     Browser (dashboard.html + Chart.js)              │
└─────────────────────────────────────────────────────────────────────┘
```

Training and the dashboard are **fully decoupled**: metrics flow one-way
(parameter server → queue → FastAPI → WebSocket → browser). Kill the dashboard
or close the browser and training continues untouched.

## Quick start

```bash
pip install -r requirements.txt
python train.py                       # 4 workers + dashboard
# open http://127.0.0.1:8000 in a browser
```

Options:

```bash
python train.py --env cartpole            # pick any registered physics task
python train.py --num_workers 8
python train.py --envs_per_worker 8       # vectorized envs per worker
python train.py --no_dashboard            # training only
python train.py --device cuda             # auto | cuda | cpu | mps (see below)
python train.py --lr_schedule cosine --lr_warmup_steps 1000
python train.py --min_ready_workers 3     # straggler-tolerant aggregation
python train.py --max_steps 2000 --lr 3e-4 --rollout_length 512
```

One `python train.py` launches the parameter server, all workers, and the
dashboard as separate processes.

Training checkpoints to `checkpoints/latest.pkl` every `--checkpoint_interval`
steps and resumes from it automatically on the next run. Pass `--no_checkpoint`
to start fresh.

### Across containers

`docker-compose.yml` runs the parameter server and each worker as separate
containers with their own network namespaces, reaching each other by DNS name
over a bridge network — the same code path a real multi-machine cluster takes.

```bash
docker compose up --build     # server + 2 workers + dashboard on :8000
```

```
Worker 0 connected from ('172.18.0.3', 34402)
Worker 1 connected from ('172.18.0.4', 52204)
```

This is worth doing rather than running everything on loopback, because
loopback cannot exercise DNS resolution, per-node resource limits, or network
partition — and the heartbeat monitor and `--min_ready_workers` straggler
tolerance only do anything under exactly those conditions. Verified by killing
a node mid-run:

```bash
docker compose up -d
docker kill drl-worker-1      # SIGKILL, no clean disconnect
```

The survivor keeps training; the heartbeat monitor evicts the dead node from
the aggregation barrier so the run does not stall.

The dashboard runs inside the server container because it reads a
`multiprocessing.Queue`, which is shared memory and cannot cross a container
boundary.

## Any physics task, any policy

The framework is **not** tied to one environment or one action-space type. The
actor, PPO update, and rollout storage read three attributes off the env
(`observation_dim`, `action_dim`, `action_type`) and adapt automatically:

- **`action_type = "continuous"`** → a diagonal-Gaussian policy (mean head +
  learned log-std), actions are float vectors.
- **`action_type = "discrete"`** → a Categorical policy (logits head), actions
  are integer indices.

Four from-scratch physics envs ship in the registry (`env/registry.py`), and you
select one with `--env`:

| name           | obs dim | action space        | task                          |
|----------------|:------:|---------------------|-------------------------------|
| `swingbot`     | 4      | continuous (1)      | pendulum swing-up + balance   |
| `cartpole`     | 4      | discrete (2)        | balance a pole                |
| `mountain_car` | 2      | continuous (1)      | build momentum up a hill      |
| `acrobot`      | 6      | discrete (3)        | two-link underactuated swing  |

All four are hand-written classic-control physics (Euler / RK4), no Gym. They
exist to prove generality — both policy types and a range of state/action dims
train through the exact same distributed pipeline.

## The wire protocol

Every message is `[4-byte big-endian uint32 length][pickled payload]`. TCP is a
byte stream with no message boundaries, so both the header and the payload are
read in a loop until exactly the expected number of bytes arrive. `send_message`
and `recv_message` in `comms/protocol.py` are the **only** place raw socket
send/recv is called. Typed payloads (`"type"` key): `PUSH_GRADIENTS`,
`PULL_WEIGHTS`, `WEIGHTS_RESPONSE`, `ACK`, `HEARTBEAT`, `HEARTBEAT_ACK`.

## Training loop (per worker)

1. `PULL_WEIGHTS` → load the latest global weights; keep a copy as the base.
2. Collect a rollout of `rollout_length` steps across `envs_per_worker`
   environments stepped together, batching one forward pass per timestep.
3. Compute advantages with GAE (γ=0.99, λ=0.95), **independently per
   environment** — the rollout buffer is `(steps, envs, …)` so the backward
   recurrence never crosses an env boundary.
4. Run `ppo_epochs` passes of the clipped surrogate objective + value MSE +
   entropy bonus, taking a real Adam step per minibatch. Because the policy
   actually moves, the importance ratio departs from 1 and the clip does work.
5. `PUSH_GRADIENTS` carrying `base_weights - updated_weights` — the local
   update expressed as a pseudo-gradient.
6. Wait for `ACK`, heartbeat periodically, `PULL_WEIGHTS`, repeat.

The parameter server waits for all alive workers (synchronous aggregation),
averages their deltas, applies `w = w - server_lr·(momentum·v + avg_delta)`,
increments the global step, and ACKs everyone. With the defaults
(`server_lr=1`, `momentum=0`) this is exactly FedAvg: the new global weights
are the mean of the workers' locally-updated weights.

Two learning rates, two jobs: `--lr` is the worker's local Adam rate and is the
one that shapes training; `--server_lr` is only the blend factor for combining
worker updates.

Set `--min_ready_workers N` to apply an update as soon as N workers report
instead of waiting for all of them. This tolerates stragglers at the cost of
gradient staleness; it is off by default.

## Plug in your own environment (under 30 minutes)

1. Subclass `BaseEnv` (`env/base.py`):

```python
from env.base import BaseEnv
import numpy as np

class MyEnv(BaseEnv):
    observation_dim = 6
    action_type = "continuous"       # or "discrete"
    action_dim = 2                   # continuous: #components; discrete: #choices
    action_low, action_high = -1.0, 1.0   # continuous only

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        ...                      # return a float32 array of shape (observation_dim,)

    def step(self, action):
        # action is a float vector (continuous) or an int (discrete)
        ...                      # return (next_obs, reward, done, info)
```

2. Register it by name in `env/registry.py`:

```python
from my_env import MyEnv
ENVS = { ..., "myenv": MyEnv }
```

3. Run it:

```bash
python train.py --env myenv
```

That's it. The actor's policy head (Gaussian vs Categorical), the PPO
log-prob/entropy, and the rollout storage are all derived from
`observation_dim` / `action_dim` / `action_type`, so nothing else changes.

## Repository layout

```
comms/       protocol.py (framing), serialization.py (torch<->numpy)
network/     actor.py, critic.py, utils.py (weight/grad serialization)
env/         base.py (extension point), registry.py (name->class),
             swingbot.py, cartpole.py, mountain_car.py, acrobot.py (from-scratch physics)
worker/      worker.py (main loop), rollout_buffer.py (GAE), ppo.py, client.py
server/      parameter_server.py, weight_store.py, connection_handler.py,
             heartbeat_monitor.py, metrics_emitter.py
dashboard/   server.py (FastAPI), dashboard.html (Chart.js), metrics_schema.py
tests/       test_protocol, test_swingbot, test_serialization, test_ppo
config.py    every hyperparameter in one dataclass
train.py     launches everything
```

## Tests

```bash
python -m tests.test_protocol        # 100 messages round-trip exactly
python -m tests.test_swingbot        # physics + reward sanity
python -m tests.test_serialization   # weights survive the wire
python -m tests.test_ppo             # PPO produces non-zero gradients
python -m tests.test_envs            # all envs + both policy types train one step
python -m tests.test_gae             # advantages stay per-env; done truncates
```

`test_gae` guards the subtlest bug this codebase has had: flattening N envs
into one sequential buffer makes step `t` of env `i` bootstrap off env `i+1`,
corrupting every advantage while still producing plausible-looking loss curves.
See `CHANGELOG.md`.

## Measured results

All four environments learn (2 workers × 4 envs, 300 steps):

| env | action space | start | end |
|---|---|---:|---:|
| `swingbot` | continuous | −136 | **+144** |
| `cartpole` | discrete | 19 | **500** (task cap — solved) |
| `mountain_car` | continuous | −33 | **−0.19** |
| `acrobot` | discrete | −128 | **−83** |

Throughput on CPU, scaling environments (cartpole, `rollout_length=256`):

| workers | envs/worker | total envs | env-steps/s |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 5,120 |
| 1 | 4 | 4 | 11,377 |
| 2 | 4 | 8 | 18,618 |
| 4 | 4 | 16 | 37,236 |

7.3× from 1 → 16 envs; sub-linear because the aggregation barrier serializes.
See `CHANGELOG.md` for the full record, including what is *not* verified.

## Design notes & where to change things

- **Async aggregation.** `--min_ready_workers N` applies the update once N
  workers report rather than waiting for all. Introduces gradient staleness;
  the tradeoff is not characterized.
- **Faster serialization.** Pickle + numpy is fine on one machine. Swap in
  MessagePack/flatbuffers at the `comms/serialization.py` boundary; the numpy
  representation stays the same.
- **Multi-machine.** Point `server_host` at a routable address and open the
  port — this is what `docker-compose.yml` already exercises across containers.
- **GPU.** `--device cuda|mps` is plumbed through, but measured **slower** than
  CPU here (mps 3,103 vs cpu 20,480 env-steps/s): 64-unit MLPs with CPU-side
  env stepping pay more in per-timestep transfers than they gain in
  parallelism. CPU is the default for that reason. Scale the networks up before
  reaching for a GPU. CUDA specifically is untested.

## Out of scope (intentionally)

Faster serialization, hyperparameter sweeps, off-policy methods, and gradient
compression. Each has a clearly marked extension point above.
