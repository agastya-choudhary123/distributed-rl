distributed-rl
--------------

distributed-rl is a distributed PPO trainer built on raw TCP sockets. No Ray, no
gRPC, no ZeroMQ, no `torch.distributed`, no Gym. The wire protocol, gradient
serialization, synchronization, physics environments and dashboard are all in
this tree, so the path from socket to weight update is readable end to end.

It trains a continuous-control task to convergence and ships four from-scratch
environments covering both continuous and discrete policies.

### Documentation quick links

* [Quick start](#quick-start)
* [Environments](#environments)
* [Results](#results)
* [Adding an environment](#adding-an-environment)
* [CHANGELOG.md](CHANGELOG.md)

### Quick start

```
$ pip install -r requirements.txt
$ python train.py                       # 4 workers + dashboard
```

Then open http://127.0.0.1:8000.

One `python train.py` launches the parameter server, all workers and the
dashboard as separate processes.

```
$ python train.py --env cartpole
$ python train.py --num_workers 8 --envs_per_worker 8
$ python train.py --no_dashboard
$ python train.py --device cuda                       # auto | cuda | cpu | mps
$ python train.py --lr_schedule cosine --lr_warmup_steps 1000
$ python train.py --min_ready_workers 3               # straggler-tolerant
$ python train.py --max_steps 2000 --lr 3e-4 --rollout_length 512
```

Training checkpoints to `checkpoints/latest.pkl` every `--checkpoint_interval`
steps and resumes from it automatically. `--no_checkpoint` starts fresh.

### Across containers

`docker-compose.yml` runs the parameter server and each worker as separate
containers with their own network namespaces, reaching each other by DNS name
over a bridge network.

```
$ docker compose up --build     # server + 2 workers + dashboard on :8000
Worker 0 connected from ('172.18.0.3', 34402)
Worker 1 connected from ('172.18.0.4', 52204)
```

This is worth doing rather than running everything on loopback, because loopback
cannot exercise DNS resolution, per-node resource limits or network partition,
and the heartbeat monitor and `--min_ready_workers` straggler tolerance only do
anything under those conditions. Verified by killing a node mid-run:

```
$ docker compose up -d
$ docker kill drl-worker-1      # SIGKILL, no clean disconnect
```

The survivor keeps training; the heartbeat monitor evicts the dead node from the
aggregation barrier so the run does not stall.

The dashboard runs inside the server container because it reads a
`multiprocessing.Queue`, which is shared memory and cannot cross a container
boundary.

### Architecture

```
┌─────────────────────── Parameter Server Process ──────────────┐
│   TCP Server Socket :9999                                     │
│   [worker 0 thread] [worker 1 thread] ... [worker N thread]   │
│                    │  (aggregation barrier)                   │
│              ┌─────▼──────┐   ┌──────────────┐                │
│              │ WeightStore│   │ Heartbeat    │                │
│              │ + Lock     │   │ Monitor      │                │
│              └─────┬──────┘   └──────────────┘                │
│              ┌─────▼──────┐                                   │
│              │MetricsQueue│──────────────┐                    │
└────────────────────────────────────────┼─────────────────────┘
                                          │
  [Worker 0 Process] ... [Worker N Process]│
   SwingBot + PPO, TCP client ────────────┘
                                          │
  ┌──────── Dashboard (FastAPI) ──────────▼─────────┐
  │  drains MetricsQueue -> WebSocket :8000         │
  └─────────────────────────────────────────────────┘
```

Metrics flow one way, so killing the dashboard or closing the browser leaves
training untouched.

Every message is `[4-byte big-endian uint32 length][pickled payload]`. TCP is a
byte stream with no message boundaries, so both the header and the payload are
read in a loop until exactly the expected number of bytes arrive. `send_message`
and `recv_message` in `comms/protocol.py` are the only place raw socket
send/recv is called. Payload types: `PUSH_GRADIENTS`, `PULL_WEIGHTS`,
`WEIGHTS_RESPONSE`, `ACK`, `HEARTBEAT`, `HEARTBEAT_ACK`.

Per worker, the loop is: pull the latest global weights and keep a copy as the
base; collect a rollout of `rollout_length` steps across `envs_per_worker`
environments stepped together, one forward pass per timestep; compute advantages
with GAE (γ=0.99, λ=0.95) independently per environment, since the rollout
buffer is `(steps, envs, …)` and the backward recurrence must never cross an env
boundary; run `ppo_epochs` passes of the clipped surrogate plus value MSE plus
entropy bonus, taking a real Adam step per minibatch; push
`base_weights - updated_weights` as a pseudo-gradient; wait for the ACK and
repeat.

The server waits for all alive workers, averages their deltas, applies
`w = w - server_lr·(momentum·v + avg_delta)` and ACKs everyone. With the
defaults (`server_lr=1`, `momentum=0`) that is exactly FedAvg. Two learning
rates do two jobs: `--lr` is the worker's local Adam rate and shapes training,
while `--server_lr` is only the blend factor for combining worker updates.

### Environments

The actor, PPO update and rollout storage read three attributes off the env
(`observation_dim`, `action_dim`, `action_type`) and adapt, so `continuous`
gives a diagonal-Gaussian policy and `discrete` a Categorical one.

| name | obs dim | action space | task |
|---|---|---|---|
| `swingbot` | 4 | continuous (1) | pendulum swing-up and balance |
| `cartpole` | 4 | discrete (2) | balance a pole |
| `mountain_car` | 2 | continuous (1) | build momentum up a hill |
| `acrobot` | 6 | discrete (3) | two-link underactuated swing |

All four are hand-written classic-control physics, Euler or RK4, no Gym. They
exist to prove generality: both policy types and a range of state and action
dims train through the same distributed pipeline.

### Results

All four learn, at 2 workers x 4 envs over 300 steps:

| env | action space | start | end |
|---|---|---:|---:|
| `swingbot` | continuous | −136 | +144 |
| `cartpole` | discrete | 19 | 500 (task cap, solved) |
| `mountain_car` | continuous | −33 | −0.19 |
| `acrobot` | discrete | −128 | −83 |

Throughput on CPU, scaling environments (cartpole, `rollout_length=256`):

| workers | envs/worker | total envs | env-steps/s |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 5,120 |
| 1 | 4 | 4 | 11,377 |
| 2 | 4 | 8 | 18,618 |
| 4 | 4 | 16 | 37,236 |

7.3x from 1 to 16 envs, sub-linear because the aggregation barrier serializes.

### Adding an environment

Subclass `BaseEnv`, register it by name in `env/registry.py`, and run it with
`--env myenv`:

```python
from env.base import BaseEnv

class MyEnv(BaseEnv):
    observation_dim = 6
    action_type = "continuous"            # or "discrete"
    action_dim = 2                        # continuous: components; discrete: choices
    action_low, action_high = -1.0, 1.0   # continuous only

    def reset(self) -> np.ndarray:
        ...                    # float32 array of shape (observation_dim,)

    def step(self, action):
        ...                    # (next_obs, reward, done, info)
```

Nothing else changes. The policy head, PPO log-prob and entropy, and the rollout
storage are all derived from those three attributes.

### Limitations

GPU is plumbed through with `--device cuda|mps` but measured slower than CPU
here: mps 3,103 against cpu 20,480 env-steps/s. 64-unit MLPs with CPU-side env
stepping pay more in per-timestep transfers than they gain in parallelism, so
CPU is the default. Scale the networks up before reaching for a GPU. CUDA
specifically is untested.

`--min_ready_workers N` applies an update once N workers report rather than
waiting for all of them. It trades gradient staleness for straggler tolerance,
and that tradeoff is not characterized. Off by default.

Pickle plus numpy is fine on one machine. Swapping in MessagePack or flatbuffers
at the `comms/serialization.py` boundary would be the move for anything wider;
the numpy representation stays the same.

Out of scope on purpose: hyperparameter sweeps, off-policy methods, and gradient
compression.

### Tests

```
$ python -m tests.test_protocol        # 100 messages round-trip exactly
$ python -m tests.test_swingbot        # physics + reward sanity
$ python -m tests.test_serialization   # weights survive the wire
$ python -m tests.test_ppo             # PPO produces non-zero gradients
$ python -m tests.test_envs            # all envs, both policy types, one step
$ python -m tests.test_gae             # advantages stay per-env; done truncates
```

`test_gae` guards the subtlest bug this codebase has had. Flattening N envs into
one sequential buffer makes step `t` of env `i` bootstrap off env `i+1`,
corrupting every advantage while still producing plausible-looking loss curves.

### Layout

```
comms/       protocol.py (framing), serialization.py (torch <-> numpy)
network/     actor.py, critic.py, utils.py (weight/grad serialization)
env/         base.py, registry.py, swingbot.py, cartpole.py,
             mountain_car.py, acrobot.py
worker/      worker.py, rollout_buffer.py (GAE), ppo.py, client.py
server/      parameter_server.py, weight_store.py, connection_handler.py,
             heartbeat_monitor.py, metrics_emitter.py
dashboard/   server.py (FastAPI), dashboard.html (Chart.js), metrics_schema.py
tests/       protocol, swingbot, serialization, ppo, envs, gae
config.py    every hyperparameter in one dataclass
train.py     launches everything
```
