# What changed, and why

Every number below was measured on this machine (Apple Silicon, CPU) with the
command shown. Nothing here is estimated.

## Bugs fixed

### 1. GAE was computed across interleaved environments (silent, fatal)

Vectorized rollouts wrote transitions into a flat buffer:

```
index:   0        1        2        3
         env0_t0  env1_t0  env0_t1  env1_t1
```

`compute_gae` walks backward treating index `t` and `t+1` as consecutive
timesteps of one trajectory, so **env0's transition bootstrapped off env1's
value estimate**. Every advantage was corrupted. Training produced
plausible-looking loss curves and learned nothing.

Fix: `RolloutBuffer` is now `(num_steps, num_envs, ...)` — time-major,
env-minor — and GAE runs as a vectorized backward recurrence along the time
axis with one running advantage per env. Flattening happens only in
`get_minibatches()`, after GAE, where sample order no longer matters.

Guarded by `tests/test_gae.py`, which checks each env's advantages against an
independently written single-trajectory reference, that changing one env's
rewards cannot move another env's advantages, and that `done` truncates
bootstrapping.

### 2. The clipped surrogate objective was inert — this was not PPO

Workers computed gradients but never applied an optimizer step, so the policy
did not move between PPO epochs. That makes
`ratio = exp(new_logprob - old_logprob)` identically 1, so:

- the clip never engaged — it was A2C wearing a PPO costume,
- `actor_loss` was always exactly ±0.0000 (`-mean(advantages)` on
  zero-mean normalized advantages),
- `ppo_epochs=4` recomputed the identical gradient four times.

Fix: each worker now runs true local PPO with its own Adam optimizer, then
ships `old_weights - new_weights` as a pseudo-gradient. The server averages
these and applies `w <- w - server_lr * (momentum*v + avg_delta)`. With
`server_lr=1, momentum=0` this is exactly FedAvg: the new global weights are
the mean of the workers' locally-updated weights.

This is why `lr` (worker-local, the one that matters) and `server_lr` (blend
factor) are now separate settings, and why the LR schedule moved to the worker.

### 3. Checkpointing could never fire

The save was inside the `listener.accept()` loop, so it only ran when a *new
worker connected*. Once all workers were attached the server blocked in
`accept()` forever and no checkpoint was ever written. Moved onto the update
path in `GradientAggregator`.

### 4. Child processes logged into the void

`logging.basicConfig` was called only in the parent's `main()`. Under the
`spawn` start method children get a fresh interpreter and never inherit that
config, so every `logger.info` in the server and workers was dropped. This is
why failures looked like "the process just exited". Added `logging_setup.py`,
called at the top of each process entry point.

### 5. Duplicate `_connect_with_retry`

Two definitions of the same function; the second silently shadowed the first.
The dead one also contained unreachable code referencing undefined names.
Also broadened the caught exception from `ConnectionRefusedError` to `OSError`
and replaced exponential backoff (which could stall for minutes) with a flat
0.5 s retry.

### 6. `squeeze(-1)` on discrete actions

`Categorical.sample()` already returns shape `(num_envs,)`. The extra
`squeeze(-1)` was a no-op for `num_envs > 1` but collapsed the tensor to a
scalar when `envs_per_worker=1`. Removed.

### 7. Missing `--checkpoint_interval` CLI flag

Present in the config, never wired into the argument parser.

### 8. Health probes were registered as live workers

Found only under Docker. The compose healthcheck opens a TCP connection to
:9999 every 2 seconds. Because the server assigned worker ids by connection
order (a shortcut the original code's own comment flagged), **every probe
registered as a brand-new live worker** — 177 of them in a single 400-step run.

This is not cosmetic. A registered worker joins the aggregation barrier, so
`waiting_on = alive ∩ sockets − have` could include a health probe and stall
training until it disconnected.

Fix: a connection is not registered until it sends a message carrying its own
`worker_id`. Anything that merely opens a socket and hangs up — health probe,
port scan — is never counted. Workers now also register under their real id
instead of an arbitrary connection-order slot. Guarded by
`tests/test_registration.py`.

### 9. `device` defaulted to `"cuda"`

A plain `python train.py` crashed with `Torch not compiled with CUDA enabled`
on any machine without an NVIDIA GPU. Every earlier test had passed
`--device cpu` explicitly, which masked it. Now defaults to `cpu`, with the
benchmark below justifying that as the right choice rather than a fallback.

## Verified results

All four environments, both action-space types, learn:

```
python3 train.py --env <env> --num_workers 2 --envs_per_worker 4 \
                 --rollout_length 256 --max_steps 300 --no_dashboard
```

| env | action space | return @ step 0 | return @ step 290 |
|---|---|---:|---:|
| swingbot | continuous | −136 | **+144** |
| cartpole | discrete | 19 | **500** (task cap — solved) |
| mountain_car | continuous | −33 | **−0.19** |
| acrobot | discrete | −128 | **−83** |

### Throughput scaling

```
python3 train.py --env cartpole --rollout_length 256 --max_steps 100 --no_dashboard
```

| workers | envs/worker | total envs | wall clock | env-steps/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 5 s | 5,120 |
| 1 | 4 | 4 | 9 s | 11,377 |
| 2 | 4 | 8 | 11 s | 18,618 |
| 4 | 4 | 16 | 11 s | 37,236 |

7.3× throughput from 1 → 16 environments. Sub-linear because the parameter
server serializes on the aggregation barrier.

### GPU is slower here, and that is the expected result

```
python3 train.py --env cartpole --device <dev> --num_workers 2 \
                 --envs_per_worker 4 --rollout_length 256 --max_steps 100
```

| device | wall clock | env-steps/s |
|---|---:|---:|
| cpu | 10 s | **20,480** |
| mps | 66 s | 3,103 |

**MPS is 6.6× slower than CPU.** The networks are 64-unit MLPs and the
environments step on the CPU, so every timestep pays a host↔device transfer and
a kernel launch that dwarf the arithmetic being parallelized. This is why
`device` now defaults to `cpu` rather than auto-selecting a GPU — an earlier
default of `cuda` also meant a plain `python train.py` crashed outright on any
machine without CUDA.

`--device cuda|mps` remains available for when the networks are scaled up
enough to invert this. **CUDA specifically is untested** — no NVIDIA hardware
was available here.

### Checkpoint resume

```
run 1 (from scratch):  step 0 return=25.7  ...  step 50 return=121.5  → saved
run 2 (resumed):       Loaded checkpoint (step 50)
                       step 0 return=90.7  ...  step 50 return=248.9
```

Run 2 starts where run 1 left off rather than at ~25, confirming the learned
policy survives the round trip.

### Container topology (entry points verified over TCP)

`run_server.py` and `run_worker.py` — the exact commands the containers run —
were launched as three independent OS processes communicating only over a TCP
socket, which is the same code path containers take:

```
server:    Listening on 127.0.0.1:9977
           Worker 0 connected from ('127.0.0.1', 50721)
           Worker 1 connected from ('127.0.0.1', 50720)
           Saved checkpoint  ×3
worker 0:  step   0/150 return= 18.75
           step  50/150 return=157.19
           step 100/150 return=454.34
           step 140/150 return=461.96
           Worker 0 finished (150 steps)
```

150 synchronized global steps across two independently-launched workers in
~12 s, then a clean shutdown.

### Containers (the real thing)

`docker compose up` — server and each worker in its own container, own network
namespace, reaching each other by DNS name over a bridge network:

```
Worker 0 connected from ('172.18.0.3', 34402)
Worker 1 connected from ('172.18.0.4', 52204)
worker 0 step   0/400 return=-136.25
worker 0 step 200/400 return= 137.09
worker 0 step 390/400 return= 151.02
Saved checkpoint  ×8
Worker 0 finished (400 steps)   Worker 1 finished (400 steps)
```

Distinct container IPs, resolved through the service name `parameter-server` —
this exercises DNS resolution and cross-namespace routing that loopback cannot.

### Fault tolerance under node loss

The reason containers matter and loopback is not enough: on loopback every
worker runs at identical speed and nothing can be partitioned, so the heartbeat
monitor and `--min_ready_workers` were never actually exercised.

`docker kill drl-worker-1` mid-run (SIGKILL — no clean disconnect):

```
>>> KILLING worker-1 at ~step 20 <<<
worker 0 step 290/400 return=136.84   ← survivor keeps going
worker 0 step 340/400 return=145.94
server: Saved checkpoint  (still running)
drl-worker-0: Up   drl-server: Up (healthy)
```

The heartbeat monitor evicted the dead node from the aggregation barrier and
training continued without stalling. This is the code path that had never once
run before containers.

### Test suite

```
test_protocol      PASS
test_swingbot      PASS
test_serialization PASS
test_ppo           PASS
test_envs          PASS   (rewritten to exercise the vectorized path)
test_gae           PASS   (new — regression guard for bug #1)
```

## Features added

- **Vectorized environments** (`env/vectorized.py`) — N envs per worker stepped
  together, one batched forward pass.
- **Checkpoint / resume** — periodic save of weights, momentum, and step
  counter; resumes automatically if `checkpoints/latest.pkl` exists.
- **LR schedules** — constant / linear / cosine, with optional warmup, applied
  to the worker's local optimizer.
- **Straggler-tolerant aggregation** — `--min_ready_workers N` applies an
  update once N workers report instead of waiting for all.
- **Loss observability** — actor loss, critic loss, and entropy tracked per
  worker, sent on the heartbeat, and charted on the dashboard.
- **Config validation** — `TrainConfig.__post_init__` rejects invalid
  hyperparameters at startup.
- **Structured logging** — replaces `print`, configured per process.
- **Docker** — `docker-compose.yml` runs the server and each worker as separate
  containers on a bridge network, communicating only over TCP.

## Known limitations

- **CUDA untested** — no NVIDIA hardware available. MPS was tested and is
  slower than CPU for these network sizes (see benchmark above).
- **`min_ready_workers` introduces gradient staleness.** Off by default; the
  convergence tradeoff is not characterized. Node *loss* is verified (above),
  but a deliberately *slow* node (via container CPU limits) is not.
- **The dashboard runs inside the server container**, not its own, because it
  reads a `multiprocessing.Queue` (shared memory) which cannot cross a
  container boundary. Making it a standalone container requires exposing
  metrics over TCP/HTTP instead.
- **Network partition is untested.** Node kill is covered; `docker network
  disconnect` (reachable-but-silent) is a distinct failure mode and is not.

## Environment note: what actually broke Docker

Worth recording, because the first diagnosis was incomplete.

Docker Desktop was repeatedly killing its own daemon. The host volume was
**94% full (27.9 GB free of 494 GB)** — traced to
`~/.cache/cacheflow-bench-repos`, **163 GB** of orphaned benchmark snapshots
(247 `.bin` files, ~1 GB each, from Jun 28–30). Not the git clones:
`requests/.git` is 3.5 MB while `requests/.cacheflow/snapshots` was 59 GB.

Deleting those reclaimed 162 GB (94% → 53% full) — **and Docker still failed.**
Freeing space was necessary but not sufficient. The host log showed why:

```
panic: assertion failed: Page expected to be: 18, but self identifies as 0
```

A bbolt page-corruption panic in containerd's `meta.db`. The full-disk
condition had corrupted that database mid-write, and the corruption persisted
on disk after space was freed, so the engine panicked ~1 s into every startup.
Because `meta.db` lives inside the VM image, it could not be repaired from the
host — `Docker.raw` had to be deleted and the VM recreated. Docker came up in
10 s after that.

Sequence: disk exhaustion → DB corruption → daemon crash loop that outlived
its own cause.
