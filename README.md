# jax-playground

JAX-native physics environments + PPO. Four envs, all vectorised through
`jax.vmap` and jit-compiled — env stepping, rollout, GAE, and gradient
updates all live inside a single jit'd function and run end-to-end on GPU.
Typical throughput on an RTX 5090 is 2-3 M env-steps/sec including PPO
compute, ~50 M sps for pure physics; the numpy + SubprocVecEnv reference
stack tops out around 2.5k sps.

## Envs

| name | obs / act | description |
|---|---|---|
| `hyro` | 43 / 4 | HyroSphere: 4 tetrahedral wheels inside a sphere. Reward = height + small shaping. |
| `linear` | 65 / 6 | LinearSphere: 6 cardinal sliders. Same reward as hyro. |
| `pursuit` | 20 / 2 | Evader (agent) escapes a heuristic pursuer in a 1920×1440 arena with two Г-shaped obstacle walls. Obs is self-pose (scaled) + relative pursuer pose + 8-ray body-frame lidar; absolute world coords are intentionally dropped so the policy can't memorize obstacle layout. Catch radius 80 px terminates. |
| `gathering` | 18 / 2 | Collect 2 respawning targets in the same arena as pursuit. Reward on hit, small step/spin penalty. Same lidar-based obs structure. |
| `pursuit_selfplay` | 20 / 2 (×2) | Two-agent zero-sum variant of pursuit. `env_step` takes BOTH actions; the trained role's action comes from the policy under training, the other from a frozen snapshot. Trained via `train_selfplay.py`. |
| `swarm` | 180 / 18 | 64-agent shared-policy swarm in 800×800. Restore a perturbed shape (circle/square/triangle/hex/line) using holonomic motion, 4-channel × 3-band wave comms (short/medium/long), and a 4-D direct unicast to the agent's 3 nearest neighbors. Per-agent obs: 6-sector × 3-band × 4-channel sensor, 6 nearest-neighbor vectors, velocity, rel-to-centroid, summed inbound messages, and a frozen "home" snapshot of waves + neighbors + inbound-msg degree at the un-perturbed target. |

## Setup

```bash
poetry install
poetry run python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)] — CUDA 12.8 wheels (RTX 5090 / Blackwell needs sm_120)
```

## Train

### Per-env recommended commands

```bash
# spheres — long horizon, big batch
poetry run python train.py --env hyro      --updates 2000 --n-envs 2048 --n-steps 128
poetry run python train.py --env linear    --updates 2000 --n-envs 2048

# 2D vehicles — the "safe-but-slow" PPO recipe converges reliably for these.
poetry run python train.py --env pursuit   --updates 1500 --n-envs 1024 --n-steps 128 \
    --minibatch 8192 --lr 3e-5 --clip 0.1 --target-kl 0.02 --ent-coef 0.001 \
    --hidden 256 256 --name pursuit-lidar
poetry run python train.py --env gathering --updates 1500 --n-envs 1024 --n-steps 128 \
    --minibatch 8192 --lr 3e-5 --clip 0.1 --target-kl 0.02 --ent-coef 0.001 \
    --hidden 256 256 --name gathering-lidar

# swarm — shared-policy multi-agent. Use the same "safe-but-slow" recipe as the
# vehicle envs: high default ent-coef saturates log_std at the +2 clamp and
# the policy goes random; high default lr causes mid-training KL spikes.
poetry run python train.py --env swarm     --updates 2000 --n-envs 64 --n-steps 128 \
    --minibatch 4096 --lr 1e-4 --clip 0.1 --target-kl 0.01 --ent-coef 0.001 \
    --log-std-init -0.5 --hidden 512 512 512 --name swarm-deep
```

### Training flags

All from `train.py` argparse; defaults in parentheses.

| Flag | Default | Description |
|---|---|---|
| `--env` | `hyro` | One of `hyro`, `linear`, `pursuit`, `gathering`, `swarm`. |
| `--updates` | `200` | Number of PPO update cycles. Each cycle = 1 rollout + n_epochs of SGD over it. |
| `--n-envs` | `256` | Parallel envs in the vmap'd rollout. Bigger = lower variance per update, more GPU memory. |
| `--n-steps` | `128` | Rollout length per env. `n_envs × n_steps` = batch size per update. |
| `--minibatch` | `4096` | SGD minibatch size. Must divide `n_envs × n_steps`. |
| `--n-epochs` | `4` | SGD epochs per rollout. Higher = more sample reuse, faster bias toward off-policy regime. |
| `--lr` | `3e-4` | Adam learning rate. **For 2D vehicle envs, drop to `3e-5`** — the default is unstable post-lidar. **For swarm, `1e-4`** — `3e-4` causes periodic KL spikes (>0.1) once the policy starts to specialize, undoing earlier progress. |
| `--gamma` | `0.995` | Discount factor. |
| `--gae-lambda` | `0.95` | GAE bias-variance trade-off. |
| `--clip` | `0.2` | PPO ratio clip. Use `0.1` for tighter on-policy bias when training is noisy. |
| `--ent-coef` | `0.01` | Entropy bonus. Use `0.001` for vehicle / swarm envs (`0.01` causes policy collapse — for swarm specifically, `log_std` hits the `+2` clamp on every action dim and the policy goes random). |
| `--vf-coef` | `0.5` | Value loss weight. |
| `--max-grad-norm` | `0.5` | Global-norm gradient clip. Don't touch — load-bearing for stability. |
| `--target-kl` | `0.02` | SB3-style approx-KL early-stop. When KL exceeds this within a rollout's epochs, remaining epochs are skipped. |
| `--hidden` | `256 256` | MLP layer sizes for the shared actor+critic body. |
| `--log-std-init` | `0.0` | Initial log_std (per-action-dim Gaussian). Network clamps to `[−2, 2]` for numerical safety. |
| `--seed` | `0` | RNG seed for env reset, action sampling, and minibatch permutation. |
| `--name` | timestamp | Run directory name. Output → `runs/<name>/`. |

### Per-run artifacts

Each run writes to `runs/<name>/`:

- `model.pkl` — pickle of `{args, params, obs_stats}`. Loaded by `play.py`, `play_selfplay.py`, `viewer.py`, and `train_selfplay.py --opp-model`.
- `metrics.csv` — per-update return / KL / clip / entropy / value loss for live plotting.

### Obs normalization

Pursuit and gathering scale their obs to roughly [−1, 1] **at the env layer** (every component is divided by its physical max). The PPO running-stats normalization is left at the identity (init `mean=0, var=1`, no updates) to avoid the divide-by-near-zero-variance NaN that bites dims starting from zero (`vx`, `ax`, `ω`). Don't re-enable the Welford updates in `train.py:train_update` without first scaling obs back to raw units.

### Swarm env knobs

These live in `SwarmParams` (`jax_playground/envs/swarm/physics.py`) — not `train.py` flags. They control the *task*, not the optimizer. Edit `default_params()` to change them.

| Knob | Default | Description |
|---|---|---|
| `n_agents` | `64` | Agents per env. Each step the wave kernel does an `n_agents²` einsum, so doubling this ~4× the per-step cost. Train batch dim becomes `n_envs · n_agents`. |
| `K` | `4` | Wave channels per band. `ACT_DIM = 2 + N_BANDS·K + d_msg`, `OBS_DIM = 12·N_BANDS·K + 28 + 2·d_msg`. Bumping K gives more comm bandwidth at the cost of action-space dimensionality. |
| `d_msg` | `4` | Direct-unicast payload size. Each agent sends one `d_msg`-vector per step to its 3 nearest neighbors; receivers sum all inbound messages. |
| `world_w`, `world_h` | `800` | Arena size in pixels. Positions hard-clipped to bounds. |
| `safe_box` | `700` | Centered soft-penalty box. Per-step `boundary_penalty × (fraction of agents outside)` is added to reward. |
| `lambda_short`, `lambda_med`, `lambda_long` | `60`, `150`, `400` | Wave-decay scales. `exp(-r / λ)` per band. Short = local-neighbor only; medium fills the gap; long reaches the whole arena. |
| `r_max_nbr` | `200` | Neighbor-vector gating: vectors to neighbors farther than this are zeroed in the obs. |
| `displacement` | `80` | Per-axis ±uniform scatter at reset (px). Agents start at `target + Uniform(-D, D)²`. Bigger = harder task. |
| `shape_radius_min/max` | `120` / `220` | Range of sampled shape scales (random per reset). |
| `max_accel` | `10` | Per-step accel cap (px/step²) after `tanh`-clipping the action. Lowered from 30 so a noisy random policy doesn't immediately fly into walls. |
| `max_speed` | `40` | Hard speed cap (px/step). |
| `vel_damp` | `0.85` | Per-step velocity multiplier — friction. With no accel, |v| → 33% after 5 steps, 11% after 10. |
| `step_penalty` | `-1e-3` | Per-step reward floor — encourages solving sooner. |
| `terminal_bonus` | `5.0` | Added on terminal step if `solved`. |
| `solve_threshold` | `0.05` | Chamfer / target_scale below which the episode is marked solved (~5 % of shape radius). |
| `timestep_limit` | `256` | Episode length cap. |

Reward per step (broadcast across agents in the env):
`reward = -shape_err / target_scale + boundary_penalty · out_frac + step_penalty + (terminal_bonus if solved)`

## Self-play

Train ONE role at a time against a frozen opponent. Each cycle's checkpoint plugs in as the next cycle's `--opp-model`.

```bash
# Cycle 0: bootstrap pursuer against a single-agent evader trained against the heuristic.
poetry run python train_selfplay.py --role pursuer \
    --opp-model runs/pursuit-lidar/model.pkl \
    --updates 1500 --n-envs 1024 --n-steps 128 --minibatch 8192 \
    --lr 3e-5 --clip 0.1 --target-kl 0.02 --ent-coef 0.001 \
    --hidden 256 256 --name sp-pursuer-0

# Cycle 1: train evader against the cycle-0 pursuer.
poetry run python train_selfplay.py --role evader \
    --opp-model runs/sp-pursuer-0/model.pkl \
    --updates 1500 --name sp-evader-1 …
```

`train_selfplay.py` accepts every flag `train.py` does, plus `--role {evader,pursuer}` and `--opp-model <path>` (omit for a stationary zero-action opponent — bootstrap mode for the very first cycle). Same checkpoint format, so cycles chain freely.

### Continue an existing self-play sequence

`continue_selfplay.sh` auto-detects the highest `runs/sp-{evader,pursuer}-N` and runs more cycles, alternating roles. After each cycle finishes, it launches the viewer on the fresh checkpoint for a fixed window before moving on.

```bash
./continue_selfplay.sh                        # 4 more cycles, 30s viewer each
./continue_selfplay.sh -n 8 --view-secs 60    # 8 cycles, 60s view each
./continue_selfplay.sh --no-view              # train-only
./continue_selfplay.sh --updates 3000         # longer per-cycle training
```

## Evaluate

```bash
# Single-agent envs (hyro/linear/pursuit/gathering/swarm)
poetry run python play.py                            # most recent run
poetry run python play.py --run runs/<dir>           # specific run
poetry run python play.py --episodes 10 --stochastic

# Self-play checkpoints
poetry run python play_selfplay.py --run runs/sp-pursuer-0 --episodes 10
poetry run python play_selfplay.py --run runs/sp-evader-3 --opp-model runs/sp-pursuer-2/model.pkl
```

`play.py` prints per-episode return + env diagnostic (peak z, distance, score).
`play_selfplay.py` reports return, episode length, catch rate, and mean min-distance — pulls the opponent from the checkpoint's saved args unless overridden.

## Interactive viewer

```bash
# 3D — OpenGL
poetry run python viewer.py --env hyro                          # manual keyboard
poetry run python viewer.py --run runs/<hyro-or-linear-run>     # trained policy

# 2D — pygame (pursuit / gathering)
poetry run python viewer.py --env pursuit                       # WASD = thrust + steer
poetry run python viewer.py --run runs/pursuit-lidar            # trained evader vs heuristic pursuer

# 2D — self-play (auto-detects the role from the checkpoint, loads opponent)
poetry run python viewer.py --run runs/sp-pursuer-0
poetry run python viewer.py --run runs/sp-evader-3 --opp-model runs/sp-pursuer-2/model.pkl
```

3D controls: mouse drag → orbit camera, scroll → zoom, space pause, backspace reset, `1..6` toggle overlays, `?` help.
2D controls: space pause, backspace reset, Esc/Q quit. The world (1920×1440 for pursuit) is auto-downscaled to fit on a 1080p screen.

`swarm` has its own 2D viewer (no manual control — 64 agents). Controls:
space pause, backspace reset, G toggle centroid-aligned ghost,
W toggle wave glow.

## Layout

```
jax_playground/
  _math.py             rodrigues, skew, wrap_angle, gravity
  policy.py            ActorCritic + Welford running stats (shared)
  viz3d.py             OpenGL helpers for 3D envs
  render2d.py          pygame helpers for pursuit / gathering / selfplay
  envs/
    __init__.py        REGISTRY + make_batched(env_kind, params, n_envs)
    hyrosphere/        physics.py + env.py
    linearsphere/      physics.py + env.py
    pursuit/           vehicle.py (shared) + env.py
    gathering/         env.py (uses pursuit's vehicle.py)
    pursuit_selfplay/  env.py — dual-action variant of pursuit (uses pursuit's vehicle.py)
    swarm/             physics.py (shapes+waves+chamfer) + env.py + render.py
train.py               Single-file PPO (PureJaxRL-style), single-agent envs
train_selfplay.py      Single-role PPO against a frozen opponent (pursuit_selfplay)
continue_selfplay.sh   Auto-chain self-play cycles with viewer between each
play.py                Headless eval for single-agent envs
play_selfplay.py       Headless eval for self-play checkpoints
viewer.py              Interactive viewer — 3D OpenGL (hyro/linear), 2D pygame (others)
```

## Design notes

### Pure-function physics

In each env's `physics.py` (or `env.py`), state is a NamedTuple PyTree and
`step` is a pure `(state, action, params) → state` function. No in-place
writes. Branches use `jnp.where`. All physics is jit-compilable.

### Batched envs via `jax.vmap`

`envs.make_batched(env_kind, params, n_envs)` returns
`(reset_batch_fn, step_batch_fn, obs_dim, act_dim)`. The two functions are
`jax.vmap` of the per-env `env_reset` / `env_step`, then `jax.jit`. There's
no SubprocVecEnv / IPC.

### PPO loop on GPU

`train.py` keeps the rollout buffer, GAE backward-scan, and all SGD epochs
inside a single `jax.jit`-compiled `train_update` function. Welford running
mean/var lives on-device. Target-KL early stop (`--target-kl`) skips
remaining epochs in a rollout once KL exceeds the bound, via `lax.cond`.

### Where the openai-physics / openai-game lineage shows up

`hyro` and `linear` come from `../openai-physics`; the math is documented
in that repo's `docs/`. `pursuit` and `gathering` come from
`../openai-game`. The numpy implementations are kept around as references
but are no longer required at runtime.
