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
| `pursuit` | 14 / 2 | Evader (agent) escapes a heuristic pursuer in a 960×720 arena. Catch radius 80 px terminates. |
| `gathering` | 12 / 2 | Collect 2 respawning targets. Reward on hit, small step/spin penalty. |

## Setup

```bash
poetry install
poetry run python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)] — CUDA 12.8 wheels (RTX 5090 / Blackwell needs sm_120)
```

## Train

```bash
poetry run python train.py --env hyro      --updates 2000 --n-envs 2048 --n-steps 128
poetry run python train.py --env linear    --updates 2000 --n-envs 2048
poetry run python train.py --env pursuit   --updates 500  --n-envs 1024
poetry run python train.py --env gathering --updates 500  --n-envs 1024

# safe-but-slow knobs:
#   --lr 3e-5 --clip 0.1 --target-kl 0.02 --ent-coef 0.001
```

Per-run artifacts under `runs/<env>-<timestamp>/`:

- `model.pkl` — params + obs-norm stats
- `metrics.csv` — per-update return / KL / clip / entropy / value loss

## Evaluate

```bash
poetry run python play.py                            # most recent run
poetry run python play.py --run runs/<dir>           # specific run
poetry run python play.py --episodes 10 --stochastic
```

`play.py` prints per-episode return and an env-specific diagnostic (peak z
for spheres, distance for pursuit, score for gathering).

## Interactive viewer (3D only)

```bash
poetry run python viewer.py --env hyro                          # manual keyboard control
poetry run python viewer.py --run runs/<dir>                    # drive with a trained policy
```

Mouse drag → orbit camera, scroll → zoom, space pause, backspace reset.
Pursuit / gathering need the 2D pygame renderer, which is not ported yet —
use `play.py` for headless eval until that's wired up.

## Layout

```
jax_playground/
  _math.py             rodrigues, skew, wrap_angle, gravity
  policy.py            ActorCritic + Welford running stats (shared)
  viz3d.py             OpenGL helpers for 3D envs
  envs/
    __init__.py        REGISTRY + make_batched(env_kind, params, n_envs)
    hyrosphere/        physics.py + env.py
    linearsphere/      physics.py + env.py
    pursuit/           vehicle.py (shared) + env.py
    gathering/         env.py (uses pursuit's vehicle.py)
train.py               Single-file PPO (PureJaxRL-style)
play.py                Headless eval
viewer.py              Interactive OpenGL (hyro/linear)
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
