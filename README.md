# jax-hyrosphere

JAX rewrite of the HyroSphere / LinearSphere physics environments from
[../openai-physics](../openai-physics), with a single-file JAX-native PPO
training loop. The whole pipeline — physics, rollout, GAE, gradient updates —
runs as a single `jax.jit`-compiled function on GPU. Typical throughput on an
RTX 5090 is **100k+ env steps/sec** (the numpy + SubprocVecEnv reference tops
out around 2.5k sps).

## Setup

```bash
poetry install
poetry run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]  (or similar; RTX 5090 needs CUDA 12.8 wheels)
```

## Train

```bash
# 256 parallel envs × 128 rollout steps × 200 updates = 6.5M env steps
poetry run python train.py --env hyro --updates 200 --n-envs 256

# scale up to saturate the GPU
poetry run python train.py --env linear --updates 500 --n-envs 1024 --n-steps 64
```

Outputs in `runs/<env>-<timestamp>/`:

- `model.pkl` — final params + optimizer state + obs-norm stats
- `metrics.csv` — per-update return / KL / clip / entropy / value loss

## What's here

```
jax_hyrosphere/
  physics.py        Pure JAX HyroSphere / LinearSphere step functions
  env.py            Reward, observation, vmap'd batched env wrappers
train.py            Single-file PPO (actor-critic, GAE, clipped surrogate)
```

The `physics.py` module mirrors the equations of motion documented in
`../openai-physics/docs/`. Every function is a pure mapping `(state, action,
params) → state`; state is a `NamedTuple` PyTree so JAX can pytree-flatten it
for `vmap`, `jit`, and `grad`.

## Design notes

### Pure function physics

In the numpy reference each env mutates `self.position`, `self.velocity`, etc.
in `move()`. JAX requires functional updates — no in-place writes inside a
jit'd function. We pay this cost by:

1. Storing every dynamic field in a `HyroState` / `LinearState` NamedTuple.
2. Returning a *new* NamedTuple from `step` with all updated fields.
3. Using `jnp.where` for branches (e.g. friction direction at zero slip, the
   contact-impulse clamp on `v_z`).

### Vectorization via `jax.vmap`

`env.make_batched(env_kind, params, n_envs)` returns `(reset_batch, step_batch,
obs_dim, act_dim)`. The `step_batch` is `jax.vmap(step_fn)` — one call steps
all `n_envs` envs in parallel as a single GPU tensor op. There's no
SubprocVecEnv / IPC; everything lives on the device.

### PPO loop on GPU

`train.py` keeps the rollout buffer, GAE backward-scan, and all SGD epochs
inside a single `jax.jit`-compiled `train_update` function. The Python host
calls `train_update` once per PPO update cycle, but every operation inside
that call stays on the GPU. Observation normalization is a JAX-native running
mean/var (Welford parallel combination), so VecNormalize doesn't need
host-side bookkeeping.

### Approximations preserved from numpy reference

Same approximations as `../openai-physics`:

- $\vec{L}_O \approx \mathbf{J}\,\vec{\Omega}$ — wheel-spin contribution to
  angular momentum is dropped (exact for LinearSphere, approximate for
  HyroSphere when wheels spin fast).
- Quasi-static normal force `N = M·g`.
- Penetration handled by clamping the downward component of `v_O` to zero
  when the ball is in contact.
- Multiplicative angular damping `Ω ← (1 − κ) Ω` per step.

## Repo separation

This is a sibling repo to `../openai-physics`. The numpy version stays
authoritative for the interactive OpenGL viewer; this JAX version is for
high-throughput training. There's currently no shared renderer — the JAX
state can be exported to numpy via `jax.device_get(...)` if you want to feed
it back into the numpy viewer.
