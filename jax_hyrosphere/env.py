"""Batched RL env wrappers around the JAX physics.

The pattern is: pure `reset(key)` and `step(state, action, key)` functions
that map over a leading batch dim via `jax.vmap`. Auto-reset on `done` is
handled by composing `jnp.where` per state leaf, so the whole episode loop
is jit-compilable.

Reward and observation match the numpy reference env (../openai-physics):

    reward = 5·(z − r) + 0.5·v_z + 0.05·|Ω| + 0.05·|v|

Observation layout (HyroSphere, 43): velocity (3), wheel ω (4), ξ (4),
Ω (3), αΩ (3), R_i (12), dR_i (12), height (1), in_contact (1).

LinearSphere obs is 65 dims with the slider analogues.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_hyrosphere import physics
from jax_hyrosphere.physics import (
    HyroParams, HyroState, LinearParams, LinearState,
    hyro_reset, hyro_step, linear_reset, linear_step,
    default_hyro_params, default_linear_params,
    _hyro_wheel_geometry,
)


MAX_EPISODE_STEPS = 2000


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


def _reward(position, velocity, Omega, radius):
    height = position[2] - radius
    return (
        5.0 * height
        + 0.5 * velocity[2]
        + 0.05 * jnp.linalg.norm(Omega)
        + 0.05 * jnp.linalg.norm(velocity)
    )


# --------------------------------------------------------------------------
# HyroSphere env
# --------------------------------------------------------------------------


class HyroEnvState(NamedTuple):
    phys: HyroState
    step: jnp.ndarray   # scalar int32
    key: jax.Array


def _hyro_obs(state: HyroState, params: HyroParams) -> jnp.ndarray:
    R, _ = _hyro_wheel_geometry(state, params)
    # dRdt computed the same way as in step (needs Omega + omega):
    omega_total = state.omega[:, None] * state.U + state.Omega[None, :]
    dRdt = jnp.cross(omega_total, R, axis=1)
    height = state.position[2] - params.radius
    in_contact = jnp.where(state.in_contact, 1.0, 0.0)
    return jnp.concatenate([
        state.velocity,            # 3
        state.omega,               # 4
        state.ksi,                 # 4
        state.Omega,               # 3
        state.dOmegadt,            # 3
        R.reshape(-1),             # 12
        dRdt.reshape(-1),          # 12
        jnp.array([height]),       # 1
        jnp.array([in_contact]),   # 1
    ]).astype(jnp.float32)


def hyro_env_reset(key: jax.Array, params: HyroParams) -> tuple[HyroEnvState, jnp.ndarray]:
    k_init, k_next = jax.random.split(key)
    phys = hyro_reset(k_init, params)
    es = HyroEnvState(phys=phys, step=jnp.int32(0), key=k_next)
    obs = _hyro_obs(phys, params)
    return es, obs


def hyro_env_step(es: HyroEnvState, action: jnp.ndarray, params: HyroParams):
    """Auto-reset env step. Returns (next_state, obs, reward, done, info)."""
    new_phys = hyro_step(es.phys, action, params)
    new_step = es.step + 1
    reward = _reward(new_phys.position, new_phys.velocity, new_phys.Omega, params.radius)
    done = new_step >= MAX_EPISODE_STEPS

    # Auto-reset on done so the policy always gets a fresh obs.
    reset_key, next_key = jax.random.split(es.key)
    reset_phys = hyro_reset(reset_key, params)

    final_phys = jax.tree_util.tree_map(
        lambda nxt, rst: jnp.where(done, rst, nxt), new_phys, reset_phys
    )
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)
    obs = _hyro_obs(final_phys, params)
    info = {
        "peak_z": new_phys.peak_z,
        "raw_reward": reward,
    }
    return (
        HyroEnvState(phys=final_phys, step=final_step, key=next_key),
        obs,
        reward,
        done,
        info,
    )


# --------------------------------------------------------------------------
# LinearSphere env
# --------------------------------------------------------------------------


class LinearEnvState(NamedTuple):
    phys: LinearState
    step: jnp.ndarray
    key: jax.Array


def _linear_obs(state: LinearState, params: LinearParams) -> jnp.ndarray:
    R = state.shifts[:, None] * state.U
    dRdt = jnp.cross(state.Omega[None, :], R, axis=1) + state.speeds[:, None] * state.U
    height = state.position[2] - params.radius
    in_contact = jnp.where(state.in_contact, 1.0, 0.0)
    return jnp.concatenate([
        state.velocity,            # 3
        state.speeds,              # 6
        state.accelerations,       # 6
        state.shifts,              # 6
        state.Omega,               # 3
        state.dOmegadt,            # 3
        R.reshape(-1),             # 18
        dRdt.reshape(-1),          # 18
        jnp.array([height]),       # 1
        jnp.array([in_contact]),   # 1
    ]).astype(jnp.float32)


def linear_env_reset(key: jax.Array, params: LinearParams) -> tuple[LinearEnvState, jnp.ndarray]:
    k_init, k_next = jax.random.split(key)
    phys = linear_reset(k_init, params)
    es = LinearEnvState(phys=phys, step=jnp.int32(0), key=k_next)
    obs = _linear_obs(phys, params)
    return es, obs


def linear_env_step(es: LinearEnvState, action: jnp.ndarray, params: LinearParams):
    new_phys = linear_step(es.phys, action, params)
    new_step = es.step + 1
    reward = _reward(new_phys.position, new_phys.velocity, new_phys.Omega, params.radius)
    done = new_step >= MAX_EPISODE_STEPS

    reset_key, next_key = jax.random.split(es.key)
    reset_phys = linear_reset(reset_key, params)
    final_phys = jax.tree_util.tree_map(
        lambda nxt, rst: jnp.where(done, rst, nxt), new_phys, reset_phys
    )
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)
    obs = _linear_obs(final_phys, params)
    info = {"peak_z": new_phys.peak_z, "raw_reward": reward}
    return (
        LinearEnvState(phys=final_phys, step=final_step, key=next_key),
        obs,
        reward,
        done,
        info,
    )


# --------------------------------------------------------------------------
# Batched (vmap) versions — these are what the PPO training loop calls
# --------------------------------------------------------------------------


def make_batched(env_kind: str, params, n_envs: int):
    """Returns (reset_batch, step_batch, obs_dim, act_dim)."""
    if env_kind == "hyro":
        reset_fn = hyro_env_reset
        step_fn = hyro_env_step
        obs_dim = 43
        act_dim = 4
    elif env_kind == "linear":
        reset_fn = linear_env_reset
        step_fn = linear_env_step
        obs_dim = 65
        act_dim = 6
    else:
        raise ValueError(env_kind)

    @jax.jit
    def reset_batch(key: jax.Array):
        keys = jax.random.split(key, n_envs)
        return jax.vmap(lambda k: reset_fn(k, params))(keys)

    @jax.jit
    def step_batch(states, actions):
        return jax.vmap(lambda s, a: step_fn(s, a, params))(states, actions)

    return reset_batch, step_batch, obs_dim, act_dim
