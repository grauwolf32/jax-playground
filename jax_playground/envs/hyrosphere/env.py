"""Batched RL env wrapper for HyroSphere."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground.envs.hyrosphere import physics
from jax_playground.envs.hyrosphere.physics import (
    HyroParams, HyroState, default_params, reset as phys_reset, step as phys_step,
    wheel_geometry,
)


MAX_EPISODE_STEPS = 2000
OBS_DIM = 43
ACT_DIM = 4


class EnvState(NamedTuple):
    phys: HyroState
    step: jnp.ndarray
    key: jax.Array


def _reward(position, velocity, Omega, radius):
    height = position[2] - radius
    return (
        5.0 * height
        + 0.5 * velocity[2]
        + 0.05 * jnp.linalg.norm(Omega)
        + 0.05 * jnp.linalg.norm(velocity)
    )


def _obs(state: HyroState, params: HyroParams) -> jnp.ndarray:
    R, _ = wheel_geometry(state, params)
    omega_total = state.omega[:, None] * state.U + state.Omega[None, :]
    dRdt = jnp.cross(omega_total, R, axis=1)
    height = state.position[2] - params.radius
    in_contact = jnp.where(state.in_contact, 1.0, 0.0)
    return jnp.concatenate([
        state.velocity, state.omega, state.ksi, state.Omega, state.dOmegadt,
        R.reshape(-1), dRdt.reshape(-1),
        jnp.array([height]), jnp.array([in_contact]),
    ]).astype(jnp.float32)


def env_reset(key: jax.Array, params: HyroParams):
    k_init, k_next = jax.random.split(key)
    p_state = phys_reset(k_init, params)
    es = EnvState(phys=p_state, step=jnp.int32(0), key=k_next)
    return es, _obs(p_state, params)


def env_step(es: EnvState, action: jnp.ndarray, params: HyroParams):
    new_phys = phys_step(es.phys, action, params)
    new_step = es.step + 1
    reward = _reward(new_phys.position, new_phys.velocity, new_phys.Omega, params.radius)
    done = new_step >= MAX_EPISODE_STEPS

    reset_key, next_key = jax.random.split(es.key)
    reset_phys = phys_reset(reset_key, params)
    final_phys = jax.tree_util.tree_map(
        lambda nxt, rst: jnp.where(done, rst, nxt), new_phys, reset_phys
    )
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)
    obs = _obs(final_phys, params)
    info = {"peak_z": new_phys.peak_z, "raw_reward": reward}
    return EnvState(phys=final_phys, step=final_step, key=next_key), obs, reward, done, info
