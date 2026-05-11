"""gathering-v0 in pure JAX. Agent collects two respawning targets.

Reward shaping (per step):
    +target_reward    per target reached (target then respawns at random)
    +step_penalty     small negative (don't dawdle)
    +dw_penalty       penalty on |Δphi| (don't spin in place)
    +speed_bonus·|v|

Auto-reset on truncation only (no terminal condition).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground.envs.pursuit import vehicle
from jax_playground.envs.pursuit.vehicle import VehicleParams


OBS_DIM = 12
ACT_DIM = 2

_TARGET_RADIUS = jnp.float32(80.0)


def _reward_consts(params: VehicleParams):
    T = params.timestep_limit
    return (jnp.float32(3.0),
            jnp.float32(-2.0 / T),
            jnp.float32(-1.8 / T),
            jnp.float32(1.0 / T))


class EnvState(NamedTuple):
    agent: jnp.ndarray
    target_1: jnp.ndarray   # (2,)
    target_2: jnp.ndarray   # (2,)
    score: jnp.ndarray      # scalar — counts targets reached this episode
    step: jnp.ndarray
    key: jax.Array


def _random_point(key, params: VehicleParams):
    k1, k2 = jax.random.split(key)
    return jnp.array([
        jax.random.uniform(k1, (), minval=0.0, maxval=params.world_w),
        jax.random.uniform(k2, (), minval=0.0, maxval=params.world_h),
    ], dtype=jnp.float32)


def _random_place(key, params: VehicleParams):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (), minval=0.0, maxval=params.world_w)
    y = jax.random.uniform(k2, (), minval=0.0, maxval=params.world_h)
    phi = jax.random.uniform(k3, (), minval=0.0, maxval=2.0 * jnp.pi)
    return vehicle.init_state(x, y, phi)


def _obs(es: EnvState) -> jnp.ndarray:
    s = es.agent
    dx1, dy1 = s[0] - es.target_1[0], s[1] - es.target_1[1]
    dx2, dy2 = s[0] - es.target_2[0], s[1] - es.target_2[1]
    d1 = jnp.sqrt(dx1 * dx1 + dy1 * dy1)
    d2 = jnp.sqrt(dx2 * dx2 + dy2 * dy2)
    cp, sp = jnp.cos(s[6]), jnp.sin(s[6])
    t1a = vehicle.signed_angle(cp, sp, -dx1, -dy1)
    t2a = vehicle.signed_angle(cp, sp, -dx2, -dy2)
    # Permutation-invariant: nearest target first.
    swap = d1 > d2
    d1, d2 = jnp.where(swap, d2, d1), jnp.where(swap, d1, d2)
    t1a, t2a = jnp.where(swap, t2a, t1a), jnp.where(swap, t1a, t2a)
    return jnp.array([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        d1, t1a, d2, t2a,
    ], dtype=jnp.float32)


def env_reset(key: jax.Array, params: VehicleParams):
    k_agent, k_t1, k_t2, k_next = jax.random.split(key, 4)
    agent = _random_place(k_agent, params)
    t1 = _random_point(k_t1, params)
    t2 = _random_point(k_t2, params)
    es = EnvState(
        agent=agent, target_1=t1, target_2=t2,
        score=jnp.float32(0.0), step=jnp.int32(0), key=k_next,
    )
    return es, _obs(es)


def env_step(es: EnvState, action: jnp.ndarray, params: VehicleParams):
    alpha = action[0]
    beta = action[1]
    target_r, step_penalty, dw_penalty, speed_bonus = _reward_consts(params)

    old_phi = es.agent[6]
    new_agent = vehicle.step(es.agent, alpha, beta, params)

    # Target hits — respawn the hit target. Two PRNG keys for the two respawns.
    next_key, k_resp1, k_resp2 = jax.random.split(es.key, 3)
    dx1 = new_agent[0] - es.target_1[0]
    dy1 = new_agent[1] - es.target_1[1]
    hit1 = jnp.sqrt(dx1 * dx1 + dy1 * dy1) <= _TARGET_RADIUS
    new_t1 = jnp.where(hit1, _random_point(k_resp1, params), es.target_1)

    dx2 = new_agent[0] - es.target_2[0]
    dy2 = new_agent[1] - es.target_2[1]
    hit2 = jnp.sqrt(dx2 * dx2 + dy2 * dy2) <= _TARGET_RADIUS
    new_t2 = jnp.where(hit2, _random_point(k_resp2, params), es.target_2)

    n_hits = hit1.astype(jnp.float32) + hit2.astype(jnp.float32)
    new_score = es.score + n_hits
    new_step = es.step + 1

    agent_speed = vehicle.speed(new_agent)
    reward = (
        target_r * n_hits
        + jnp.abs(new_agent[6] - old_phi) * dw_penalty
        + agent_speed * speed_bonus
        + step_penalty
    )

    truncated = new_step >= params.timestep_limit
    done = truncated

    # Auto-reset on truncation.
    k_reset, next_key2 = jax.random.split(next_key)
    k_agent_r, k_t1_r, k_t2_r = jax.random.split(k_reset, 3)
    reset_agent = _random_place(k_agent_r, params)
    reset_t1 = _random_point(k_t1_r, params)
    reset_t2 = _random_point(k_t2_r, params)

    final_agent = jnp.where(done, reset_agent, new_agent)
    final_t1 = jnp.where(done, reset_t1, new_t1)
    final_t2 = jnp.where(done, reset_t2, new_t2)
    final_score = jnp.where(done, 0.0, new_score)
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)

    new_es = EnvState(
        agent=final_agent, target_1=final_t1, target_2=final_t2,
        score=final_score, step=final_step, key=next_key2,
    )
    return new_es, _obs(new_es), reward, done, {"score": new_score}
