"""pursuit-v0 in pure JAX. Evader (agent) escapes a heuristic pursuer.

Reward shaping (per step):
    +caught_penalty   if d ≤ catch_radius (terminates the episode)
    +step_reward      every step the evader is alive
    +speed_bonus·|v|
    +distance_bonus·(d − distance_offset)

Pursuer is hand-coded bang-bang: full thrust, angular vel proportional to
the bearing toward the evader (capped to ±1 after scaling by sqrt(max_dw)).

Auto-reset on termination (caught) or truncation (timestep_limit).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground.envs.pursuit import vehicle
from jax_playground.envs.pursuit.vehicle import VehicleParams, default_params


OBS_DIM = 15
ACT_DIM = 2

_CATCH_RADIUS = jnp.float32(80.0)
_DISTANCE_OFFSET = jnp.float32(120.0)
_PURSUER_TURN_SCALE = jnp.float32(jnp.sqrt(2.0))   # sqrt(max_dw), the original value


# Rewards (caught_penalty, step_reward, speed_bonus, distance_bonus) scaled
# 1/timestep_limit so episodic sums stay O(1).
def _reward_consts(params: VehicleParams):
    T = params.timestep_limit
    return jnp.float32(-100.0), jnp.float32(2.0 / T), jnp.float32(1.0 / T), jnp.float32(5.0 / T)


class EnvState(NamedTuple):
    pursuer: jnp.ndarray   # (8,)
    evader: jnp.ndarray    # (8,)
    step: jnp.ndarray      # scalar int32
    key: jax.Array


def _random_place(key: jax.Array, params: VehicleParams):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (), minval=0.0, maxval=params.world_w)
    y = jax.random.uniform(k2, (), minval=0.0, maxval=params.world_h)
    phi = jax.random.uniform(k3, (), minval=0.0, maxval=2.0 * jnp.pi)
    return vehicle.init_state(x, y, phi)


def _obs(es: EnvState, params: VehicleParams) -> jnp.ndarray:
    e = es.evader
    p = es.pursuer
    dx_rel = p[0] - e[0]
    dy_rel = p[1] - e[1]
    dvx_rel = p[2] - e[2]
    dvy_rel = p[3] - e[3]
    dphi_rel = p[6] - e[6]
    d = jnp.sqrt(dx_rel * dx_rel + dy_rel * dy_rel)
    front_dist = vehicle.ray_distance(e[0], e[1], e[6], params)
    return jnp.array([
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7],
        dx_rel, dy_rel, dvx_rel, dvy_rel, dphi_rel, d,
        front_dist,
    ], dtype=jnp.float32)


def env_reset(key: jax.Array, params: VehicleParams):
    k_p, k_e, k_next = jax.random.split(key, 3)
    pursuer = _random_place(k_p, params)
    evader = _random_place(k_e, params)
    es = EnvState(pursuer=pursuer, evader=evader, step=jnp.int32(0), key=k_next)
    return es, _obs(es, params)


def env_step(es: EnvState, action: jnp.ndarray, params: VehicleParams):
    alpha = action[0]
    beta = action[1]
    caught_penalty, step_reward, speed_bonus, distance_bonus = _reward_consts(params)

    # Heuristic pursuer: aim full thrust along its heading, steer by bearing.
    dx = es.pursuer[0] - es.evader[0]
    dy = es.pursuer[1] - es.evader[1]
    cp1 = jnp.cos(es.pursuer[6])
    sp1 = jnp.sin(es.pursuer[6])
    bearing = vehicle.signed_angle(cp1, sp1, -dx, -dy)
    p1_alpha = jnp.float32(1.0)
    p1_beta = jnp.clip(bearing / _PURSUER_TURN_SCALE, -1.0, 1.0)

    new_evader = vehicle.step(es.evader, alpha, beta, params)
    new_pursuer = vehicle.step(es.pursuer, p1_alpha, p1_beta, params)
    new_step = es.step + 1

    d = jnp.sqrt(
        (new_pursuer[0] - new_evader[0]) ** 2
        + (new_pursuer[1] - new_evader[1]) ** 2
    )
    caught = d <= _CATCH_RADIUS
    truncated = new_step >= params.timestep_limit
    done = caught | truncated

    evader_speed = vehicle.speed(new_evader)
    reward = (
        jnp.where(caught, caught_penalty, 0.0)
        + evader_speed * speed_bonus
        + step_reward
        + (d - _DISTANCE_OFFSET) * distance_bonus
    )

    # Auto-reset.
    reset_key, next_key = jax.random.split(es.key)
    k_p, k_e = jax.random.split(reset_key)
    reset_pursuer = _random_place(k_p, params)
    reset_evader = _random_place(k_e, params)
    final_pursuer = jnp.where(done, reset_pursuer, new_pursuer)
    final_evader = jnp.where(done, reset_evader, new_evader)
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)

    new_es = EnvState(pursuer=final_pursuer, evader=final_evader,
                       step=final_step, key=next_key)
    return new_es, _obs(new_es, params), reward, done, {"distance": d, "caught": caught}
