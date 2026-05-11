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


OBS_DIM = 20    # was 15: dropped 2 absolute coords + 1 front_dist, added 8 lidar
ACT_DIM = 2

_CATCH_RADIUS = jnp.float32(80.0)
_DISTANCE_OFFSET = jnp.float32(120.0)
_PURSUER_TURN_SCALE = jnp.float32(jnp.sqrt(2.0))   # sqrt(max_dw), the original value


# Rewards (caught_penalty, step_reward, speed_bonus, distance_bonus) scaled
# so per-step rewards live in O(1/T) range and episodic sums are O(few).
# distance_bonus is normalized by max_dist so per-step contribution is
# bounded — the prior 5/T·(d−offset) scaled with raw d (≤2400) and dwarfed
# both the catch penalty and any value-function target.
def _reward_consts(params: VehicleParams):
    T = params.timestep_limit
    return (jnp.float32(-10.0),
            jnp.float32(2.0 / T),
            jnp.float32(1.0 / (T * params.max_speed)),
            jnp.float32(5.0 / (T * params.max_dist)))


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
    # All quantities scaled to roughly [-1, 1] so the network sees bounded
    # inputs without depending on running-stats normalization (which can be
    # cold for the first rollout).
    s_speed = params.max_speed
    s_dist = params.max_dist
    s_acc = params.max_ds
    s_phi = jnp.float32(jnp.pi)
    s_om = params.max_dw

    dx_rel = (p[0] - e[0]) / s_dist
    dy_rel = (p[1] - e[1]) / s_dist
    dvx_rel = (p[2] - e[2]) / s_speed
    dvy_rel = (p[3] - e[3]) / s_speed
    dphi_rel = (p[6] - e[6]) / s_phi
    d = jnp.sqrt(((p[0] - e[0]) ** 2 + (p[1] - e[1]) ** 2)) / s_dist
    lidar = vehicle.lidar_distances(e[0], e[1], e[6], params) / s_dist   # (8,)
    return jnp.concatenate([
        jnp.array([e[2] / s_speed, e[3] / s_speed,
                   e[4] / s_acc, e[5] / s_acc,
                   e[6] / s_phi, e[7] / s_om], dtype=jnp.float32),         # 6
        jnp.array([dx_rel, dy_rel, dvx_rel, dvy_rel, dphi_rel, d],
                  dtype=jnp.float32),                                       # 6
        lidar.astype(jnp.float32),                                          # 8
    ])


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
