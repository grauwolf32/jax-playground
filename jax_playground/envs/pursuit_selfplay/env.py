"""pursuit_selfplay-v0 in pure JAX. Both vehicles are policy-driven.

Symmetric two-agent env:
    - Same physics as pursuit (uses pursuit.vehicle).
    - Each agent sees a 20-dim POV obs (same shape as pursuit-v0): self
      pose (scaled) + relative opponent pose + lidar.
    - env_step takes BOTH actions (pursuer + evader). The caller is
      responsible for computing them — typically the trained policy for one
      role and a frozen snapshot for the other.
    - Reward returned is zero-sum: the evader's pursuit reward. The training
      loop negates it for the pursuer's update.

Auto-reset on catch or timeout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground.envs.pursuit import vehicle
from jax_playground.envs.pursuit.vehicle import VehicleParams, default_params  # noqa: F401


OBS_DIM = 20    # same as pursuit
ACT_DIM = 2

_CATCH_RADIUS = jnp.float32(80.0)
_DISTANCE_OFFSET = jnp.float32(120.0)


def _reward_consts(params: VehicleParams):
    """Same scaling as pursuit-v0 so the two envs reward-shape identically."""
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


def _pov_obs(self_v: jnp.ndarray, opp_v: jnp.ndarray,
             params: VehicleParams) -> jnp.ndarray:
    """20-dim POV obs from `self_v`'s perspective (other agent is opp_v).

    Same structure and scaling as pursuit-v0's evader obs, but parameterized
    by viewpoint so both roles can use the SAME network architecture.
    """
    s_speed = params.max_speed
    s_dist = params.max_dist
    s_acc = params.max_ds
    s_phi = jnp.float32(jnp.pi)
    s_om = params.max_dw

    dx_rel = (opp_v[0] - self_v[0]) / s_dist
    dy_rel = (opp_v[1] - self_v[1]) / s_dist
    dvx_rel = (opp_v[2] - self_v[2]) / s_speed
    dvy_rel = (opp_v[3] - self_v[3]) / s_speed
    dphi_rel = (opp_v[6] - self_v[6]) / s_phi
    d = jnp.sqrt((opp_v[0] - self_v[0]) ** 2
                  + (opp_v[1] - self_v[1]) ** 2) / s_dist
    lidar = vehicle.lidar_distances(self_v[0], self_v[1], self_v[6], params) / s_dist
    return jnp.concatenate([
        jnp.array([self_v[2] / s_speed, self_v[3] / s_speed,
                   self_v[4] / s_acc, self_v[5] / s_acc,
                   self_v[6] / s_phi, self_v[7] / s_om], dtype=jnp.float32),
        jnp.array([dx_rel, dy_rel, dvx_rel, dvy_rel, dphi_rel, d],
                  dtype=jnp.float32),
        lidar.astype(jnp.float32),
    ])


def env_reset(key: jax.Array, params: VehicleParams):
    """Place both vehicles. Returns (state, evader_obs, pursuer_obs)."""
    k_p, k_e, k_next = jax.random.split(key, 3)
    pursuer = _random_place(k_p, params)
    evader = _random_place(k_e, params)
    es = EnvState(pursuer=pursuer, evader=evader, step=jnp.int32(0), key=k_next)
    return es, _pov_obs(evader, pursuer, params), _pov_obs(pursuer, evader, params)


def env_step(es: EnvState,
             evader_action: jnp.ndarray, pursuer_action: jnp.ndarray,
             params: VehicleParams):
    """Step both vehicles. Returns:
        (state, evader_obs, pursuer_obs, evader_reward, done, info)

    The pursuer's reward is `-evader_reward` (zero-sum); the training loop
    applies the sign.
    """
    caught_penalty, step_reward, speed_bonus, distance_bonus = _reward_consts(params)

    new_evader = vehicle.step(es.evader, evader_action[0], evader_action[1], params)
    new_pursuer = vehicle.step(es.pursuer, pursuer_action[0], pursuer_action[1], params)
    new_step = es.step + 1

    d = jnp.sqrt((new_pursuer[0] - new_evader[0]) ** 2
                  + (new_pursuer[1] - new_evader[1]) ** 2)
    caught = d <= _CATCH_RADIUS
    truncated = new_step >= params.timestep_limit
    done = caught | truncated

    evader_speed = vehicle.speed(new_evader)
    evader_reward = (
        jnp.where(caught, caught_penalty, 0.0)
        + evader_speed * speed_bonus
        + step_reward
        + (d - _DISTANCE_OFFSET) * distance_bonus
    )

    # Auto-reset on done.
    reset_key, next_key = jax.random.split(es.key)
    k_p, k_e = jax.random.split(reset_key)
    reset_pursuer = _random_place(k_p, params)
    reset_evader = _random_place(k_e, params)
    final_pursuer = jnp.where(done, reset_pursuer, new_pursuer)
    final_evader = jnp.where(done, reset_evader, new_evader)
    final_step = jnp.where(done, 0, new_step).astype(jnp.int32)

    new_es = EnvState(pursuer=final_pursuer, evader=final_evader,
                       step=final_step, key=next_key)
    return (new_es,
            _pov_obs(final_evader, final_pursuer, params),
            _pov_obs(final_pursuer, final_evader, params),
            evader_reward, done, {"distance": d, "caught": caught})
