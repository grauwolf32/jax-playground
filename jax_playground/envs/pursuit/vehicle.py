"""Pure JAX vehicle dynamics shared by pursuit and gathering envs.

Ported from openai-game/gym_pursuit/envs/vehicle.py. Semi-implicit Euler
integration in the original (numpy) order:
    phi   += omega * dt          ; wrap to [-pi, pi]
    x,y   += vx, vy * dt
    vx,vy += ax, ay * dt
    ax,ay  = k·cos(phi) - friction_k·|v|·v
    omega  = beta · max_dw

then clip to the arena (zeroing the perpendicular velocity component on
contact) and cap |v| at max_speed.

State is an 8-vector kept compatible with the original repo:
    [x, y, vx, vy, ax, ay, phi, omega]
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground._math import wrap_angle


class VehicleParams(NamedTuple):
    max_dw: jnp.ndarray
    max_ds: jnp.ndarray
    friction_k: jnp.ndarray
    dt: jnp.ndarray
    world_w: jnp.ndarray
    world_h: jnp.ndarray
    timestep_limit: int       # static
    max_speed: jnp.ndarray
    max_dist: jnp.ndarray


def default_params(max_dw: float = 2.0,
                   max_ds: float = 6.0,
                   friction_k: float = 0.013,
                   dt: float = 0.5,
                   world_w: int = 960,
                   world_h: int = 720,
                   timestep_limit: int = 1500) -> VehicleParams:
    max_speed = float(jnp.sqrt(max_ds / friction_k))
    max_dist = float(jnp.sqrt(world_w * world_w + world_h * world_h))
    return VehicleParams(
        max_dw=jnp.float32(max_dw),
        max_ds=jnp.float32(max_ds),
        friction_k=jnp.float32(friction_k),
        dt=jnp.float32(dt),
        world_w=jnp.float32(world_w),
        world_h=jnp.float32(world_h),
        timestep_limit=int(timestep_limit),
        max_speed=jnp.float32(max_speed),
        max_dist=jnp.float32(max_dist),
    )


# 8-vector state. Kept as a flat jnp array so vmap is trivial.
def init_state(x: jnp.ndarray, y: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Vehicle state with all velocities/accels at zero."""
    return jnp.array([x, y, 0.0, 0.0, 0.0, 0.0, wrap_angle(phi), 0.0], dtype=jnp.float32)


def step(state: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray,
         params: VehicleParams) -> jnp.ndarray:
    """One vehicle step. `alpha`, `beta` will be clipped to [-1, 1] here.

    Mirrors the order of the numpy implementation exactly so trajectories
    line up at float32 precision.
    """
    alpha = jnp.clip(alpha, -1.0, 1.0)
    beta = jnp.clip(beta, -1.0, 1.0)
    dt = params.dt

    x, y, vx, vy, ax, ay, phi, omega = state
    cp, sp = jnp.cos(phi), jnp.sin(phi)

    new_phi = wrap_angle(phi + omega * dt)
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt

    speed = jnp.sqrt(new_vx * new_vx + new_vy * new_vy)
    k = alpha * params.max_ds
    new_ax = k * cp - params.friction_k * speed * new_vx
    new_ay = k * sp - params.friction_k * speed * new_vy
    new_omega = beta * params.max_dw

    # Arena wall reaction: clip position and zero the perpendicular velocity.
    over_left = new_x < 0.0
    over_right = new_x > params.world_w
    over_top = new_y < 0.0
    over_bot = new_y > params.world_h
    new_x = jnp.where(over_left, 0.0, jnp.where(over_right, params.world_w, new_x))
    new_vx = jnp.where(over_left | over_right, 0.0, new_vx)
    new_y = jnp.where(over_top, 0.0, jnp.where(over_bot, params.world_h, new_y))
    new_vy = jnp.where(over_top | over_bot, 0.0, new_vy)

    # Cap |v| at max_speed.
    speed = jnp.sqrt(new_vx * new_vx + new_vy * new_vy)
    scale = jnp.where(speed > params.max_speed, params.max_speed / jnp.maximum(speed, 1e-9), 1.0)
    new_vx = new_vx * scale
    new_vy = new_vy * scale

    return jnp.array([new_x, new_y, new_vx, new_vy, new_ax, new_ay, new_phi, new_omega],
                      dtype=jnp.float32)


# Accessor helpers — useful when reading state outside JIT.
def x(s):      return s[..., 0]
def y(s):      return s[..., 1]
def vx(s):     return s[..., 2]
def vy(s):     return s[..., 3]
def phi(s):    return s[..., 6]
def omega(s):  return s[..., 7]
def speed(s):  return jnp.sqrt(s[..., 2] ** 2 + s[..., 3] ** 2)


def signed_angle(cp, sp, dx, dy):
    """Signed angle from (cos(phi), sin(phi)) heading to (dx, dy) vector.

    Returns a value in (-pi, pi] matching the numpy helper.
    """
    cross = cp * dy - sp * dx
    dot = cp * dx + sp * dy
    return jnp.arctan2(cross, dot)
