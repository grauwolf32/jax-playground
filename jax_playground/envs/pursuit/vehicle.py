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
    obstacles: jnp.ndarray    # (N, 4) AABBs as (x0, y0, x1, y1); N static


def _default_obstacles(world_w: float, world_h: float) -> jnp.ndarray:
    """Two mirrored Г shapes near the top of the arena.

    Geometry (for the default 1920×1440 world, scales with world size):
    Left Г       Right Г (mirror)
       ████      ████        horizontal bars across the top
       █            █        verticals going down on the outside
       █            █
       █            █
    Each Г is built from two overlapping AABBs.
    """
    w, h = float(world_w), float(world_h)
    # bars are 1/24th of world height (~60 px on 1440 → reasonable thickness),
    # placed 1/6 of the way from top, length 1/4 of world width on each side.
    bar_t = h / 24.0
    margin = w / 8.0
    bar_len = w / 4.0
    top_y = h / 6.0
    bot_y = h * 5.0 / 6.0
    # Left Г.
    Lh = (margin, top_y, margin + bar_len, top_y + bar_t)             # horizontal bar
    Lv = (margin, top_y, margin + bar_t, bot_y)                       # vertical bar
    # Right Г (mirrored).
    Rh = (w - margin - bar_len, top_y, w - margin, top_y + bar_t)
    Rv = (w - margin - bar_t, top_y, w - margin, bot_y)
    return jnp.array([Lh, Lv, Rh, Rv], dtype=jnp.float32)


def default_params(max_dw: float = 2.0,
                   max_ds: float = 6.0,
                   friction_k: float = 0.013,
                   dt: float = 0.5,
                   world_w: int = 1920,
                   world_h: int = 1440,
                   timestep_limit: int = 1500,
                   obstacles: jnp.ndarray | None = None) -> VehicleParams:
    if obstacles is None:
        obstacles = _default_obstacles(world_w, world_h)
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
        obstacles=obstacles.astype(jnp.float32),
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

    # Static obstacles: AABBs. If the vehicle ends up inside one, push it
    # back to the nearest face and zero the inward velocity component. The
    # Python loop unrolls statically (obstacles.shape[0] is a static int).
    for i in range(params.obstacles.shape[0]):
        x0, y0, x1, y1 = (params.obstacles[i, 0], params.obstacles[i, 1],
                          params.obstacles[i, 2], params.obstacles[i, 3])
        inside = (new_x > x0) & (new_x < x1) & (new_y > y0) & (new_y < y1)
        dx_left = new_x - x0
        dx_right = x1 - new_x
        dy_top = new_y - y0
        dy_bot = y1 - new_y
        min_d = jnp.minimum(jnp.minimum(dx_left, dx_right),
                             jnp.minimum(dy_top, dy_bot))
        push_left = inside & (min_d == dx_left)
        push_right = inside & (min_d == dx_right)
        push_top = inside & (min_d == dy_top)
        push_bot = inside & (min_d == dy_bot)
        new_x = jnp.where(push_left, x0,
                           jnp.where(push_right, x1, new_x))
        new_y = jnp.where(push_top, y0,
                           jnp.where(push_bot, y1, new_y))
        new_vx = jnp.where(push_left | push_right, 0.0, new_vx)
        new_vy = jnp.where(push_top | push_bot, 0.0, new_vy)

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


def _ray_aabb_t(px, py, dx, dy, x0, y0, x1, y1):
    """Parametric distance along ray (px,py)+t·(dx,dy) to AABB.

    Returns +inf if no intersection ahead (t ≤ 0 cases yield +inf too).
    Direction (dx, dy) need not be unit length; t is in units of |d|⁻¹·meters.
    For unit direction this is the Euclidean distance.
    """
    eps = 1e-6
    safe_dx = jnp.where(jnp.abs(dx) < eps, eps, dx)
    safe_dy = jnp.where(jnp.abs(dy) < eps, eps, dy)
    tx1 = (x0 - px) / safe_dx
    tx2 = (x1 - px) / safe_dx
    ty1 = (y0 - py) / safe_dy
    ty2 = (y1 - py) / safe_dy
    tmin_x = jnp.minimum(tx1, tx2)
    tmax_x = jnp.maximum(tx1, tx2)
    tmin_y = jnp.minimum(ty1, ty2)
    tmax_y = jnp.maximum(ty1, ty2)
    t_enter = jnp.maximum(tmin_x, tmin_y)
    t_exit = jnp.minimum(tmax_x, tmax_y)
    hit = (t_enter <= t_exit) & (t_exit > 0)
    return jnp.where(hit, jnp.maximum(t_enter, 0.0), jnp.inf)


def ray_distance(px, py, phi, params: VehicleParams) -> jnp.ndarray:
    """Distance from (px, py) along heading (cos phi, sin phi) to the
    nearest blocker — any obstacle face or arena wall ahead. The agent's
    own body is treated as a point, so a vehicle near a wall sees the wall
    as the closest hit and gets distance ~0.
    """
    dx = jnp.cos(phi)
    dy = jnp.sin(phi)
    eps = 1e-6
    safe_dx = jnp.where(jnp.abs(dx) < eps, eps, dx)
    safe_dy = jnp.where(jnp.abs(dy) < eps, eps, dy)

    # Arena walls (vehicle is inside the box). For each face, the ray hits
    # at t = (face_coord - p) / d when that t is positive.
    t_left = (0.0 - px) / safe_dx
    t_right = (params.world_w - px) / safe_dx
    t_top = (0.0 - py) / safe_dy
    t_bot = (params.world_h - py) / safe_dy
    arena_candidates = jnp.stack([t_left, t_right, t_top, t_bot])
    arena_t = jnp.min(jnp.where(arena_candidates > 0, arena_candidates, jnp.inf))

    # Obstacles
    obs_t = jnp.inf
    for i in range(params.obstacles.shape[0]):
        x0, y0, x1, y1 = (params.obstacles[i, 0], params.obstacles[i, 1],
                          params.obstacles[i, 2], params.obstacles[i, 3])
        obs_t = jnp.minimum(obs_t, _ray_aabb_t(px, py, dx, dy, x0, y0, x1, y1))

    return jnp.minimum(arena_t, obs_t)
