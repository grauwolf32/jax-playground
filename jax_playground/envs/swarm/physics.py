"""swarm-v0 physics: 2D holonomic agents with wave-emission comms.

Each env contains `n_agents` agents sharing policy weights. They emit on
2 bands (short / long range), K channels per band. The per-agent sensor
is a hex of 6 sectors; for each sector × band × channel we accumulate
    contrib = w_jbk · exp(-r_ij / λ_b) · max(0, cos(θ_ij - sector_angle))
over all emitters j ≠ i. The result is a (n, 6, 2, K) tensor.

Goal is to restore a sampled target shape after a random per-agent
displacement; reward is the symmetric Chamfer distance to the target
after aligning swarm centroid to target centroid.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


N_SECTORS = 6
N_BANDS = 3            # short / medium / long wave bands
N_NBR = 6              # neighbor-vector channel
N_DIRECT_NBR = 3       # direct unicast: each agent sends its message to its 3 nearest
N_SHAPES = 5

# Sector centers (6 hex sectors, axis-aligned at 0/60/120/180/240/300°).
SECTOR_ANGLES = jnp.linspace(0.0, 2.0 * jnp.pi, N_SECTORS + 1)[:-1]


class SwarmParams(NamedTuple):
    n_agents: int = 64
    K: int = 4
    d_msg: int = 4                   # direct-message payload dim (per agent, per step)
    world_w: float = 800.0
    world_h: float = 800.0
    safe_box: float = 700.0          # outside this centered box → soft penalty
    lambda_short: float = 60.0
    lambda_med: float = 150.0        # intermediate band — fills the gap between short/long
    lambda_long: float = 400.0
    r_max_nbr: float = 200.0
    displacement: float = 80.0       # max ±per-axis perturbation at reset
    shape_radius_min: float = 120.0
    shape_radius_max: float = 220.0
    dt: float = 1.0
    max_accel: float = 10.0           # px/step² — small per-step nudge so action noise is mild
    max_speed: float = 40.0
    vel_damp: jnp.float32 = jnp.float32(0.85)   # friction: ~33% of v survives 5 steps, ~11% after 10
    step_penalty: float = -1e-3
    boundary_penalty: float = -2e-3   # per-step, multiplied by fraction of agents outside safe_box
    terminal_bonus: float = 5.0
    solve_threshold: float = 0.05     # Chamfer / target_scale; below this → solved (~5% of shape radius)
    timestep_limit: int = 256


def default_params() -> SwarmParams:
    return SwarmParams()


# --------------------------------------------------------------------------
# Shape templates — all return (n, 2) unit-scale points centered at origin.
# --------------------------------------------------------------------------


def _circle_points(n: int) -> jnp.ndarray:
    ts = jnp.linspace(0.0, 1.0, n + 1)[:-1]
    a = 2.0 * jnp.pi * ts
    return jnp.stack([jnp.cos(a), jnp.sin(a)], axis=-1)


def _polygon_points(n: int, sides: int) -> jnp.ndarray:
    """n points evenly along the perimeter of a regular polygon inscribed
    in unit circle."""
    vs = jnp.linspace(0.0, 2.0 * jnp.pi, sides + 1)[:-1]
    verts = jnp.stack([jnp.cos(vs), jnp.sin(vs)], axis=-1)
    ts = jnp.linspace(0.0, float(sides), n + 1)[:-1]
    k = jnp.floor(ts).astype(jnp.int32) % sides
    f = (ts - jnp.floor(ts))[:, None]
    return verts[k] + f * (verts[(k + 1) % sides] - verts[k])


def _line_points(n: int) -> jnp.ndarray:
    xs = jnp.linspace(-1.0, 1.0, n)
    return jnp.stack([xs, jnp.zeros_like(xs)], axis=-1)


def shape_template(shape_id: jnp.ndarray, n: int) -> jnp.ndarray:
    """Pick one of N_SHAPES templates by id (0..4). Returns (n, 2)."""
    return jax.lax.switch(
        shape_id,
        [lambda: _circle_points(n),
         lambda: _polygon_points(n, 4),
         lambda: _polygon_points(n, 3),
         lambda: _polygon_points(n, 6),
         lambda: _line_points(n)],
    )


def sample_target(key: jax.Array, p: SwarmParams):
    """Sample (target positions (n_agents, 2), shape_id) at random rotation/scale."""
    k_id, k_rot, k_scale = jax.random.split(key, 3)
    shape_id = jax.random.randint(k_id, (), 0, N_SHAPES)
    pts = shape_template(shape_id, p.n_agents)
    rot = jax.random.uniform(k_rot, (), minval=0.0, maxval=2.0 * jnp.pi)
    scale = jax.random.uniform(k_scale, (),
                                minval=p.shape_radius_min,
                                maxval=p.shape_radius_max)
    c, s = jnp.cos(rot), jnp.sin(rot)
    R = jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])])
    pts = (pts * scale) @ R.T
    center = jnp.array([p.world_w / 2.0, p.world_h / 2.0], dtype=jnp.float32)
    return (pts + center).astype(jnp.float32), shape_id


# --------------------------------------------------------------------------
# Wave sensor & neighbor vectors
# --------------------------------------------------------------------------


def wave_sensor(pos: jnp.ndarray, emissions: jnp.ndarray,
                p: SwarmParams) -> jnp.ndarray:
    """Compute (n, N_SECTORS, N_BANDS, K) sensor readings.

    Args:
      pos:       (n, 2)
      emissions: (n, N_BANDS, K) — agent j's emission on band b, channel k.
    """
    n = pos.shape[0]
    # Pairwise displacement i → j: dx[i, j] = pos[j] - pos[i]
    dx = pos[None, :, :] - pos[:, None, :]                # (n, n, 2)
    r = jnp.linalg.norm(dx, axis=-1)                      # (n, n)
    eye = jnp.eye(n, dtype=bool)
    r_safe = jnp.where(eye, jnp.inf, r)                   # block self-contrib
    theta = jnp.arctan2(dx[..., 1], dx[..., 0])           # (n, n)

    # Sector cosine masks: cos_st[s, i, j] = max(0, cos(θ_ij - φ_s))
    cos_st = jnp.maximum(
        0.0,
        jnp.cos(theta[None, :, :] - SECTOR_ANGLES[:, None, None]),
    )                                                     # (S, n, n)
    # Per-band exponential decay (must be N_BANDS long)
    lambdas = jnp.array([p.lambda_short, p.lambda_med, p.lambda_long],
                         dtype=jnp.float32)
    decay = jnp.exp(-r_safe[None, :, :] / lambdas[:, None, None])  # (B, n, n)
    combined = cos_st[:, None, :, :] * decay[None, :, :, :]        # (S, B, n, n)
    # sensor[i, s, b, k] = Σ_j combined[s, b, i, j] · emissions[j, b, k]
    return jnp.einsum("sbij,jbk->isbk", combined, emissions)


def neighbor_vectors(pos: jnp.ndarray, p: SwarmParams) -> jnp.ndarray:
    """Return (n, N_NBR, 2): vectors to the N_NBR nearest agents, zeroed
    beyond r_max_nbr. Self excluded; ordered nearest-first.
    """
    n = pos.shape[0]
    dx = pos[None, :, :] - pos[:, None, :]
    r = jnp.linalg.norm(dx, axis=-1)
    eye = jnp.eye(n, dtype=bool)
    r_safe = jnp.where(eye, jnp.inf, r)
    _, idx = jax.lax.top_k(-r_safe, N_NBR)
    nbr_vecs = jnp.take_along_axis(dx, idx[..., None], axis=1)
    nbr_r = jnp.take_along_axis(r_safe, idx, axis=1)
    return jnp.where((nbr_r <= p.r_max_nbr)[..., None], nbr_vecs, 0.0)


# --------------------------------------------------------------------------
# Integration + emissions parsing
# --------------------------------------------------------------------------


def parse_action(action: jnp.ndarray, p: SwarmParams):
    """Slice the (n, ACT_DIM) action into its three sub-fields.

    Layout: [ax, ay, w_band0_chan0..K, w_band1_chan0..K, ..., msg_0..d_msg]
    Returns:
      accel:     (n, 2)   raw, NOT scaled — integrate() applies max_accel.
      emissions: (n, N_BANDS, K)   wave amplitudes per band per channel, [-1, 1].
      msgs:      (n, d_msg)        unicast payload, [-1, 1].
    """
    n = action.shape[0]
    K = p.K
    d_msg = p.d_msg
    n_wave = N_BANDS * K
    accel = action[:, :2]
    waves = jnp.clip(action[:, 2:2 + n_wave], -1.0, 1.0)
    msgs = jnp.clip(action[:, 2 + n_wave:2 + n_wave + d_msg], -1.0, 1.0)
    return accel, waves.reshape(n, N_BANDS, K), msgs


def emissions_from_action(action: jnp.ndarray, p: SwarmParams) -> jnp.ndarray:
    """Back-compat helper — returns just the wave-emission slice."""
    _, em, _ = parse_action(action, p)
    return em


def direct_message_pass(pos: jnp.ndarray, msgs: jnp.ndarray) -> jnp.ndarray:
    """Each agent unicasts `msgs[j]` to its N_DIRECT_NBR nearest neighbors.
    Receiver i collects (sums) every message addressed to it.

    pos:  (n, 2)        positions
    msgs: (n, d_msg)    sender j's outgoing message
    returns received: (n, d_msg) — Σ_{j: i ∈ topK(j)} msgs[j]
    """
    n = pos.shape[0]
    dx = pos[None, :, :] - pos[:, None, :]
    r = jnp.linalg.norm(dx, axis=-1)
    eye = jnp.eye(n, dtype=bool)
    r_safe = jnp.where(eye, jnp.inf, r)
    # idx[j] = N_DIRECT_NBR nearest-neighbor indices of agent j.
    _, idx = jax.lax.top_k(-r_safe, N_DIRECT_NBR)             # (n, K_dir)
    # Adjacency A[j, i] = 1 if i is in idx[j]. Then received[i] = A.T @ msgs.
    onehot = jax.nn.one_hot(idx, n, dtype=msgs.dtype)         # (n, K_dir, n)
    A = onehot.sum(axis=1)                                    # (n, n)
    return A.T @ msgs                                         # (n, d_msg)


def integrate(pos: jnp.ndarray, vel: jnp.ndarray,
              action: jnp.ndarray, p: SwarmParams):
    """Holonomic step. (ax, ay) ∈ [-1, 1]² × max_accel. Speed clamped to max_speed.
    Positions hard-clipped to world bounds."""
    accel = jnp.clip(action[:, :2], -1.0, 1.0) * p.max_accel
    new_vel = (vel + accel * p.dt) * p.vel_damp
    speed = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
    over = speed > p.max_speed
    new_vel = jnp.where(over,
                         new_vel * p.max_speed / (speed + 1e-8),
                         new_vel)
    new_pos = pos + new_vel * p.dt
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], 0.0, p.world_w),
        jnp.clip(new_pos[:, 1], 0.0, p.world_h),
    ], axis=-1)
    return new_pos, new_vel


# --------------------------------------------------------------------------
# Chamfer reward (centroid-aligned, symmetric mean nearest neighbor)
# --------------------------------------------------------------------------


def chamfer_centered(pos: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Symmetric mean nearest-neighbor distance, after shifting pos so its
    centroid matches target's. Returns scalar (jnp)."""
    shifted = pos - pos.mean(axis=0) + target.mean(axis=0)
    d = jnp.linalg.norm(shifted[:, None, :] - target[None, :, :], axis=-1)
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def fraction_outside_safe(pos: jnp.ndarray, p: SwarmParams) -> jnp.ndarray:
    """Fraction of agents whose position is outside the centered safe_box."""
    margin = (p.world_w - p.safe_box) / 2.0
    out_x = (pos[:, 0] < margin) | (pos[:, 0] > p.world_w - margin)
    out_y = (pos[:, 1] < margin) | (pos[:, 1] > p.world_h - margin)
    return jnp.mean((out_x | out_y).astype(jnp.float32))
