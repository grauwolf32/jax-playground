"""swarm-v0: 2D agent swarm restoring a perturbed formation.

Per env (single jax-vmap unit): n_agents share weights. Each step, every
agent picks (ax, ay), emits 3K wave amplitudes (K per band: short/medium/
long range, exponential decay), and unicasts a d_msg-vector to its 3
nearest neighbors. The state holds positions, velocities, the target
shape, the most-recent emissions + messages (used to build the *next*
obs), and a frozen "home" snapshot — what each agent's wave sensor +
neighbor vectors + inbound messages looked like with everyone at home
emitting unit waves and unit messages. The home snapshot is the agent's
positional address in the shape; without it, identical agents in identical
local neighborhoods would be indistinguishable.

Per-agent obs (OBS_DIM = 12·N_BANDS·K + 28 + 2·d_msg; with K=4, N_BANDS=3, d_msg=4 → 180):
   live waves          6·3·K     sensor reading from previous step's emissions
   own velocity        2
   nbr vectors         N_NBR·2   nearest-neighbor displacement vectors (gated)
   rel-to-centroid     2         own pos minus swarm centroid
   home waves          6·3·K     stored home wave fingerprint
   home nbr vectors    N_NBR·2   stored home neighbor structure
   received msgs       d_msg     sum of unicast messages from agents that have me in their top-3
   home received       d_msg     same, computed at home with everyone emitting unit messages

Action (ACT_DIM = 2 + N_BANDS·K + d_msg; with K=4, N_BANDS=3, d_msg=4 → 18):
   ax, ay                                       holonomic accel, clipped to [-1, 1]
   w_short_1..K, w_med_1..K, w_long_1..K        wave amplitudes per band, clipped to [-1, 1]
   msg_1..d_msg                                 unicast payload for top-3 nearest neighbors

Reward (shared, broadcast across agents in make_batched):
   -chamfer/target_scale - boundary_pen + step_penalty + (terminal_bonus if solved)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground.envs.swarm import physics as P
from jax_playground.envs.swarm.physics import SwarmParams


# Module-level dims assume defaults K=4, N_BANDS=3, d_msg=4. The trainer
# reads these once at registry time; if you change K / d_msg in params,
# also bump these.
_DEFAULT_K = 4
_DEFAULT_DMSG = 4
OBS_DIM = 12 * P.N_BANDS * _DEFAULT_K + 28 + 2 * _DEFAULT_DMSG
ACT_DIM = 2 + P.N_BANDS * _DEFAULT_K + _DEFAULT_DMSG

# Marker so envs.make_batched knows to flatten n_envs · n_agents into the
# trainer's batch dim and to broadcast the per-env scalar reward / done.
IS_SWARM = True


class EnvState(NamedTuple):
    pos: jnp.ndarray              # (n_agents, 2)
    vel: jnp.ndarray              # (n_agents, 2)
    target: jnp.ndarray           # (n_agents, 2) — shape positions, world coords
    last_emissions: jnp.ndarray   # (n_agents, N_BANDS, K) — used to build next obs
    last_msgs: jnp.ndarray        # (n_agents, d_msg) — direct unicast payload from prev step
    home_wave: jnp.ndarray        # (n_agents, N_SECTORS·N_BANDS·K) — frozen
    home_nbr: jnp.ndarray         # (n_agents, N_NBR·2) — frozen
    home_received: jnp.ndarray    # (n_agents, d_msg) — direct-msg fingerprint at home
    shape_id: jnp.ndarray         # () int — for diagnostics
    step: jnp.ndarray             # () int
    key: jax.Array


def _build_obs(es: EnvState, p: SwarmParams) -> jnp.ndarray:
    n = p.n_agents
    K = p.K
    # Live wave field built from last_emissions (which were *just emitted* in
    # the previous step, then we moved to current pos).
    live = P.wave_sensor(es.pos, es.last_emissions, p)         # (n, S, B, K)
    live_flat = live.reshape(n, P.N_SECTORS * P.N_BANDS * K)

    nbr_vecs = P.neighbor_vectors(es.pos, p)                   # (n, N_NBR, 2)
    nbr_flat = (nbr_vecs / p.r_max_nbr).reshape(n, P.N_NBR * 2)

    centroid = es.pos.mean(axis=0, keepdims=True)              # (1, 2)
    rel_centroid = (es.pos - centroid) / p.r_max_nbr           # (n, 2)

    own_vel = es.vel / p.max_speed                             # (n, 2)

    # Direct unicast: each agent receives msgs from senders that hold it in
    # their top-N_DIRECT_NBR. Built from last step's outgoing messages and the
    # current positions (so neighbor topology reflects "now" — the message is
    # delivered to whoever is now nearest the sender).
    received = P.direct_message_pass(es.pos, es.last_msgs)     # (n, d_msg)

    return jnp.concatenate([
        live_flat,                                              # 6·N_BANDS·K
        own_vel,                                                # 2
        nbr_flat,                                               # 12
        rel_centroid,                                           # 2
        es.home_wave,                                           # 6·N_BANDS·K
        es.home_nbr,                                            # 12
        received,                                               # d_msg
        es.home_received,                                       # d_msg
    ], axis=-1).astype(jnp.float32)


def _clip_world(pos: jnp.ndarray, p: SwarmParams) -> jnp.ndarray:
    return jnp.stack([
        jnp.clip(pos[:, 0], 0.0, p.world_w),
        jnp.clip(pos[:, 1], 0.0, p.world_h),
    ], axis=-1)


def env_reset(key: jax.Array, p: SwarmParams):
    k_target, k_disp, k_next = jax.random.split(key, 3)
    target, shape_id = P.sample_target(k_target, p)            # (n, 2), ()

    # Home snapshot: positions = target, all emissions/messages = 1.
    ones_em = jnp.ones((p.n_agents, P.N_BANDS, p.K), dtype=jnp.float32)
    home_wave_full = P.wave_sensor(target, ones_em, p)         # (n, S, B, K)
    home_wave = home_wave_full.reshape(p.n_agents, -1)

    home_nbr_full = P.neighbor_vectors(target, p)              # (n, N_NBR, 2)
    home_nbr = (home_nbr_full / p.r_max_nbr).reshape(p.n_agents, -1)

    ones_msg = jnp.ones((p.n_agents, p.d_msg), dtype=jnp.float32)
    home_received = P.direct_message_pass(target, ones_msg)    # (n, d_msg)

    # Scatter from home.
    disp = jax.random.uniform(k_disp, (p.n_agents, 2),
                              minval=-p.displacement,
                              maxval=p.displacement)
    pos = _clip_world(target + disp, p)
    vel = jnp.zeros((p.n_agents, 2), dtype=jnp.float32)
    last_em = jnp.zeros((p.n_agents, P.N_BANDS, p.K), dtype=jnp.float32)
    last_msgs = jnp.zeros((p.n_agents, p.d_msg), dtype=jnp.float32)

    es = EnvState(
        pos=pos, vel=vel, target=target,
        last_emissions=last_em, last_msgs=last_msgs,
        home_wave=home_wave, home_nbr=home_nbr, home_received=home_received,
        shape_id=shape_id, step=jnp.int32(0), key=k_next,
    )
    return es, _build_obs(es, p)


def env_step(es: EnvState, action: jnp.ndarray, p: SwarmParams):
    """action: (n_agents, ACT_DIM)."""
    new_pos, new_vel = P.integrate(es.pos, es.vel, action, p)
    _, new_em, new_msgs = P.parse_action(action, p)

    # Reward — shared across agents. Chamfer normalized by the *target shape's*
    # own scale (mean distance from its centroid) so the signal is invariant
    # to sampled rotation/scale and to world size. With this norm,
    # shape_err_n=1 means "off by one shape radius".
    shape_err = P.chamfer_centered(new_pos, es.target)
    target_scale = jnp.mean(
        jnp.linalg.norm(es.target - es.target.mean(axis=0), axis=-1)
    )
    shape_err_n = shape_err / (target_scale + 1e-6)

    out_frac = P.fraction_outside_safe(new_pos, p)
    boundary_pen = p.boundary_penalty * out_frac

    new_step = es.step + 1
    solved = shape_err_n < p.solve_threshold
    truncated = new_step >= p.timestep_limit
    done = truncated | solved
    terminal = jnp.where(solved, p.terminal_bonus, 0.0)

    reward = (-shape_err_n + boundary_pen + p.step_penalty + terminal).astype(jnp.float32)

    next_key, k_reset = jax.random.split(es.key)
    new_es = EnvState(
        pos=new_pos, vel=new_vel, target=es.target,
        last_emissions=new_em, last_msgs=new_msgs,
        home_wave=es.home_wave, home_nbr=es.home_nbr,
        home_received=es.home_received,
        shape_id=es.shape_id, step=new_step.astype(jnp.int32),
        key=next_key,
    )

    # Auto-reset on done — same pattern as gathering env. Build a fresh
    # state from k_reset, then `where`-blend at every leaf.
    reset_es, _ = env_reset(k_reset, p)
    final_es = jax.tree_util.tree_map(
        lambda r, n: jnp.where(done, r, n), reset_es, new_es,
    )
    obs = _build_obs(final_es, p)
    info = {
        "shape_err": shape_err,
        "shape_err_n": shape_err_n,
        "solved": solved.astype(jnp.float32),
        "out_frac": out_frac,
    }
    return final_es, obs, reward, done, info
