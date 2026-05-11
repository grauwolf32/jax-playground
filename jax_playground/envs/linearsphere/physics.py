"""Pure JAX physics for LinearSphere — 6 sliders along cardinal axes.

Same dynamics formulation as HyroSphere (about ball center O) — see sibling
hyrosphere/physics.py for shared structure. The cardinal axis frame and
momentum-conserving slider boundary impulses are LinearSphere-specific.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_playground._math import G, Z_HAT, rodrigues


_CARD_U = jnp.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
], dtype=jnp.float32)


class LinearParams(NamedTuple):
    radius: jnp.ndarray
    mass: jnp.ndarray
    dot_masses: jnp.ndarray        # (6,)
    total_mass: jnp.ndarray
    J_ball: jnp.ndarray
    mu: jnp.ndarray
    max_speed: jnp.ndarray
    friction_loss: jnp.ndarray
    dt: jnp.ndarray
    n: int                         # 6 sliders


class LinearState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    Omega: jnp.ndarray
    dOmegadt: jnp.ndarray
    shifts: jnp.ndarray            # (6,) ∈ [0, radius]
    speeds: jnp.ndarray
    accelerations: jnp.ndarray
    U: jnp.ndarray                 # (6, 3)
    last_F_fric: jnp.ndarray
    in_contact: jnp.ndarray
    peak_z: jnp.ndarray


def default_params(radius: float = 1.0,
                   mass: float = 8.0,
                   dot_masses: jnp.ndarray | None = None,
                   mu: float = 0.15,
                   max_speed: float = 100.0,
                   friction_loss: float = 0.001,
                   dt: float = 0.01) -> LinearParams:
    if dot_masses is None:
        dot_masses = jnp.full((6,), 2.0, dtype=jnp.float32)
    total_mass = mass + jnp.sum(dot_masses)
    J_ball = (2.0 / 3.0 * mass * radius ** 2) * jnp.eye(3)
    return LinearParams(
        radius=jnp.float32(radius),
        mass=jnp.float32(mass),
        dot_masses=dot_masses.astype(jnp.float32),
        total_mass=jnp.float32(total_mass),
        J_ball=J_ball.astype(jnp.float32),
        mu=jnp.float32(mu),
        max_speed=jnp.float32(max_speed),
        friction_loss=jnp.float32(friction_loss),
        dt=jnp.float32(dt),
        n=6,
    )


def reset(key: jax.Array, params: LinearParams) -> LinearState:
    k_shift, k_speed, k_Omega, k_v = jax.random.split(key, 4)
    return LinearState(
        position=jnp.array([0.0, 0.0, params.radius], dtype=jnp.float32),
        velocity=jnp.concatenate([
            jax.random.uniform(k_v, (2,), minval=-0.1, maxval=0.1),
            jnp.zeros(1),
        ]).astype(jnp.float32),
        Omega=jax.random.uniform(k_Omega, (3,), minval=-0.3, maxval=0.3).astype(jnp.float32),
        dOmegadt=jnp.zeros(3, dtype=jnp.float32),
        shifts=jax.random.uniform(k_shift, (6,), minval=0.1, maxval=0.5).astype(jnp.float32),
        speeds=jax.random.uniform(k_speed, (6,), minval=-2.0, maxval=2.0).astype(jnp.float32),
        accelerations=jnp.zeros(6, dtype=jnp.float32),
        U=_CARD_U,
        last_F_fric=jnp.zeros(3, dtype=jnp.float32),
        in_contact=jnp.bool_(True),
        peak_z=jnp.float32(params.radius),
    )


def step(state: LinearState, action: jnp.ndarray, params: LinearParams) -> LinearState:
    R = state.shifts[:, None] * state.U
    U = state.U
    dt = params.dt
    os_vec = jnp.array([0.0, 0.0, -params.radius])

    # Kinematics
    v_body = state.speeds[:, None] * U
    dRdt = jnp.cross(state.Omega[None, :], R, axis=1) + v_body
    d2Rdt2 = (
        jnp.cross(state.dOmegadt[None, :], R, axis=1)
        + jnp.cross(state.Omega[None, :], jnp.cross(state.Omega[None, :], R, axis=1), axis=1)
        + 2.0 * jnp.cross(state.Omega[None, :], v_body, axis=1)
        + state.accelerations[:, None] * U
    )

    # Inertia
    R_norm_sq = jnp.sum(R * R, axis=1)
    sum_m_R2 = jnp.sum(params.dot_masses * R_norm_sq)
    R_w = R * params.dot_masses[:, None]
    R_outer = R_w.T @ R
    J = params.J_ball + sum_m_R2 * jnp.eye(3) - R_outer

    R_dot_dR = jnp.sum(R * dRdt, axis=1)
    sum_m_RdotDR = jnp.sum(params.dot_masses * R_dot_dR)
    term_R_dR = (R * params.dot_masses[:, None]).T @ dRdt
    term_dR_R = (dRdt * params.dot_masses[:, None]).T @ R
    dJdt = 2.0 * sum_m_RdotDR * jnp.eye(3) - term_R_dR - term_dR_R

    # Friction
    in_contact = state.position[2] <= params.radius + 1e-3
    v_S = state.velocity + jnp.cross(state.Omega, os_vec)
    v_S_z = v_S[2]
    v_S_clean = jnp.where(v_S_z < 0.0, v_S - v_S_z * Z_HAT, v_S)
    v_S_norm = jnp.linalg.norm(v_S_clean)
    F_fric_dir = jnp.where(v_S_norm > 1e-4, -v_S_clean / jnp.maximum(v_S_norm, 1e-9), jnp.zeros(3))
    F_fric_active = params.mu * params.total_mass * 9.8 * F_fric_dir
    F_fric = jnp.where(in_contact, F_fric_active, jnp.zeros(3))

    # Torques. Slider boundary impulses act along U_i so R × U = 0 → no torque.
    gravity_torques = jnp.cross(R, params.dot_masses[:, None] * G[None, :], axis=1)
    Ms_all = jnp.cross(os_vec, F_fric) + jnp.sum(gravity_torques, axis=0)

    Ks = J @ state.Omega
    new_dOmegadt = jnp.linalg.solve(J, Ms_all - dJdt @ state.Omega - jnp.cross(state.Omega, Ks))

    F_ext = params.total_mass * G + F_fric
    com_accel_rel_O = jnp.sum(params.dot_masses[:, None] * d2Rdt2, axis=0) / params.total_mass
    dvcdt = F_ext / params.total_mass - com_accel_rel_O

    new_velocity_raw = state.velocity + dvcdt * dt
    needs_clamp = in_contact & (new_velocity_raw[2] < 0.0)
    new_velocity = jnp.where(needs_clamp,
                              new_velocity_raw - new_velocity_raw[2] * Z_HAT,
                              new_velocity_raw)
    new_position = state.position + new_velocity * dt

    # Slider update + boundary impacts
    speeds_after_accel = state.speeds + state.accelerations * dt
    speeds_clipped = jnp.sign(speeds_after_accel) * jnp.minimum(
        jnp.abs(speeds_after_accel), params.max_speed
    )
    shifts_after = state.shifts + speeds_clipped * dt

    hit_inner = shifts_after <= 0.0
    hit_outer = shifts_after > params.radius
    violating_inner = hit_inner & (speeds_clipped < 0.0)
    violating_outer = hit_outer & (speeds_clipped > 0.0)
    violating = violating_inner | violating_outer

    impulse_dv = jnp.sum(
        violating[:, None].astype(jnp.float32)
        * params.dot_masses[:, None]
        * speeds_clipped[:, None]
        * U
        / params.total_mass,
        axis=0,
    )
    new_velocity = new_velocity + impulse_dv

    new_speeds = jnp.where(violating, 0.0, speeds_clipped)
    new_shifts = jnp.clip(shifts_after, 0.0, params.radius)
    new_accelerations = jnp.where(
        (hit_inner & (action < 0.0)) | (hit_outer & (action > 0.0)),
        0.0,
        action,
    )

    new_Omega = (state.Omega + new_dOmegadt * dt) * (1.0 - params.friction_loss)
    rot_angle = jnp.linalg.norm(new_Omega) * dt
    M_rot = rodrigues(new_Omega, rot_angle)
    new_U = state.U @ M_rot.T

    return LinearState(
        position=new_position,
        velocity=new_velocity,
        Omega=new_Omega,
        dOmegadt=new_dOmegadt,
        shifts=new_shifts,
        speeds=new_speeds,
        accelerations=new_accelerations.astype(jnp.float32),
        U=new_U,
        last_F_fric=F_fric,
        in_contact=in_contact,
        peak_z=jnp.maximum(state.peak_z, new_position[2]),
    )
