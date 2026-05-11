"""Pure JAX physics for the HyroSphere — tetrahedral 4-wheel sphere.

Dynamics are formulated about the ball center O (modified Euler with
L_O ≈ J·Ω approximation). See ../../../docs (in the openai-physics sibling
repo) for the math, or the inline comments below.

All step functions are pure: `step(state, action, params) → state`. State is
a NamedTuple PyTree so JAX can flatten it for vmap / jit / grad.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_playground._math import G, Z_HAT, rodrigues


# Tetrahedral axis frame built once with numpy, then wrapped as jnp constant.
def _build_tetra_U() -> np.ndarray:
    u1 = np.array([0.0, 0.0, 1.0])
    theta = np.arccos(-1.0 / 3.0)
    Rx = np.array([
        [1.0, 0.0,             0.0],
        [0.0, np.cos(theta),  -np.sin(theta)],
        [0.0, np.sin(theta),   np.cos(theta)],
    ])
    u2 = Rx @ u1
    phi = 2.0 * np.pi / 3.0
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0.0],
        [np.sin(phi),  np.cos(phi), 0.0],
        [0.0,          0.0,         1.0],
    ])
    u3 = Rz @ u2
    u4 = Rz @ u3
    U = np.stack([u1, u2, u3, u4], axis=0)
    return U / np.linalg.norm(U, axis=1, keepdims=True)


_TETRA_U = jnp.asarray(_build_tetra_U(), dtype=jnp.float32)


class HyroParams(NamedTuple):
    t_len: jnp.ndarray
    radius: jnp.ndarray
    mass: jnp.ndarray
    dot_masses: jnp.ndarray        # (4,)
    total_mass: jnp.ndarray
    J_ball: jnp.ndarray            # (3, 3)
    mu: jnp.ndarray
    max_omega: jnp.ndarray
    friction_loss: jnp.ndarray
    dt: jnp.ndarray
    n: int                         # static — 4 wheels


class HyroState(NamedTuple):
    position: jnp.ndarray     # (3,)
    velocity: jnp.ndarray
    Omega: jnp.ndarray
    dOmegadt: jnp.ndarray
    phi: jnp.ndarray          # (4,)
    omega: jnp.ndarray        # (4,)
    ksi: jnp.ndarray          # (4,)
    U: jnp.ndarray            # (4, 3)
    last_F_fric: jnp.ndarray
    in_contact: jnp.ndarray
    peak_z: jnp.ndarray


def default_params(t_len: float = 1.0,
                   mass: float = 4.0,
                   dot_masses: jnp.ndarray | None = None,
                   mu: float = 0.15,
                   max_omega: float = 100.0,
                   friction_loss: float = 0.001,
                   dt: float = 0.01) -> HyroParams:
    if dot_masses is None:
        dot_masses = jnp.array([2.0, 2.0, 2.0, 2.0], dtype=jnp.float32)
    radius = t_len * jnp.sqrt(3.0 / 8.0)
    total_mass = mass + jnp.sum(dot_masses)
    J_ball = (2.0 / 3.0 * mass * radius ** 2) * jnp.eye(3)
    return HyroParams(
        t_len=jnp.float32(t_len),
        radius=jnp.float32(radius),
        mass=jnp.float32(mass),
        dot_masses=dot_masses.astype(jnp.float32),
        total_mass=jnp.float32(total_mass),
        J_ball=J_ball.astype(jnp.float32),
        mu=jnp.float32(mu),
        max_omega=jnp.float32(max_omega),
        friction_loss=jnp.float32(friction_loss),
        dt=jnp.float32(dt),
        n=4,
    )


def reset(key: jax.Array, params: HyroParams) -> HyroState:
    k_phi, k_omega, k_Omega, k_v = jax.random.split(key, 4)
    return HyroState(
        position=jnp.array([0.0, 0.0, params.radius], dtype=jnp.float32),
        velocity=jnp.concatenate([
            jax.random.uniform(k_v, (2,), minval=-0.1, maxval=0.1),
            jnp.zeros(1),
        ]).astype(jnp.float32),
        Omega=jax.random.uniform(k_Omega, (3,), minval=-0.3, maxval=0.3).astype(jnp.float32),
        dOmegadt=jnp.zeros(3, dtype=jnp.float32),
        phi=jax.random.uniform(k_phi, (4,), minval=-jnp.pi, maxval=jnp.pi).astype(jnp.float32),
        omega=jax.random.uniform(k_omega, (4,), minval=-5.0, maxval=5.0).astype(jnp.float32),
        ksi=jnp.zeros(4, dtype=jnp.float32),
        U=_TETRA_U,
        last_F_fric=jnp.zeros(3, dtype=jnp.float32),
        in_contact=jnp.bool_(True),
        peak_z=jnp.float32(params.radius),
    )


def wheel_geometry(state: HyroState, params: HyroParams):
    """R_i and B_i for all 4 wheels — needed by env obs and the renderer adapter."""
    U = state.U
    A = U * jnp.sqrt(1.0 / 8.0)
    A_next = jnp.roll(A, -1, axis=0)
    b_raw = jnp.cross(A, A_next, axis=1)
    b_norms = jnp.linalg.norm(b_raw, axis=1, keepdims=True)
    b_base = b_raw / b_norms * (params.t_len / 2.0)
    B = jax.vmap(lambda axis, angle, v: rodrigues(axis, angle) @ v)(U, state.phi, b_base)
    R = A + B
    return R, B


def step(state: HyroState, action: jnp.ndarray, params: HyroParams) -> HyroState:
    """One physics step. `action` (4,) is the next wheel angular acceleration."""
    R, _ = wheel_geometry(state, params)
    U = state.U
    dt = params.dt
    os_vec = jnp.array([0.0, 0.0, -params.radius])

    # Kinematics in non-rotating frame at O.
    omega_total = state.omega[:, None] * U + state.Omega[None, :]
    dRdt = jnp.cross(omega_total, R, axis=1)
    domega_total = (
        state.ksi[:, None] * U
        + state.omega[:, None] * jnp.cross(state.Omega[None, :], U, axis=1)
        + state.dOmegadt[None, :]
    )
    d2Rdt2 = jnp.cross(domega_total, R, axis=1) + jnp.cross(omega_total, dRdt, axis=1)

    # Inertia tensor about O.
    R_norm_sq = jnp.sum(R * R, axis=1)
    sum_m_R2 = jnp.sum(params.dot_masses * R_norm_sq)
    R_w = R * params.dot_masses[:, None]
    R_outer = R_w.T @ R
    J = params.J_ball + sum_m_R2 * jnp.eye(3) - R_outer

    # dJ/dt with trace term.
    R_dot_dR = jnp.sum(R * dRdt, axis=1)
    sum_m_RdotDR = jnp.sum(params.dot_masses * R_dot_dR)
    term_R_dR = (R * params.dot_masses[:, None]).T @ dRdt
    term_dR_R = (dRdt * params.dot_masses[:, None]).T @ R
    dJdt = 2.0 * sum_m_RdotDR * jnp.eye(3) - term_R_dR - term_dR_R

    # Friction at contact point.
    in_contact = state.position[2] <= params.radius + 1e-3
    v_S = state.velocity + jnp.cross(state.Omega, os_vec)
    v_S_z = v_S[2]
    v_S_clean = jnp.where(v_S_z < 0.0, v_S - v_S_z * Z_HAT, v_S)
    v_S_norm = jnp.linalg.norm(v_S_clean)
    F_fric_dir = jnp.where(
        v_S_norm > 1e-4,
        -v_S_clean / jnp.maximum(v_S_norm, 1e-9),
        jnp.zeros(3),
    )
    F_fric_active = params.mu * params.total_mass * 9.8 * F_fric_dir
    F_fric = jnp.where(in_contact, F_fric_active, jnp.zeros(3))

    # External torques about O.
    gravity_torques = jnp.cross(R, params.dot_masses[:, None] * G[None, :], axis=1)
    Ms_all = jnp.cross(os_vec, F_fric) + jnp.sum(gravity_torques, axis=0)

    # Modified Euler equation.
    Ks = J @ state.Omega
    rhs = Ms_all - dJdt @ state.Omega - jnp.cross(state.Omega, Ks)
    new_dOmegadt = jnp.linalg.solve(J, rhs)

    # Linear dynamics.
    F_ext = params.total_mass * G + F_fric
    com_accel_rel_O = jnp.sum(params.dot_masses[:, None] * d2Rdt2, axis=0) / params.total_mass
    dvcdt = F_ext / params.total_mass - com_accel_rel_O

    new_velocity_raw = state.velocity + dvcdt * dt
    needs_clamp = in_contact & (new_velocity_raw[2] < 0.0)
    new_velocity = jnp.where(needs_clamp,
                              new_velocity_raw - new_velocity_raw[2] * Z_HAT,
                              new_velocity_raw)
    new_position = state.position + new_velocity * dt

    # Wheel state.
    new_omega_raw = state.omega + state.ksi * dt
    new_omega = jnp.sign(new_omega_raw) * jnp.minimum(jnp.abs(new_omega_raw), params.max_omega)
    new_phi = jnp.mod(state.phi + new_omega * dt + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # Body orientation.
    new_Omega = (state.Omega + new_dOmegadt * dt) * (1.0 - params.friction_loss)
    rot_angle = jnp.linalg.norm(new_Omega) * dt
    M_rot = rodrigues(new_Omega, rot_angle)
    new_U = state.U @ M_rot.T

    return HyroState(
        position=new_position,
        velocity=new_velocity,
        Omega=new_Omega,
        dOmegadt=new_dOmegadt,
        phi=new_phi,
        omega=new_omega,
        ksi=action.astype(jnp.float32),
        U=new_U,
        last_F_fric=F_fric,
        in_contact=in_contact,
        peak_z=jnp.maximum(state.peak_z, new_position[2]),
    )
