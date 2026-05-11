"""Pure JAX physics for HyroSphere and LinearSphere.

All functions are pure: `step(state, action, params) -> state`. State is a
NamedTuple PyTree so JAX can pytree-flatten it for vmap / jit / grad. No
mutation, no global state. Branches use `jnp.where` (data-dependent) or
`jax.lax.cond` (control-flow-dependent) rather than Python `if`.

State conventions match the numpy reference in ../openai-physics:
all dynamics formulated about the ball center O, modified Euler with the
L_O ≈ J·Ω approximation. See ../openai-physics/docs/dynamics.md for the math.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


G = jnp.array([0.0, 0.0, -9.8])
Z_HAT = jnp.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _skew(v: jnp.ndarray) -> jnp.ndarray:
    """3x3 skew-symmetric (cross-product) matrix for axis v.

    [v]_x such that [v]_x w = v × w. Same convention as numpy reference.
    """
    return jnp.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def rodrigues(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix for rotation by `angle` around `axis`.

    Returns the identity if |axis| is below 1e-4 (avoids NaNs at zero Ω).
    """
    norm = jnp.linalg.norm(axis)
    safe_norm = jnp.where(norm > 1e-4, norm, 1.0)
    unit = axis / safe_norm
    K = _skew(unit)
    R = jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)
    return jnp.where(norm > 1e-4, R, jnp.eye(3))


# Build the tetrahedral axis frame once, at import time, with numpy. We then
# wrap it as a jnp array to use as a static constant inside the jit'd step.
def _build_tetra_U() -> np.ndarray:
    """Four unit vectors forming a regular tetrahedron, U[0] = +z."""
    u1 = np.array([0.0, 0.0, 1.0])
    theta = np.arccos(-1.0 / 3.0)
    # rotate u1 around +x by theta:
    Rx = np.array([
        [1.0, 0.0,             0.0],
        [0.0, np.cos(theta),  -np.sin(theta)],
        [0.0, np.sin(theta),   np.cos(theta)],
    ])
    u2 = Rx @ u1
    # then rotate twice around +z by 2π/3:
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

_CARD_U = jnp.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# HyroSphere
# ---------------------------------------------------------------------------


class HyroParams(NamedTuple):
    t_len: jnp.ndarray             # scalar — edge length
    radius: jnp.ndarray            # scalar — t_len * sqrt(3/8)
    mass: jnp.ndarray              # scalar — ball mass
    dot_masses: jnp.ndarray        # (4,)
    total_mass: jnp.ndarray        # scalar
    J_ball: jnp.ndarray            # (3, 3) shell inertia about O
    mu: jnp.ndarray                # scalar — Coulomb friction
    max_omega: jnp.ndarray         # scalar — wheel speed cap
    friction_loss: jnp.ndarray     # scalar — Ω damping per step
    dt: jnp.ndarray                # scalar
    n: int                         # static — 4 wheels


class HyroState(NamedTuple):
    position: jnp.ndarray     # (3,)
    velocity: jnp.ndarray     # (3,)
    Omega: jnp.ndarray        # (3,) body angular velocity
    dOmegadt: jnp.ndarray     # (3,) body angular accel (one-step lag)
    phi: jnp.ndarray          # (4,) wheel angles
    omega: jnp.ndarray        # (4,) wheel speeds
    ksi: jnp.ndarray          # (4,) wheel angular accels (last action)
    U: jnp.ndarray            # (4, 3) body-fixed axes in world coords
    last_F_fric: jnp.ndarray  # (3,) diagnostic only
    in_contact: jnp.ndarray   # scalar bool — diagnostic
    peak_z: jnp.ndarray       # scalar — episode max z


def default_hyro_params(t_len: float = 1.0,
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


def hyro_reset(key: jax.Array, params: HyroParams) -> HyroState:
    """Initial state with light per-episode randomization (matches numpy env)."""
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


def _hyro_wheel_geometry(state: HyroState, params: HyroParams):
    """R_i (mass positions rel to O) and B_i (orbital offsets) for all wheels."""
    U = state.U
    A = U * jnp.sqrt(1.0 / 8.0)                  # (4, 3) orbit centers
    A_next = jnp.roll(A, -1, axis=0)
    b_raw = jnp.cross(A, A_next, axis=1)         # (4, 3) perpendicular-to-U directions
    b_norms = jnp.linalg.norm(b_raw, axis=1, keepdims=True)
    b_base = b_raw / b_norms * (params.t_len / 2.0)
    # Rotate each b_base by phi[i] around U[i]
    B = jax.vmap(lambda axis, angle, v: rodrigues(axis, angle) @ v)(U, state.phi, b_base)
    R = A + B
    return R, B


def hyro_step(state: HyroState, action: jnp.ndarray, params: HyroParams) -> HyroState:
    """One physics step for HyroSphere.

    `action` (shape (4,)) is the new wheel angular acceleration applied at the
    end of this step (the *current* ksi was already in state, set on the
    previous call). This matches the numpy env's semantics.
    """
    R, _ = _hyro_wheel_geometry(state, params)
    U = state.U
    dt = params.dt
    n = params.n
    os_vec = jnp.array([0.0, 0.0, -params.radius])

    # --- Kinematics in non-rotating frame at O ----------------------------
    omega_total = state.omega[:, None] * U + state.Omega[None, :]    # (4, 3)
    dRdt = jnp.cross(omega_total, R, axis=1)                          # (4, 3)
    domega_total = (
        state.ksi[:, None] * U
        + state.omega[:, None] * jnp.cross(state.Omega[None, :], U, axis=1)
        + state.dOmegadt[None, :]
    )                                                                 # (4, 3)
    d2Rdt2 = jnp.cross(domega_total, R, axis=1) + jnp.cross(omega_total, dRdt, axis=1)

    # --- Inertia tensor about O ------------------------------------------
    # J = J_ball + Σ m_i (|R|² I − R⊗R)
    R_norm_sq = jnp.sum(R * R, axis=1)                                # (4,)
    sum_m_R2 = jnp.dot(state.U.dtype.type(state.U.shape[0]) * 0 + params.dot_masses, R_norm_sq)  # no-op coerce
    sum_m_R2 = jnp.sum(params.dot_masses * R_norm_sq)
    R_w = R * params.dot_masses[:, None]                              # m_i R_i
    R_outer = R_w.T @ R                                               # (3, 3)
    J = params.J_ball + sum_m_R2 * jnp.eye(3) - R_outer

    # --- dJ/dt with trace term -------------------------------------------
    R_dot_dR = jnp.sum(R * dRdt, axis=1)                              # (4,)
    sum_m_RdotDR = jnp.sum(params.dot_masses * R_dot_dR)
    term_R_dR = (R * params.dot_masses[:, None]).T @ dRdt             # (3, 3)
    term_dR_R = (dRdt * params.dot_masses[:, None]).T @ R             # (3, 3)
    dJdt = 2.0 * sum_m_RdotDR * jnp.eye(3) - term_R_dR - term_dR_R

    # --- Friction at contact point ---------------------------------------
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

    # --- External torques about O ----------------------------------------
    gravity_torques = jnp.cross(R, params.dot_masses[:, None] * G[None, :], axis=1)
    Ms_all = jnp.cross(os_vec, F_fric) + jnp.sum(gravity_torques, axis=0)

    # --- Modified Euler equation -----------------------------------------
    Ks = J @ state.Omega
    rhs = Ms_all - dJdt @ state.Omega - jnp.cross(state.Omega, Ks)
    new_dOmegadt = jnp.linalg.solve(J, rhs)

    # --- Linear dynamics: a_O = a_COM − Σm_i d²R_i / M_total -------------
    F_ext = params.total_mass * G + F_fric
    com_accel_rel_O = jnp.sum(params.dot_masses[:, None] * d2Rdt2, axis=0) / params.total_mass
    dvcdt = F_ext / params.total_mass - com_accel_rel_O

    new_velocity_raw = state.velocity + dvcdt * dt
    # Contact clamp (normal impulse): zero downward v_z if in contact
    needs_clamp = in_contact & (new_velocity_raw[2] < 0.0)
    new_velocity = jnp.where(needs_clamp,
                              new_velocity_raw - new_velocity_raw[2] * Z_HAT,
                              new_velocity_raw)
    new_position = state.position + new_velocity * dt

    # --- Wheel state update -----------------------------------------------
    new_omega_raw = state.omega + state.ksi * dt
    new_omega = jnp.sign(new_omega_raw) * jnp.minimum(jnp.abs(new_omega_raw), params.max_omega)
    new_phi = jnp.mod(state.phi + new_omega * dt + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    # --- Body orientation update ------------------------------------------
    new_Omega = (state.Omega + new_dOmegadt * dt) * (1.0 - params.friction_loss)
    rot_angle = jnp.linalg.norm(new_Omega) * dt
    M_rot = rodrigues(new_Omega, rot_angle)
    # numpy reference does U @ M_rot.T. Match it.
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


# ---------------------------------------------------------------------------
# LinearSphere
# ---------------------------------------------------------------------------


class LinearParams(NamedTuple):
    radius: jnp.ndarray
    mass: jnp.ndarray
    dot_masses: jnp.ndarray        # (6,)
    total_mass: jnp.ndarray
    J_ball: jnp.ndarray            # (3, 3)
    mu: jnp.ndarray
    max_speed: jnp.ndarray
    friction_loss: jnp.ndarray
    dt: jnp.ndarray
    n: int                         # 6 sliders


class LinearState(NamedTuple):
    position: jnp.ndarray     # (3,)
    velocity: jnp.ndarray     # (3,)
    Omega: jnp.ndarray
    dOmegadt: jnp.ndarray
    shifts: jnp.ndarray       # (6,) ∈ [0, radius]
    speeds: jnp.ndarray       # (6,)
    accelerations: jnp.ndarray  # (6,) — last action
    U: jnp.ndarray            # (6, 3)
    last_F_fric: jnp.ndarray
    in_contact: jnp.ndarray
    peak_z: jnp.ndarray


def default_linear_params(radius: float = 1.0,
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


def linear_reset(key: jax.Array, params: LinearParams) -> LinearState:
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


def linear_step(state: LinearState, action: jnp.ndarray, params: LinearParams) -> LinearState:
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

    # Torques: gravity per dot mass + friction at S. Slider boundary impulses
    # act along U_i so R × U = 0 → contribute no torque.
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

    # Slider update + boundary handling
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

    # Momentum-conserving boundary impulse on v_O: sum across all violating sliders
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
