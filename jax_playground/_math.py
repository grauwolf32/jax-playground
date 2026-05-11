"""Small JAX math helpers shared across env physics.

Kept at the package root with a leading underscore so it's clearly internal;
env modules import from here rather than redefining `rodrigues` etc.
"""

from __future__ import annotations

import jax.numpy as jnp


G = jnp.array([0.0, 0.0, -9.8])
Z_HAT = jnp.array([0.0, 0.0, 1.0])


def skew(v: jnp.ndarray) -> jnp.ndarray:
    """3x3 skew-symmetric (cross-product) matrix for axis v.

    [v]_x such that [v]_x w = v × w. Standard convention.
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
    K = skew(unit)
    R = jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)
    return jnp.where(norm > 1e-4, R, jnp.eye(3))


def wrap_angle(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap an angle (or batch of angles) into [-pi, pi]."""
    return jnp.mod(a + jnp.pi, 2.0 * jnp.pi) - jnp.pi
