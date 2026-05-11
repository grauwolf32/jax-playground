"""JAX-native physics + PPO playground.

Envs are registered under jax_playground.envs; see envs/__init__.py for the
`REGISTRY` dict and `make_batched(env_kind, params, n_envs)` factory.

Shared utilities:
    jax_playground.policy   actor-critic, Welford running stats
    jax_playground._math    rodrigues, skew, wrap_angle, gravity constants
    jax_playground.viz3d    OpenGL helpers for 3D envs
"""

from jax_playground import envs as envs   # noqa: F401
from jax_playground import policy as policy   # noqa: F401
