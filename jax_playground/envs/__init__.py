"""Env registry. Each env exports `make_batched(params, n_envs)`.

Convention per env module:
    - `physics.default_params() -> params`
    - `env.env_reset(key, params) -> (state, obs)`
    - `env.env_step(state, action, params) -> (state, obs, reward, done, info)`
    - module-level OBS_DIM, ACT_DIM constants

`make_batched(env_kind, params, n_envs)` vmaps the per-env reset/step and
returns (reset_batch_fn, step_batch_fn, obs_dim, act_dim).
"""

from __future__ import annotations

import jax

from jax_playground.envs.hyrosphere import env as hyro_env
from jax_playground.envs.hyrosphere.physics import default_params as default_hyro_params
from jax_playground.envs.linearsphere import env as linear_env
from jax_playground.envs.linearsphere.physics import default_params as default_linear_params


REGISTRY = {
    "hyro": {
        "env": hyro_env,
        "default_params": default_hyro_params,
        "obs_dim": hyro_env.OBS_DIM,
        "act_dim": hyro_env.ACT_DIM,
    },
    "linear": {
        "env": linear_env,
        "default_params": default_linear_params,
        "obs_dim": linear_env.OBS_DIM,
        "act_dim": linear_env.ACT_DIM,
    },
}


# Late registration so importing this module is cheap if pursuit/gathering
# deps fail (e.g. before flax/jax are installed). Each env wires itself in.
try:
    from jax_playground.envs.pursuit import env as pursuit_env  # noqa: F401
    from jax_playground.envs.pursuit.vehicle import default_params as default_pursuit_params
    REGISTRY["pursuit"] = {
        "env": pursuit_env,
        "default_params": default_pursuit_params,
        "obs_dim": pursuit_env.OBS_DIM,
        "act_dim": pursuit_env.ACT_DIM,
    }
except ImportError:
    pass

try:
    from jax_playground.envs.gathering import env as gathering_env  # noqa: F401
    from jax_playground.envs.pursuit.vehicle import default_params as default_gathering_params
    REGISTRY["gathering"] = {
        "env": gathering_env,
        "default_params": default_gathering_params,
        "obs_dim": gathering_env.OBS_DIM,
        "act_dim": gathering_env.ACT_DIM,
    }
except ImportError:
    pass


def list_envs() -> list[str]:
    return list(REGISTRY.keys())


def make_batched(env_kind: str, params, n_envs: int):
    if env_kind not in REGISTRY:
        raise ValueError(f"unknown env: {env_kind!r}. Choices: {list_envs()}")
    entry = REGISTRY[env_kind]
    env_mod = entry["env"]
    obs_dim, act_dim = entry["obs_dim"], entry["act_dim"]

    @jax.jit
    def reset_batch(key: jax.Array):
        keys = jax.random.split(key, n_envs)
        return jax.vmap(lambda k: env_mod.env_reset(k, params))(keys)

    @jax.jit
    def step_batch(states, actions):
        return jax.vmap(lambda s, a: env_mod.env_step(s, a, params))(states, actions)

    return reset_batch, step_batch, obs_dim, act_dim
