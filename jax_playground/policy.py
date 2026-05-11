"""Shared actor-critic + obs-normalisation utilities for PPO.

Kept out of train.py so that play.py and viewer.py can load saved
checkpoints without train.py being the unpickling `__main__`. Anything
referenced by the saved `model.pkl` (RunningStats, ActorCritic) lives
here.
"""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Actor-critic network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    act_dim: int
    hidden: tuple = (256, 256)
    log_std_init: float = 0.0

    @nn.compact
    def __call__(self, x):
        a = x
        for h in self.hidden:
            a = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(a))
        mean = nn.Dense(
            self.act_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(a)
        log_std_raw = self.param("log_std", lambda _, shape: jnp.full(shape, self.log_std_init),
                                  (self.act_dim,))
        # Clamp log_std tightly so the policy can't collapse std → 0 (which
        # makes log_prob diverge → NaN). Lower bound -2 keeps std ≥ 0.135.
        log_std = jnp.clip(log_std_raw, -2.0, 2.0)

        c = x
        for h in self.hidden:
            c = nn.tanh(nn.Dense(h, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(c))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(c).squeeze(-1)
        return mean, log_std, value


def gaussian_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * ((action - mean) / std) ** 2,
        axis=-1,
    )


def gaussian_entropy(log_std):
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std, axis=-1)


# ---------------------------------------------------------------------------
# Observation normalisation (Welford running mean/var)
# ---------------------------------------------------------------------------


class RunningStats(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_stats(dim: int) -> RunningStats:
    return RunningStats(
        mean=jnp.zeros(dim, dtype=jnp.float32),
        var=jnp.ones(dim, dtype=jnp.float32),
        count=jnp.float32(1e-4),
    )


def update_running_stats(stats: RunningStats, batch: jnp.ndarray) -> RunningStats:
    batch_count = jnp.float32(batch.shape[0])
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    delta = batch_mean - stats.mean
    tot = stats.count + batch_count
    new_mean = stats.mean + delta * batch_count / tot
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta ** 2 * stats.count * batch_count / tot
    new_var = m2 / tot
    return RunningStats(mean=new_mean, var=new_var, count=tot)


def normalize(obs, stats: RunningStats, clip: float = 10.0):
    return jnp.clip((obs - stats.mean) / jnp.sqrt(stats.var + 1e-8), -clip, clip)
