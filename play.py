"""Headless evaluation of a JAX PPO checkpoint.

Loads runs/<run>/model.pkl, runs `--episodes` rollouts of the deterministic
policy (mean of the Gaussian), and prints per-episode return + peak z.

Usage:
  poetry run python play.py                    # most recent run
  poetry run python play.py --run runs/<dir>   # specific run
  poetry run python play.py --episodes 20 --stochastic
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_hyrosphere import env as envlib
from jax_hyrosphere.physics import default_hyro_params, default_linear_params
from train import ActorCritic, normalize, init_running_stats, RunningStats


def latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "model.pkl").exists()]
    if not candidates:
        raise FileNotFoundError(f"No model.pkl found under {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--env", choices=["hyro", "linear"], default=None,
                   help="If not set, inferred from the checkpoint's saved args.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = args.run or latest_run(Path("runs"))
    with (run_dir / "model.pkl").open("rb") as f:
        ckpt = pickle.load(f)

    saved_args = ckpt["args"]
    env_kind = args.env or saved_args["env"]
    print(f"[play] run={run_dir}  env={env_kind}  deterministic={not args.stochastic}")

    params = default_hyro_params() if env_kind == "hyro" else default_linear_params()
    reset_fn, step_fn, obs_dim, act_dim = envlib.make_batched(env_kind, params, 1)

    model = ActorCritic(
        act_dim=act_dim,
        hidden=tuple(saved_args["hidden"]),
        log_std_init=saved_args["log_std_init"],
    )
    net_params = ckpt["params"]
    obs_stats = RunningStats(
        mean=jnp.asarray(ckpt["obs_stats"].mean),
        var=jnp.asarray(ckpt["obs_stats"].var),
        count=jnp.asarray(ckpt["obs_stats"].count),
    )

    @jax.jit
    def policy(obs, key):
        norm = normalize(obs, obs_stats)
        mean, log_std, _v = model.apply(net_params, norm)
        if args.stochastic:
            noise = jax.random.normal(key, mean.shape)
            return mean + jnp.exp(log_std) * noise
        return mean

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    env_state, obs = reset_fn(k)
    returns, peaks = [], []
    for ep in range(args.episodes):
        rng, k_reset = jax.random.split(rng)
        env_state, obs = reset_fn(k_reset)
        ep_ret = 0.0
        peak = float(env_state.phys.position[0, 2])
        for step in range(args.max_steps):
            rng, k = jax.random.split(rng)
            action = policy(obs, k)
            env_state, obs, r, done, _info = step_fn(env_state, action)
            ep_ret += float(r[0])
            peak = max(peak, float(env_state.phys.position[0, 2]))
            if bool(done[0]):
                break
        returns.append(ep_ret)
        peaks.append(peak)
        print(f"[play] ep {ep+1:2d}/{args.episodes}  return={ep_ret:+8.2f}  peak_z={peak:.3f}")
    if returns:
        print(f"\n[play] mean return  = {np.mean(returns):+.2f}  (std {np.std(returns):.2f})")
        print(f"[play] mean peak_z  = {np.mean(peaks):.3f}")


if __name__ == "__main__":
    main()
