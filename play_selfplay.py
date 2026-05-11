"""Headless evaluation of a self-play checkpoint.

Loads runs/<run>/model.pkl from train_selfplay.py, runs episodes of the
trained role against the saved --opp-model (or a path you override), and
prints per-episode return + caught rate + mean episode length.

Usage:
  poetry run python play_selfplay.py --run runs/sp-pursuer-0
  poetry run python play_selfplay.py --run runs/sp-pursuer-0 --opp-model runs/pursuit-lidar/model.pkl
  poetry run python play_selfplay.py --run runs/sp-pursuer-0 --episodes 20 --stochastic
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_playground.envs.pursuit_selfplay import env as sp_env
from jax_playground.envs.pursuit.vehicle import default_params as default_pursuit_params
from jax_playground.policy import ActorCritic, RunningStats, normalize


def latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "model.pkl").exists()]
    if not candidates:
        raise FileNotFoundError(f"No model.pkl found under {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_ckpt_policy(ckpt_path: Path, obs_dim: int, act_dim: int):
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    saved_args = ckpt["args"]
    model = ActorCritic(
        act_dim=act_dim,
        hidden=tuple(saved_args.get("hidden", [256, 256])),
        log_std_init=saved_args.get("log_std_init", 0.0),
    )
    obs_stats = ckpt["obs_stats"]
    obs_stats = RunningStats(
        mean=jnp.asarray(obs_stats.mean),
        var=jnp.asarray(obs_stats.var),
        count=jnp.asarray(obs_stats.count),
    )
    return model, ckpt["params"], obs_stats, saved_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--opp-model", type=str, default=None,
                   help="Override opponent. Defaults to the path saved in the run's args.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = args.run or latest_run(Path("runs"))
    params = default_pursuit_params()
    obs_dim, act_dim = sp_env.OBS_DIM, sp_env.ACT_DIM

    # Trained role
    self_model, self_params, self_stats, self_args = _load_ckpt_policy(
        run_dir / "model.pkl", obs_dim, act_dim
    )
    role = self_args.get("role", "pursuer")
    role_is_evader = (role == "evader")

    # Opponent
    opp_path = args.opp_model or self_args.get("opp_model")
    if opp_path is None:
        print(f"[play_sp] no opponent — stationary (zero-action)")
        opp_model, opp_params, opp_stats = None, None, None
    else:
        print(f"[play_sp] opponent ← {opp_path}")
        opp_model, opp_params, opp_stats, _ = _load_ckpt_policy(
            Path(opp_path), obs_dim, act_dim
        )

    print(f"[play_sp] run={run_dir}  role={role}  deterministic={not args.stochastic}")

    reset_one = jax.jit(lambda k: sp_env.env_reset(k, params))
    step_one = jax.jit(lambda s, ae, ap: sp_env.env_step(s, ae, ap, params))

    @jax.jit
    def policy(model_params, stats, model, obs, key):
        norm = normalize(obs, stats)
        mean, log_std, _v = model.apply(model_params, norm)
        if args.stochastic:
            noise = jax.random.normal(key, mean.shape)
            return mean + jnp.exp(log_std) * noise
        return mean

    # JIT closures for self and opp (model is static, captured).
    @jax.jit
    def self_act(obs, key):
        norm = normalize(obs, self_stats)
        mean, log_std, _ = self_model.apply(self_params, norm)
        if args.stochastic:
            noise = jax.random.normal(key, mean.shape)
            return mean + jnp.exp(log_std) * noise
        return mean

    if opp_model is None:
        @jax.jit
        def opp_act(obs, key):
            return jnp.zeros((act_dim,), dtype=jnp.float32)
    else:
        @jax.jit
        def opp_act(obs, key):
            norm = normalize(obs, opp_stats)
            mean, log_std, _ = opp_model.apply(opp_params, norm)
            if args.stochastic:
                noise = jax.random.normal(key, mean.shape)
                return mean + jnp.exp(log_std) * noise
            return mean

    rng = jax.random.PRNGKey(args.seed)
    returns, ep_lens, caught_eps, min_d_eps = [], [], [], []
    for ep in range(args.episodes):
        rng, k_reset = jax.random.split(rng)
        env_state, evader_obs, pursuer_obs = reset_one(k_reset)
        ep_ret = 0.0
        ep_len = 0
        caught_this_ep = False
        min_d = float("inf")
        for step in range(args.max_steps):
            rng, k_self, k_opp = jax.random.split(rng, 3)
            self_obs = evader_obs if role_is_evader else pursuer_obs
            opp_obs = pursuer_obs if role_is_evader else evader_obs
            self_a = self_act(self_obs, k_self)
            opp_a = opp_act(opp_obs, k_opp)
            evader_a = self_a if role_is_evader else opp_a
            pursuer_a = opp_a if role_is_evader else self_a
            env_state, evader_obs, pursuer_obs, evader_r, done, info = step_one(
                env_state, evader_a, pursuer_a
            )
            self_r = float(evader_r) if role_is_evader else -float(evader_r)
            ep_ret += self_r
            d = float(info["distance"])
            min_d = min(min_d, d)
            ep_len += 1
            if bool(info["caught"]):
                caught_this_ep = True
            if bool(done):
                break
        returns.append(ep_ret)
        ep_lens.append(ep_len)
        caught_eps.append(int(caught_this_ep))
        min_d_eps.append(min_d)
        print(f"[play_sp] ep {ep+1:2d}/{args.episodes}  return={ep_ret:+8.2f}  "
              f"len={ep_len:4d}  caught={'Y' if caught_this_ep else 'N'}  min_d={min_d:.1f}")

    if returns:
        print(f"\n[play_sp] mean return = {np.mean(returns):+.2f}  (std {np.std(returns):.2f})")
        print(f"[play_sp] mean ep len = {np.mean(ep_lens):.1f}")
        print(f"[play_sp] catch rate  = {np.mean(caught_eps):.2%}")
        print(f"[play_sp] mean min_d  = {np.mean(min_d_eps):.1f}")


if __name__ == "__main__":
    main()
