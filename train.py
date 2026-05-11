"""Single-file JAX PPO training across the jax_playground env set.

PureJaxRL-style: env + rollout + GAE + minibatch updates all live inside a
single `jax.jit`-compiled function and run end-to-end on GPU. Throughput is
typically 50-200× a CPU + SubprocVecEnv stack at the same `n_envs`.

Usage:
  poetry run python train.py --env hyro     --updates 200 --n-envs 256
  poetry run python train.py --env linear   --updates 500 --n-envs 512
  poetry run python train.py --env pursuit  --updates 500 --n-envs 1024
  poetry run python train.py --env gathering --updates 500 --n-envs 1024

Checkpoints land in runs/<env>-<timestamp>/ as a pickled (params, opt_state,
norm_stats) tuple plus a `metrics.csv` of mean episode return.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_playground import envs as envlib
from jax_playground.policy import (
    ActorCritic, RunningStats, gaussian_entropy, gaussian_log_prob,
    init_running_stats, normalize, update_running_stats,
)


# --------------------------------------------------------------------------
# PPO transition + rollout (jit-scanned over time)
# --------------------------------------------------------------------------


class Transition(NamedTuple):
    obs: jnp.ndarray         # (T, N, obs_dim)
    action: jnp.ndarray      # (T, N, act_dim)
    log_prob: jnp.ndarray    # (T, N)
    value: jnp.ndarray       # (T, N)
    reward: jnp.ndarray      # (T, N)
    done: jnp.ndarray        # (T, N)


def make_train_step(env_kind: str, params, args):
    reset_batch, step_batch, obs_dim, act_dim = envlib.make_batched(
        env_kind, params, args.n_envs
    )
    model = ActorCritic(act_dim=act_dim, hidden=tuple(args.hidden),
                         log_std_init=args.log_std_init)

    def env_rollout(net_params, env_state, obs, rng, n_steps: int):
        """Roll n_steps in env, return transitions + final value bootstrap."""
        def body(carry, _):
            env_state, obs, rng = carry
            rng, k = jax.random.split(rng)
            mean, log_std, value = model.apply(net_params, obs)
            noise = jax.random.normal(k, mean.shape)
            action = mean + jnp.exp(log_std) * noise
            log_prob = gaussian_log_prob(action, mean, log_std)
            env_state, next_obs, reward, done, _info = step_batch(env_state, action)
            transition = Transition(
                obs=obs, action=action, log_prob=log_prob,
                value=value, reward=reward, done=done,
            )
            return (env_state, next_obs, rng), transition

        (env_state, last_obs, rng), traj = jax.lax.scan(
            body, (env_state, obs, rng), xs=None, length=n_steps
        )
        _, _, last_value = model.apply(net_params, last_obs)
        return traj, env_state, last_obs, last_value, rng

    def gae(rewards, values, dones, last_value, gamma=args.gamma, lam=args.gae_lambda):
        """Backward GAE scan. dones is per-step termination flag (after step)."""
        def body(carry, t):
            next_value, next_adv = carry
            value = values[t]
            reward = rewards[t]
            done = dones[t].astype(jnp.float32)
            delta = reward + gamma * next_value * (1.0 - done) - value
            adv = delta + gamma * lam * (1.0 - done) * next_adv
            return (value, adv), adv

        _, advs = jax.lax.scan(
            body,
            (last_value, jnp.zeros_like(last_value)),
            jnp.arange(rewards.shape[0]),
            reverse=True,
        )
        return advs

    def loss_fn(net_params, mb_obs, mb_action, mb_old_logp, mb_adv, mb_ret):
        mean, log_std, value = model.apply(net_params, mb_obs)
        new_logp = gaussian_log_prob(mb_action, mean, log_std)
        ratio = jnp.exp(new_logp - mb_old_logp)

        # Normalize advantages (per mini-batch)
        adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # Clipped surrogate
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
        pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        v_loss = 0.5 * jnp.mean((value - mb_ret) ** 2)
        ent = jnp.mean(gaussian_entropy(log_std))
        total = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

        # Diagnostics
        approx_kl = jnp.mean((ratio - 1.0) - (new_logp - mb_old_logp))
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32))
        return total, {"pg": pg_loss, "v": v_loss, "ent": ent,
                       "kl": approx_kl, "clip": clip_frac}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update_minibatch(opt_state, net_params, batch):
        (loss, metrics), grads = grad_fn(net_params, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)
        return opt_state, net_params, (loss, metrics)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr),
    )

    def train_update(carry, rng):
        """One PPO update cycle: collect rollout, compute GAE, n_epochs of SGD."""
        env_state, obs, net_params, opt_state, obs_stats, rng = carry

        # 1. collect (n_steps × n_envs)
        rng, k = jax.random.split(rng)
        traj, env_state, last_obs, last_value, _ = env_rollout(
            net_params, env_state, obs, k, args.n_steps
        )
        obs = last_obs

        # 2. update running obs stats from the entire rollout
        flat_obs = traj.obs.reshape(-1, obs_dim)
        obs_stats = update_running_stats(obs_stats, flat_obs)

        # 3. normalize observations using updated stats. We deliberately do
        # this AFTER collection so values used by the rollout aren't fudged;
        # the loss recomputes them anyway from the rollout's stored obs.
        norm_obs = normalize(traj.obs, obs_stats)
        # Also normalize last_obs for value bootstrap
        norm_last_obs = normalize(last_obs, obs_stats)
        _, _, norm_last_value = model.apply(net_params, norm_last_obs)

        # 4. compute GAE on values from the *normalized* policy (we recompute
        # values from norm_obs to stay consistent).
        _, _, norm_values = model.apply(net_params, norm_obs.reshape(-1, obs_dim))
        norm_values = norm_values.reshape(traj.reward.shape)
        adv = gae(traj.reward, norm_values, traj.done, norm_last_value)
        returns = adv + norm_values

        # Flatten time × env for minibatching
        b_obs = norm_obs.reshape(-1, obs_dim)
        b_action = traj.action.reshape(-1, act_dim)
        b_old_logp = traj.log_prob.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        n_samples = b_obs.shape[0]

        n_mb = n_samples // args.minibatch
        # Drop the tail so reshape is exact (n_mb · minibatch ≤ n_samples).
        n_truncated = n_mb * args.minibatch

        def _do_epoch(opt_state, net_params, key):
            perm = jax.random.permutation(key, n_samples)[:n_truncated]
            perm_mat = perm.reshape(n_mb, args.minibatch)
            def mb_body(carry, idx_row):
                os_, np_ = carry
                batch = (b_obs[idx_row], b_action[idx_row], b_old_logp[idx_row],
                         b_adv[idx_row], b_ret[idx_row])
                os_, np_, _ = update_minibatch(os_, np_, batch)
                return (os_, np_), 0
            (opt_state, net_params), _ = jax.lax.scan(
                mb_body, (opt_state, net_params), perm_mat
            )
            return opt_state, net_params

        def _skip_epoch(opt_state, net_params, _key):
            return opt_state, net_params

        def one_epoch(carry, key):
            opt_state, net_params, stopped, n_executed = carry
            # Run this epoch only if not yet stopped
            opt_state, net_params = jax.lax.cond(
                stopped, _skip_epoch, _do_epoch, opt_state, net_params, key
            )
            (_loss, metrics), _grads = grad_fn(
                net_params, b_obs, b_action, b_old_logp, b_adv, b_ret
            )
            # An epoch counts as executed if we didn't skip it.
            n_executed = jnp.where(stopped, n_executed, n_executed + 1)
            # Stop if KL exceeds target OR has gone non-finite (NaN comparison
            # returns False, which would silently keep training a corrupted net).
            new_stopped = (
                stopped
                | (metrics["kl"] > args.target_kl)
                | (~jnp.isfinite(metrics["kl"]))
            )
            return (opt_state, net_params, new_stopped, n_executed), metrics["kl"]

        rng, *epoch_keys = jax.random.split(rng, args.n_epochs + 1)
        (opt_state, net_params, _stopped, n_executed), _epoch_kls = jax.lax.scan(
            one_epoch,
            (opt_state, net_params, jnp.bool_(False), jnp.int32(0)),
            jnp.stack(epoch_keys),
        )

        # 5. final diagnostic pass on the full batch for logging
        (_loss, metrics), _grads = grad_fn(net_params, b_obs, b_action, b_old_logp, b_adv, b_ret)

        mean_reward = traj.reward.mean()
        mean_done = traj.done.mean()
        carry = (env_state, obs, net_params, opt_state, obs_stats, rng)
        scalars = {
            "rew_mean": mean_reward,
            "done_frac": mean_done,
            "epochs_used": n_executed,
            **metrics,
        }
        return carry, scalars

    return train_update, model, optimizer, obs_dim, act_dim, reset_batch


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=envlib.list_envs(), default="hyro")
    p.add_argument("--updates", type=int, default=200,
                   help="Number of PPO update cycles to run.")
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=128,
                   help="Rollout length per env per update. n_envs*n_steps = batch size.")
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="Skip remaining epochs in a rollout once approx-KL "
                        "exceeds this (SB3-style early stop).")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--log-std-init", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    params = envlib.REGISTRY[args.env]["default_params"]()

    run_name = args.name or f"{args.env}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] env={args.env}  n_envs={args.n_envs}  n_steps={args.n_steps}  "
          f"updates={args.updates}  total_steps={args.n_envs*args.n_steps*args.updates:,}")
    print(f"[train] run_dir={run_dir}")
    print(f"[train] jax devices: {jax.devices()}")

    train_update, model, optimizer, obs_dim, act_dim, reset_batch = make_train_step(
        args.env, params, args
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, k_reset, k_init = jax.random.split(rng, 3)
    env_state, obs = reset_batch(k_reset)
    net_params = model.init(k_init, jnp.zeros((1, obs_dim)))
    opt_state = optimizer.init(net_params)
    obs_stats = init_running_stats(obs_dim)

    # JIT the update step once. Inputs include the per-update RNG so signature is
    # (carry, rng) -> (carry, scalars).
    jit_update = jax.jit(train_update)

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "elapsed_s", "rew_mean", "kl", "clip", "ent", "v"])

        carry = (env_state, obs, net_params, opt_state, obs_stats, rng)
        t0 = time.time()
        for update in range(args.updates):
            rng_in = jax.random.fold_in(jax.random.PRNGKey(args.seed), update + 1)
            carry, scalars = jit_update(carry, rng_in)
            # Force materialization (otherwise prints will lag behind compute)
            scalars = jax.tree_util.tree_map(lambda x: float(x), scalars)
            elapsed = time.time() - t0
            steps = (update + 1) * args.n_envs * args.n_steps
            print(f"[u {update+1:4d}/{args.updates}] "
                  f"steps={steps:>10,}  "
                  f"rew={scalars['rew_mean']:+.3f}  "
                  f"kl={scalars['kl']:+.4f}  "
                  f"clip={scalars['clip']:.3f}  "
                  f"ent={scalars['ent']:+.3f}  "
                  f"v_loss={scalars['v']:.4f}  "
                  f"ep={int(scalars['epochs_used'])}/{args.n_epochs}  "
                  f"t={elapsed:.0f}s  "
                  f"sps={steps/max(elapsed,1e-9):,.0f}")
            writer.writerow([update + 1, f"{elapsed:.1f}",
                             scalars["rew_mean"], scalars["kl"],
                             scalars["clip"], scalars["ent"], scalars["v"]])
            f.flush()

    # Save final params + obs_stats
    env_state, obs, net_params, opt_state, obs_stats, rng = carry
    ckpt = {
        "args": vars(args),
        "params": jax.device_get(net_params),
        "obs_stats": jax.device_get(obs_stats),
    }
    with (run_dir / "model.pkl").open("wb") as f:
        pickle.dump(ckpt, f)
    print(f"[train] saved → {run_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
