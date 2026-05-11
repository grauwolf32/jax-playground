"""Self-play PPO for the pursuit_selfplay env.

Trains ONE role (--role) against a frozen opponent. The opponent comes from
--opp-model — either a pursuit-v0 checkpoint (initial bootstrap) or a prior
self-play checkpoint. If --opp-model is omitted the opponent is stationary
(zero-action), which is a sane bootstrap for cycle 0.

Typical workflow:

    # cycle 0: train pursuer against the bootstrap evader from pursuit-v0
    poetry run python train_selfplay.py --role pursuer \
        --opp-model runs/pursuit-lidar/model.pkl \
        --updates 1000 --name sp-pursuer-0

    # cycle 1: train evader against the cycle-0 pursuer
    poetry run python train_selfplay.py --role evader \
        --opp-model runs/sp-pursuer-0/model.pkl \
        --updates 1000 --name sp-evader-1

    # cycle 2: train pursuer against cycle-1 evader … alternate forever.

Checkpoint format is the same as train.py: {args, params, obs_stats}, so the
output of any cycle is directly usable as the next cycle's --opp-model.
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
import optax

from jax_playground.envs.pursuit_selfplay import env as sp_env
from jax_playground.envs.pursuit.vehicle import default_params as default_pursuit_params
from jax_playground.policy import (
    ActorCritic, RunningStats, gaussian_entropy, gaussian_log_prob,
    init_running_stats, normalize,
)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def _load_opponent(path: str | None, obs_dim: int, act_dim: int, args):
    """Return (opp_params, opp_obs_stats, is_stationary).

    If path is None, returns dummy params (won't be used) and the flag
    is_stationary=True — the rollout will skip the opponent forward pass and
    emit a zero action.
    """
    if path is None:
        model = ActorCritic(act_dim=act_dim, hidden=tuple(args.hidden))
        dummy = model.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))
        return dummy, init_running_stats(obs_dim), True
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    opp_params = ckpt["params"]
    opp_stats = ckpt["obs_stats"]
    # Re-wrap so the loaded stats live on-device.
    opp_stats = RunningStats(
        mean=jnp.asarray(opp_stats.mean),
        var=jnp.asarray(opp_stats.var),
        count=jnp.asarray(opp_stats.count),
    )
    return opp_params, opp_stats, False


def make_train_step(params, args, role: str,
                     opp_params, opp_obs_stats, opp_stationary: bool):
    obs_dim, act_dim = sp_env.OBS_DIM, sp_env.ACT_DIM

    @jax.jit
    def reset_batch(key):
        keys = jax.random.split(key, args.n_envs)
        return jax.vmap(lambda k: sp_env.env_reset(k, params))(keys)

    @jax.jit
    def step_batch(states, evader_actions, pursuer_actions):
        return jax.vmap(
            lambda s, ae, ap: sp_env.env_step(s, ae, ap, params)
        )(states, evader_actions, pursuer_actions)

    model = ActorCritic(act_dim=act_dim, hidden=tuple(args.hidden),
                         log_std_init=args.log_std_init)

    role_is_evader = (role == "evader")

    def _opp_action(opp_obs, key):
        if opp_stationary:
            return jnp.zeros((opp_obs.shape[0], act_dim), dtype=jnp.float32)
        norm = normalize(opp_obs, opp_obs_stats)
        mean, log_std, _ = model.apply(opp_params, norm)
        noise = jax.random.normal(key, mean.shape)
        return mean + jnp.exp(log_std) * noise

    def env_rollout(net_params, env_state,
                    raw_e_obs, raw_p_obs, obs_stats, rng, n_steps: int):
        def body(carry, _):
            env_state, raw_e, raw_p, rng = carry
            rng, k_self, k_opp = jax.random.split(rng, 3)

            self_obs_raw = raw_e if role_is_evader else raw_p
            opp_obs_raw = raw_p if role_is_evader else raw_e

            # Trained side (self).
            norm_self = normalize(self_obs_raw, obs_stats)
            s_mean, s_log_std, s_value = model.apply(net_params, norm_self)
            s_noise = jax.random.normal(k_self, s_mean.shape)
            self_action = s_mean + jnp.exp(s_log_std) * s_noise
            self_log_prob = gaussian_log_prob(self_action, s_mean, s_log_std)

            # Frozen opponent.
            opp_action = _opp_action(opp_obs_raw, k_opp)

            evader_action = self_action if role_is_evader else opp_action
            pursuer_action = opp_action if role_is_evader else self_action

            env_state, next_e_obs, next_p_obs, evader_r, done, _info = step_batch(
                env_state, evader_action, pursuer_action
            )

            # Zero-sum: pursuer reward = -evader reward.
            self_reward = evader_r if role_is_evader else -evader_r

            transition = Transition(
                obs=norm_self, action=self_action, log_prob=self_log_prob,
                value=s_value, reward=self_reward, done=done,
            )
            return (env_state, next_e_obs, next_p_obs, rng), transition

        (env_state, last_e, last_p, rng), traj = jax.lax.scan(
            body, (env_state, raw_e_obs, raw_p_obs, rng), xs=None, length=n_steps
        )
        last_self = last_e if role_is_evader else last_p
        norm_last = normalize(last_self, obs_stats)
        _, _, last_value = model.apply(net_params, norm_last)
        return traj, env_state, last_e, last_p, last_value, rng

    def gae(rewards, values, dones, last_value,
            gamma=args.gamma, lam=args.gae_lambda):
        def body(carry, t):
            next_value, next_adv = carry
            value = values[t]
            reward = rewards[t]
            done = dones[t].astype(jnp.float32)
            delta = reward + gamma * next_value * (1.0 - done) - value
            adv = delta + gamma * lam * (1.0 - done) * next_adv
            return (value, adv), adv
        _, advs = jax.lax.scan(
            body, (last_value, jnp.zeros_like(last_value)),
            jnp.arange(rewards.shape[0]), reverse=True,
        )
        return advs

    def loss_fn(net_params, mb_obs, mb_action, mb_old_logp, mb_adv, mb_ret):
        mean, log_std, value = model.apply(net_params, mb_obs)
        new_logp = gaussian_log_prob(mb_action, mean, log_std)
        ratio = jnp.exp(new_logp - mb_old_logp)
        adv = (mb_adv - mb_adv.mean()) / jnp.maximum(mb_adv.std(), 1e-3)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
        pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        v_loss = 0.5 * jnp.mean((value - mb_ret) ** 2)
        ent = jnp.mean(gaussian_entropy(log_std))
        total = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent
        approx_kl = jnp.mean((ratio - 1.0) - (new_logp - mb_old_logp))
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32))
        return total, {"pg": pg_loss, "v": v_loss, "ent": ent,
                       "kl": approx_kl, "clip": clip_frac}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr),
    )

    def update_minibatch(opt_state, net_params, batch):
        (loss, metrics), grads = grad_fn(net_params, *batch)
        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)
        return opt_state, net_params, (loss, metrics)

    def train_update(carry, rng):
        env_state, raw_e_obs, raw_p_obs, net_params, opt_state, obs_stats, rng = carry

        rng, k = jax.random.split(rng)
        traj, env_state, last_e, last_p, last_value, _ = env_rollout(
            net_params, env_state, raw_e_obs, raw_p_obs, obs_stats, k, args.n_steps
        )
        raw_e_obs = last_e
        raw_p_obs = last_p

        adv = gae(traj.reward, traj.value, traj.done, last_value)
        returns = adv + traj.value

        b_obs = traj.obs.reshape(-1, obs_dim)
        b_action = traj.action.reshape(-1, act_dim)
        b_old_logp = traj.log_prob.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        n_samples = b_obs.shape[0]
        n_mb = n_samples // args.minibatch
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
            opt_state, net_params = jax.lax.cond(
                stopped, _skip_epoch, _do_epoch, opt_state, net_params, key
            )
            (_loss, metrics), _grads = grad_fn(
                net_params, b_obs, b_action, b_old_logp, b_adv, b_ret
            )
            n_executed = jnp.where(stopped, n_executed, n_executed + 1)
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

        (_loss, metrics), _grads = grad_fn(
            net_params, b_obs, b_action, b_old_logp, b_adv, b_ret
        )

        mean_reward = traj.reward.mean()
        mean_done = traj.done.mean()
        carry = (env_state, raw_e_obs, raw_p_obs, net_params, opt_state, obs_stats, rng)
        scalars = {
            "rew_mean": mean_reward,
            "done_frac": mean_done,
            "epochs_used": n_executed,
            **metrics,
        }
        return carry, scalars

    return train_update, model, optimizer, obs_dim, act_dim, reset_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["evader", "pursuer"], required=True)
    p.add_argument("--opp-model", type=str, default=None,
                   help="Path to opponent's model.pkl (pursuit-v0 or prior "
                        "self-play). If omitted, opponent emits zero actions.")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--log-std-init", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    params = default_pursuit_params()

    run_name = args.name or f"selfplay-{args.role}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    obs_dim, act_dim = sp_env.OBS_DIM, sp_env.ACT_DIM

    opp_params, opp_obs_stats, opp_stationary = _load_opponent(
        args.opp_model, obs_dim, act_dim, args
    )
    if opp_stationary:
        print(f"[selfplay] no --opp-model; opponent is STATIONARY (zero action)")
    else:
        print(f"[selfplay] opponent loaded from {args.opp_model}")

    print(f"[selfplay] role={args.role}  n_envs={args.n_envs}  n_steps={args.n_steps}  "
          f"updates={args.updates}  total_steps={args.n_envs*args.n_steps*args.updates:,}")
    print(f"[selfplay] run_dir={run_dir}")
    print(f"[selfplay] jax devices: {jax.devices()}")

    train_update, model, optimizer, obs_dim, act_dim, reset_batch = make_train_step(
        params, args, args.role, opp_params, opp_obs_stats, opp_stationary
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, k_reset, k_init = jax.random.split(rng, 3)
    env_state, raw_e_obs, raw_p_obs = reset_batch(k_reset)
    net_params = model.init(k_init, jnp.zeros((1, obs_dim)))
    opt_state = optimizer.init(net_params)
    obs_stats = init_running_stats(obs_dim)

    jit_update = jax.jit(train_update)

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update", "elapsed_s", "rew_mean", "kl", "clip", "ent", "v"])

        carry = (env_state, raw_e_obs, raw_p_obs, net_params, opt_state, obs_stats, rng)
        t0 = time.time()
        for update in range(args.updates):
            rng_in = jax.random.fold_in(jax.random.PRNGKey(args.seed), update + 1)
            carry, scalars = jit_update(carry, rng_in)
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

    env_state, raw_e_obs, raw_p_obs, net_params, opt_state, obs_stats, rng = carry
    ckpt = {
        "args": vars(args),
        "params": jax.device_get(net_params),
        "obs_stats": jax.device_get(obs_stats),
        "role": args.role,
    }
    with (run_dir / "model.pkl").open("wb") as f:
        pickle.dump(ckpt, f)
    print(f"[selfplay] saved → {run_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
