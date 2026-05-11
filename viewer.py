"""Interactive OpenGL viewer for the JAX HyroSphere / LinearSphere physics.

Reuses the legacy `viz.py` (ported from ../openai-physics) for the actual
drawing. A small `_Adapter` class exposes JAX state through the
`.position`, `.velocity`, `.U`, `.phi`, etc. attributes the legacy
draw functions expect.

If `--run` points at a trained model.pkl, the policy drives the env;
otherwise you control the wheels/sliders with the keyboard.

Controls:
  Mouse drag (LMB)   orbit camera
  Scroll wheel       zoom
  Space              pause / resume
  Backspace          reset env
  Tab                toggle HyroSphere / LinearSphere
  Q/W/E/R(/T/Y)      hold for positive accel on mass 0..n
  A/S/D/F(/G/H)      hold for negative accel on mass 0..n
  1/2/3/4/5/6        toggle COM / axes / contact / HUD / trail / mini-plot
  ? or /             toggle help
  Esc                quit

Usage:
  poetry run python viewer.py                          # manual control, hyro
  poetry run python viewer.py --env linear             # manual, linear
  poetry run python viewer.py --run runs/big-4096      # drive with a saved policy
"""

from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame as pg
from pygame.locals import (
    DOUBLEBUF, OPENGL, KEYDOWN,
    K_ESCAPE, K_SPACE, K_TAB, K_BACKSPACE, K_QUESTION, K_SLASH,
    K_q, K_w, K_e, K_r, K_t, K_y,
    K_a, K_s, K_d, K_f, K_g, K_h,
    K_1, K_2, K_3, K_4, K_5, K_6,
    MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION,
)
from OpenGL.GL import (
    glClearColor, glClear, glEnable, glDisable, glPushMatrix, glPopMatrix,
    glTranslatef,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
    GL_LINE_SMOOTH, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST,
)
from OpenGL.GLU import gluPerspective
from OpenGL.GLUT import (
    glutInit,
    GLUT_BITMAP_9_BY_15, GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_HELVETICA_12,
)

from jax_playground import envs as envlib
from jax_playground.envs.hyrosphere.physics import (
    HyroParams, HyroState, default_params as default_hyro_params,
    reset as hyro_reset, step as hyro_step, wheel_geometry as _hyro_wheel_geometry,
)
from jax_playground.envs.linearsphere.physics import (
    LinearParams, LinearState, default_params as default_linear_params,
    reset as linear_reset, step as linear_step,
)
from jax_playground.viz3d import (
    OrbitCamera, drawGroundGrid, drawCOMMarker, drawBodyAxes,
    drawOmegaArrow, drawContactAndFriction, drawText2D,
    drawHyrosphere, drawLinearsphere, drawTrajectory, drawMiniPlot,
    setupLighting,
)
# Module-level so pickle finds RunningStats/ActorCritic in viewer's __main__.
from jax_playground.policy import ActorCritic, RunningStats, normalize  # noqa: E402


WIN_W, WIN_H = 1280, 800
DT = 0.01
ACTION_MAG = 3.0
TRAIL_LEN = 1500
PLOT_LEN = 500
POS_KEYS = (K_q, K_w, K_e, K_r, K_t, K_y)
NEG_KEYS = (K_a, K_s, K_d, K_f, K_g, K_h)
HELP_LINES = [
    "Mouse drag (LMB):  orbit camera",
    "Mouse scroll:      zoom",
    "Space:             pause / resume",
    "Backspace:         reset",
    "Tab:               switch env",
    "QWER(TY) hold:     +accel on mass 0..n",
    "ASDF(GH) hold:     -accel on mass 0..n",
    "1..5 toggle overlays (5=trail)",
    "6:                 toggle mini-plot",
    "? :                toggle this help",
    "Esc:               quit",
]


# --------------------------------------------------------------------------
# JAX → numpy attribute adapters for the legacy draw functions
# --------------------------------------------------------------------------


class _HyroAdapter:
    """Look-alike for the numpy HyroSphere object — exposes the attributes
    drawHyrosphere() etc. read."""

    def __init__(self, state: HyroState, params: HyroParams):
        self.U = np.asarray(state.U)
        self.phi = np.asarray(state.phi)
        self.omega = np.asarray(state.omega)
        self.ksi = np.asarray(state.ksi)
        self.position = np.asarray(state.position)
        self.velocity = np.asarray(state.velocity)
        self.Omega = np.asarray(state.Omega)
        self.dOmegadt = np.asarray(state.dOmegadt)
        self.t_len = float(params.t_len)
        self.radius = float(params.radius)
        self.mass = float(params.mass)
        self.dot_masses = np.asarray(params.dot_masses)
        self.last_F_fric = np.asarray(state.last_F_fric)
        self.last_in_contact = bool(state.in_contact)


class _LinearAdapter:
    def __init__(self, state: LinearState, params: LinearParams):
        self.U = np.asarray(state.U)
        self.shifts = np.asarray(state.shifts)
        self.speeds = np.asarray(state.speeds)
        self.accelerations = np.asarray(state.accelerations)
        self.position = np.asarray(state.position)
        self.velocity = np.asarray(state.velocity)
        self.Omega = np.asarray(state.Omega)
        self.dOmegadt = np.asarray(state.dOmegadt)
        self.radius = float(params.radius)
        self.mass = float(params.mass)
        self.dot_masses = np.asarray(params.dot_masses)
        self.last_F_fric = np.asarray(state.last_F_fric)
        self.last_in_contact = bool(state.in_contact)


def _com_offset(adapter, R: np.ndarray) -> np.ndarray:
    total = adapter.mass + float(adapter.dot_masses.sum())
    offset = (adapter.dot_masses[:, None] * R).sum(axis=0) / total
    return adapter.position + offset


def _hyro_R(adapter: _HyroAdapter) -> np.ndarray:
    """Reconstruct dot-mass positions relative to O — same geometry as viz."""
    from jax_hyrosphere.viz import rotate_vec
    A = adapter.U * np.sqrt(1.0 / 8.0)
    b = [np.cross(A[i], A[(i + 1) % 4]) for i in range(4)]
    b = [bi / np.linalg.norm(bi) * adapter.t_len / 2.0 for bi in b]
    b = [np.dot(rotate_vec(adapter.U[i], adapter.phi[i]), b[i]) for i in range(4)]
    return A + np.array(b)


# --------------------------------------------------------------------------
# Trained-policy bridge
# --------------------------------------------------------------------------


def load_policy(run_dir: Path, env_kind_hint: str | None):
    """Return (env_kind, action_fn) where action_fn(obs) -> action np.ndarray.

    If `run_dir` is None, returns (env_kind_hint or "hyro", None) — no policy.
    """
    if run_dir is None:
        return env_kind_hint or "hyro", None
    with (run_dir / "model.pkl").open("rb") as f:
        ckpt = pickle.load(f)
    saved_args = ckpt["args"]
    env_kind = env_kind_hint or saved_args["env"]
    from jax_playground import envs as envlib
    obs_dim = envlib.REGISTRY[env_kind]["obs_dim"]
    act_dim = envlib.REGISTRY[env_kind]["act_dim"]

    # Build a fresh ActorCritic with the saved arch, run a deterministic forward.
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
    def _act(obs):
        norm = normalize(obs, obs_stats)
        mean, _ls, _v = model.apply(net_params, norm)
        return mean

    def action_fn(obs_np: np.ndarray) -> np.ndarray:
        a = _act(jnp.asarray(obs_np))
        return np.asarray(a)

    return env_kind, action_fn


# --------------------------------------------------------------------------
# Per-env step helpers (jit'd, no batch dim — adapter handles host transfer)
# --------------------------------------------------------------------------


_hyro_step_j = jax.jit(hyro_step)
_linear_step_j = jax.jit(linear_step)


def build_state(env_kind: str, params, rng_seed: int):
    key = jax.random.PRNGKey(rng_seed)
    if env_kind == "hyro":
        return hyro_reset(key, params)
    return linear_reset(key, params)


# Observation built locally for play (must match env.py's _hyro_obs / _linear_obs).
def hyro_obs_np(state: HyroState, params: HyroParams) -> np.ndarray:
    R, _ = _hyro_wheel_geometry(state, params)
    omega_total = state.omega[:, None] * state.U + state.Omega[None, :]
    dRdt = jnp.cross(omega_total, R, axis=1)
    height = state.position[2] - params.radius
    in_contact = jnp.where(state.in_contact, 1.0, 0.0)
    obs = jnp.concatenate([
        state.velocity, state.omega, state.ksi, state.Omega, state.dOmegadt,
        R.reshape(-1), dRdt.reshape(-1),
        jnp.array([height]), jnp.array([in_contact]),
    ])
    return np.asarray(obs)


def linear_obs_np(state: LinearState, params: LinearParams) -> np.ndarray:
    R = state.shifts[:, None] * state.U
    dRdt = jnp.cross(state.Omega[None, :], R, axis=1) + state.speeds[:, None] * state.U
    height = state.position[2] - params.radius
    in_contact = jnp.where(state.in_contact, 1.0, 0.0)
    obs = jnp.concatenate([
        state.velocity, state.speeds, state.accelerations, state.shifts,
        state.Omega, state.dOmegadt,
        R.reshape(-1), dRdt.reshape(-1),
        jnp.array([height]), jnp.array([in_contact]),
    ])
    return np.asarray(obs)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def keyboard_action(env_kind: str, keys):
    n = 4 if env_kind == "hyro" else 6
    a = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if keys[POS_KEYS[i]]:
            a[i] += ACTION_MAG
        if keys[NEG_KEYS[i]]:
            a[i] -= ACTION_MAG
    return a


def init_gl():
    pg.init()
    pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
    pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 4)
    pg.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
    pg.display.set_caption("jax-hyrosphere viewer")
    glutInit([])
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    setupLighting()
    gluPerspective(45.0, WIN_W / WIN_H, 0.05, 200.0)


# --------------------------------------------------------------------------
# 2D pygame viewer (pursuit, gathering)
# --------------------------------------------------------------------------


# Keyboard control mapping for 2D vehicles. WASD → (alpha, beta) ∈ [-1, 1]².
# A/D rotate visually CCW/CW (i.e. decrease/increase phi since +y is down in
# screen coords). beta is scaled down so held-key turning is smooth at the
# 30-fps viewer tick (the physics has dt=0.5 → 1 ray of turn per step at full
# beta = 57°/step is jarring to drive).
TURN_RATE = 0.3   # fraction of max_dw applied per held key
def _keyboard_action_2d(keys) -> np.ndarray:
    from pygame.locals import K_w, K_s, K_a, K_d
    alpha = (1.0 if keys[K_w] else 0.0) - (1.0 if keys[K_s] else 0.0)
    beta = ((1.0 if keys[K_d] else 0.0) - (1.0 if keys[K_a] else 0.0)) * TURN_RATE
    return np.array([alpha, beta], dtype=np.float32)


def run_2d_viewer(env_kind: str, action_fn, args) -> None:
    """Pygame 2D viewer for pursuit + gathering envs.

    Layout:
      Window size = world size (960×720). Anti-aliased vehicle arrows,
      fading per-agent trails, translucent catch-zone for pursuit, halos
      around targets for gathering, on-screen HUD.

    Drives the env with either the trained policy (if --run given) or
    keyboard (WASD = thrust + steering).
    """
    import pygame as pg
    from collections import deque

    from jax_playground import envs as envlib
    from jax_playground.render2d import (
        Renderer2D, PURSUER, EVADER, AGENT, TARGET,
        CATCH_ZONE, TRAIL_PURSUER, TRAIL_EVADER, TEXT,
    )
    from jax_playground.envs.pursuit.vehicle import default_params as default_v_params

    params = default_v_params()
    reset_b, step_b, obs_dim, act_dim = envlib.make_batched(env_kind, params, 1)

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    env_state, obs = reset_b(k)

    world_w = int(params.world_w)
    world_h = int(params.world_h)
    title = f"{env_kind}-v0"
    # If the world is too tall to fit on a typical 1080p monitor, render at
    # a smaller window size. The render2d helpers draw in world pixels, so we
    # downscale by setting up a smaller window and blitting a scaled surface.
    max_window_h = 900
    if world_h > max_window_h:
        scale = max_window_h / float(world_h)
        win_w = int(world_w * scale)
        win_h = max_window_h
    else:
        scale = 1.0
        win_w, win_h = world_w, world_h
    renderer = Renderer2D(world_w, world_h, title, win_w=win_w, win_h=win_h)

    trails = {
        "evader": deque(maxlen=300),
        "pursuer": deque(maxlen=300),
        "agent": deque(maxlen=300),
    }
    paused = False
    step_count = 0
    clock = pg.time.Clock()
    running = True
    help_lines = (
        ["WASD       drive (thrust + steer)"] if action_fn is None else
        ["policy from --run is driving"]
    ) + [
        "Space      pause",
        "Backspace  reset",
        "Esc        quit",
    ]

    while running:
        # ---- input ------------------------------------------------------
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE or ev.key == pg.K_q:
                    running = False
                elif ev.key == pg.K_SPACE:
                    paused = not paused
                elif ev.key == pg.K_BACKSPACE:
                    rng, k = jax.random.split(rng)
                    env_state, obs = reset_b(k)
                    step_count = 0
                    for d in trails.values():
                        d.clear()

        if not paused:
            # Action
            if action_fn is not None:
                action = action_fn(np.asarray(obs[0]))[None, :]
                action = jnp.asarray(action)
            else:
                a = _keyboard_action_2d(pg.key.get_pressed())
                action = jnp.asarray(a[None, :])
            env_state, obs, r, done, info = step_b(env_state, action)
            step_count += 1

        # ---- read state for rendering -----------------------------------
        if env_kind == "pursuit":
            evader = np.asarray(env_state.evader[0])
            pursuer = np.asarray(env_state.pursuer[0])
            trails["evader"].append((float(evader[0]), float(evader[1])))
            trails["pursuer"].append((float(pursuer[0]), float(pursuer[1])))
        else:  # gathering
            agent = np.asarray(env_state.agent[0])
            target_1 = np.asarray(env_state.target_1[0])
            target_2 = np.asarray(env_state.target_2[0])
            score = float(env_state.score[0])
            trails["agent"].append((float(agent[0]), float(agent[1])))

        # ---- draw -------------------------------------------------------
        renderer.clear()
        renderer.draw_grid()
        renderer.draw_border()
        renderer.draw_obstacles(np.asarray(params.obstacles))

        if env_kind == "pursuit":
            from jax_playground.envs.pursuit.env import _CATCH_RADIUS
            renderer.draw_catch_zone(float(pursuer[0]), float(pursuer[1]),
                                      float(_CATCH_RADIUS))
            renderer.draw_trail(list(trails["pursuer"]), TRAIL_PURSUER)
            renderer.draw_trail(list(trails["evader"]), TRAIL_EVADER)
            renderer.draw_vehicle(float(pursuer[0]), float(pursuer[1]),
                                   float(pursuer[6]), PURSUER)
            renderer.draw_vehicle(float(evader[0]), float(evader[1]),
                                   float(evader[6]), EVADER)

            d = float(np.hypot(pursuer[0] - evader[0], pursuer[1] - evader[1]))
            renderer.draw_text("pursuit-v0", 12, 10, big=True)
            renderer.draw_hud_block([
                f"t      {step_count * params.dt:6.2f} s   (step {step_count})",
                f"d      {d:7.2f}   (catch ≤ 80)",
                f"pursuer v = {float(np.hypot(pursuer[2], pursuer[3])):5.2f}",
                f"evader  v = {float(np.hypot(evader[2], evader[3])):5.2f}",
                f"reward = {float(r[0]):+.4f}",
            ], 12, 36)
            renderer.draw_text("pursuer", 12, world_h - 38, color=PURSUER, big=True)
            label = "evader (policy)" if action_fn else "evader (keyboard)"
            renderer.draw_text(label, 12, world_h - 20, color=EVADER)
        else:  # gathering
            from jax_playground.envs.gathering.env import _TARGET_RADIUS
            for t in (target_1, target_2):
                renderer.draw_target(float(t[0]), float(t[1]),
                                      float(_TARGET_RADIUS), color=TARGET)
            renderer.draw_trail(list(trails["agent"]), TRAIL_EVADER)
            renderer.draw_vehicle(float(agent[0]), float(agent[1]),
                                   float(agent[6]), AGENT)
            renderer.draw_text("gathering-v0", 12, 10, big=True)
            renderer.draw_hud_block([
                f"t      {step_count * params.dt:6.2f} s   (step {step_count})",
                f"score  {int(score)}",
                f"v      {float(np.hypot(agent[2], agent[3])):5.2f}",
                f"reward = {float(r[0]):+.4f}",
            ], 12, 36)
            label = "agent (policy)" if action_fn else "agent (keyboard)"
            renderer.draw_text(label, 12, world_h - 20, color=AGENT)

        # Help in the top right.
        for i, line in enumerate(help_lines):
            renderer.draw_text(line, world_w - 250, 10 + i * 17, color=(110, 118, 130))
        if paused:
            renderer.draw_text("[PAUSED]", world_w - 100, world_h - 20,
                                color=(180, 60, 60), big=True)

        renderer.flip()
        # dt=0.5 → real-time would be 2 fps, painfully slow to watch.
        # Run at 30 fps (≈15× real-time playback).
        clock.tick(30)

    renderer.close()


def _build_action_fn(ckpt, act_dim: int) -> "callable":
    """Return a jit'd deterministic policy function from a loaded checkpoint."""
    saved = ckpt["args"]
    model = ActorCritic(
        act_dim=act_dim,
        hidden=tuple(saved.get("hidden", [256, 256])),
        log_std_init=saved.get("log_std_init", 0.0),
    )
    net_params = ckpt["params"]
    obs_stats = RunningStats(
        mean=jnp.asarray(ckpt["obs_stats"].mean),
        var=jnp.asarray(ckpt["obs_stats"].var),
        count=jnp.asarray(ckpt["obs_stats"].count),
    )

    @jax.jit
    def _act(obs):
        norm = normalize(obs, obs_stats)
        mean, _ls, _v = model.apply(net_params, norm)
        return mean

    def fn(obs_np: np.ndarray) -> np.ndarray:
        return np.asarray(_act(jnp.asarray(obs_np)))
    return fn


def run_2d_viewer_selfplay(args, self_ckpt) -> None:
    """Pygame 2D viewer for pursuit_selfplay. Renders both vehicles, each
    driven by its own (deterministic) policy.
    """
    import pygame as pg
    from collections import deque

    from jax_playground.envs.pursuit_selfplay import env as sp_env
    from jax_playground.envs.pursuit.vehicle import default_params as default_v_params
    from jax_playground.render2d import (
        Renderer2D, PURSUER, EVADER, CATCH_ZONE, TRAIL_PURSUER, TRAIL_EVADER,
    )

    params = default_v_params()
    obs_dim, act_dim = sp_env.OBS_DIM, sp_env.ACT_DIM

    role = self_ckpt["args"].get("role", self_ckpt.get("role", "pursuer"))
    self_fn = _build_action_fn(self_ckpt, act_dim)

    opp_path = args.opp_model or self_ckpt["args"].get("opp_model")
    if opp_path is None:
        print("[viewer] no opponent — stationary")
        opp_fn = lambda _o: np.zeros(act_dim, dtype=np.float32)
    else:
        print(f"[viewer] opponent ← {opp_path}")
        with Path(opp_path).open("rb") as f:
            opp_ckpt = pickle.load(f)
        opp_fn = _build_action_fn(opp_ckpt, act_dim)

    print(f"[viewer] role={role}  run={args.run}")

    reset_one = jax.jit(lambda k: sp_env.env_reset(k, params))
    step_one = jax.jit(lambda s, ae, ap: sp_env.env_step(s, ae, ap, params))

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    env_state, evader_obs, pursuer_obs = reset_one(k)

    world_w = int(params.world_w)
    world_h = int(params.world_h)
    max_window_h = 900
    if world_h > max_window_h:
        scale = max_window_h / float(world_h)
        win_w, win_h = int(world_w * scale), max_window_h
    else:
        win_w, win_h = world_w, world_h
    renderer = Renderer2D(world_w, world_h, "pursuit_selfplay-v0",
                           win_w=win_w, win_h=win_h)

    trails = {
        "evader": deque(maxlen=300),
        "pursuer": deque(maxlen=300),
    }
    paused = False
    step_count = 0
    last_reward = 0.0
    clock = pg.time.Clock()
    running = True
    role_is_evader = (role == "evader")
    help_lines = [
        f"trained:  {role}",
        f"opp:      {'(stationary)' if opp_path is None else Path(opp_path).parent.name}",
        "Space     pause",
        "Backspace reset",
        "Esc       quit",
    ]

    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif ev.key == pg.K_SPACE:
                    paused = not paused
                elif ev.key == pg.K_BACKSPACE:
                    rng, k = jax.random.split(rng)
                    env_state, evader_obs, pursuer_obs = reset_one(k)
                    step_count = 0
                    for d in trails.values():
                        d.clear()

        if not paused:
            self_obs_np = np.asarray(evader_obs if role_is_evader else pursuer_obs)
            opp_obs_np = np.asarray(pursuer_obs if role_is_evader else evader_obs)
            self_a = self_fn(self_obs_np)
            opp_a = opp_fn(opp_obs_np)
            evader_a = self_a if role_is_evader else opp_a
            pursuer_a = opp_a if role_is_evader else self_a
            env_state, evader_obs, pursuer_obs, evader_r, done, info = step_one(
                env_state, jnp.asarray(evader_a), jnp.asarray(pursuer_a)
            )
            step_count += 1
            last_reward = float(evader_r) if role_is_evader else -float(evader_r)

        evader = np.asarray(env_state.evader)
        pursuer = np.asarray(env_state.pursuer)
        trails["evader"].append((float(evader[0]), float(evader[1])))
        trails["pursuer"].append((float(pursuer[0]), float(pursuer[1])))

        renderer.clear()
        renderer.draw_grid()
        renderer.draw_border()
        renderer.draw_obstacles(np.asarray(params.obstacles))

        from jax_playground.envs.pursuit_selfplay.env import _CATCH_RADIUS
        renderer.draw_catch_zone(float(pursuer[0]), float(pursuer[1]),
                                  float(_CATCH_RADIUS))
        renderer.draw_trail(list(trails["pursuer"]), TRAIL_PURSUER)
        renderer.draw_trail(list(trails["evader"]), TRAIL_EVADER)
        renderer.draw_vehicle(float(pursuer[0]), float(pursuer[1]),
                               float(pursuer[6]), PURSUER)
        renderer.draw_vehicle(float(evader[0]), float(evader[1]),
                               float(evader[6]), EVADER)

        d = float(np.hypot(pursuer[0] - evader[0], pursuer[1] - evader[1]))
        renderer.draw_text("pursuit_selfplay-v0", 12, 10, big=True)
        renderer.draw_hud_block([
            f"t      {step_count * params.dt:6.2f} s   (step {step_count})",
            f"d      {d:7.2f}   (catch ≤ 80)",
            f"pursuer v = {float(np.hypot(pursuer[2], pursuer[3])):5.2f}",
            f"evader  v = {float(np.hypot(evader[2], evader[3])):5.2f}",
            f"reward ({role}) = {last_reward:+.4f}",
        ], 12, 36)
        renderer.draw_text(
            f"pursuer{' ★' if not role_is_evader else ''}",
            12, world_h - 38, color=PURSUER, big=True,
        )
        renderer.draw_text(
            f"evader{' ★' if role_is_evader else ''}",
            12, world_h - 20, color=EVADER,
        )

        for i, line in enumerate(help_lines):
            renderer.draw_text(line, world_w - 250, 10 + i * 17,
                                color=(110, 118, 130))
        if paused:
            renderer.draw_text("[PAUSED]", world_w - 100, world_h - 20,
                                color=(180, 60, 60), big=True)

        renderer.flip()
        clock.tick(30)

    renderer.close()


def _w2s(world_pos, camera_world, zoom, window_center):
    """world → screen: (world - camera_world) * zoom + window_center."""
    return (world_pos - camera_world) * zoom + window_center


def _draw_extended_grid(surface, camera_world, zoom, window_center,
                        *, spacing: int = 80, bold_every: int = 4) -> None:
    """Grid lines at world_x = k·spacing, transformed by camera+zoom; only
    the visible k range is drawn. Adapts step density at extreme zooms by
    multiplying spacing so the screen never gets carpeted with lines."""
    import pygame as pg
    from jax_playground.render2d import GRID, GRID_BOLD
    win_w, win_h = surface.get_size()
    # Scale spacing so on-screen step is between ~30 and ~160 px.
    step_px = spacing * zoom
    factor = 1
    while step_px * factor < 30:
        factor *= 2
    while step_px * factor > 160 and factor > 1:
        factor //= 2
    eff_spacing = spacing * factor

    cx, cy = float(camera_world[0]), float(camera_world[1])
    wc_x, wc_y = float(window_center[0]), float(window_center[1])

    # Visible world x range: x_world such that 0 <= (x_world - cx)*zoom + wc_x <= win_w
    x_min_world = (0 - wc_x) / zoom + cx
    x_max_world = (win_w - wc_x) / zoom + cx
    k_min = int(np.floor(x_min_world / eff_spacing))
    k_max = int(np.ceil(x_max_world / eff_spacing)) + 1
    for k in range(k_min, k_max):
        x_world = k * eff_spacing
        x = int(round((x_world - cx) * zoom + wc_x))
        color = GRID_BOLD if (k * factor) % bold_every == 0 else GRID
        pg.draw.line(surface, color, (x, 0), (x, win_h))

    y_min_world = (0 - wc_y) / zoom + cy
    y_max_world = (win_h - wc_y) / zoom + cy
    k_min = int(np.floor(y_min_world / eff_spacing))
    k_max = int(np.ceil(y_max_world / eff_spacing)) + 1
    for k in range(k_min, k_max):
        y_world = k * eff_spacing
        y = int(round((y_world - cy) * zoom + wc_y))
        color = GRID_BOLD if (k * factor) % bold_every == 0 else GRID
        pg.draw.line(surface, color, (0, y), (win_w, y))


def _draw_world_border(surface, world_w: int, world_h: int,
                        camera_world, zoom, window_center) -> None:
    """World boundary transformed by camera+zoom — orientation reference."""
    import pygame as pg
    from jax_playground.render2d import GRID_BOLD
    tl = _w2s(np.array([0.0, 0.0]), camera_world, zoom, window_center)
    br = _w2s(np.array([float(world_w), float(world_h)]),
              camera_world, zoom, window_center)
    rect = pg.Rect(int(round(tl[0])), int(round(tl[1])),
                   int(round(br[0] - tl[0])), int(round(br[1] - tl[1])))
    pg.draw.rect(surface, GRID_BOLD, rect, width=2)


def run_swarm_viewer(action_fn, args) -> None:
    """Pygame 2D viewer for the swarm env with a follow-camera, scroll-wheel
    zoom (around the cursor), and LMB-drag pan.

    Camera state is `camera_world` (the world point shown at window center)
    plus a `zoom` factor. When follow is on, camera_world EMAs toward the
    swarm centroid so the swarm sits at window center. Mouse drag turns
    follow off and gives manual control; C re-engages it.

    Controls: SPACE pause, BACKSPACE reset, G ghost, W waves, C follow,
    R reset zoom+camera, scroll = zoom, LMB drag = pan, Esc quit.
    """
    import pygame as pg

    from jax_playground import envs as envlib
    from jax_playground.envs.swarm import physics as swarm_phys
    from jax_playground.envs.swarm.render import (
        draw_target, draw_ghost, draw_agents, draw_wave_glow,
    )
    from jax_playground.render2d import Renderer2D

    params = swarm_phys.default_params()
    reset_b, step_b, obs_dim, act_dim = envlib.make_batched("swarm", params, 1)

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    env_state, obs = reset_b(k)

    world_w = int(params.world_w)
    world_h = int(params.world_h)
    renderer = Renderer2D(world_w, world_h, "swarm-v0",
                           win_w=world_w, win_h=world_h)
    win_center = np.array([world_w / 2.0, world_h / 2.0], dtype=np.float32)

    paused = False
    show_ghost = True
    show_waves = True
    follow_camera = True
    # camera_world: world point shown at window center. zoom: world→screen scale.
    camera_world = np.array([world_w / 2.0, world_h / 2.0], dtype=np.float32)
    zoom = 1.0
    CAMERA_EMA = 0.18
    ZOOM_MIN, ZOOM_MAX = 0.1, 8.0

    dragging = False
    drag_last_screen = (0, 0)

    step_count = 0
    last_reward = 0.0
    last_shape_err = 0.0
    last_shape_err_n = 0.0
    last_solved = 0.0
    clock = pg.time.Clock()
    running = True

    help_lines = ([
        "policy from --run is driving"] if action_fn else
        ["no policy: agents are still (action=0)"]
    ) + [
        "Space         pause",
        "Backspace     reset env",
        "G             toggle ghost",
        "W             toggle waves",
        "C             toggle follow",
        "R             reset view",
        "scroll        zoom",
        "LMB drag      pan",
        "Esc           quit",
    ]

    while running:
        mx, my = pg.mouse.get_pos()
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif ev.key == pg.K_SPACE:
                    paused = not paused
                elif ev.key == pg.K_BACKSPACE:
                    rng, k = jax.random.split(rng)
                    env_state, obs = reset_b(k)
                    step_count = 0
                elif ev.key == pg.K_g:
                    show_ghost = not show_ghost
                elif ev.key == pg.K_w:
                    show_waves = not show_waves
                elif ev.key == pg.K_c:
                    follow_camera = not follow_camera
                elif ev.key == pg.K_r:
                    camera_world = np.array([world_w / 2.0, world_h / 2.0],
                                             dtype=np.float32)
                    zoom = 1.0
                    follow_camera = True
            elif ev.type == pg.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True
                    drag_last_screen = ev.pos
                    follow_camera = False     # manual pan disengages follow
            elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pg.MOUSEMOTION and dragging:
                dx = ev.pos[0] - drag_last_screen[0]
                dy = ev.pos[1] - drag_last_screen[1]
                drag_last_screen = ev.pos
                # World moves opposite mouse drag. Screen→world delta is /zoom.
                camera_world = camera_world - np.array([dx, dy], dtype=np.float32) / zoom
            elif ev.type == pg.MOUSEWHEEL:
                # Zoom around the cursor: keep world point under cursor fixed.
                factor = 1.15 if ev.y > 0 else 1.0 / 1.15
                new_zoom = float(np.clip(zoom * factor, ZOOM_MIN, ZOOM_MAX))
                if new_zoom != zoom:
                    mouse_screen = np.array([mx, my], dtype=np.float32)
                    mouse_world = (mouse_screen - win_center) / zoom + camera_world
                    camera_world = mouse_world - (mouse_screen - win_center) / new_zoom
                    zoom = new_zoom

        if not paused:
            if action_fn is not None:
                a = action_fn(np.asarray(obs))
                action = jnp.asarray(a)
            else:
                action = jnp.zeros((params.n_agents, act_dim), dtype=jnp.float32)
            env_state, obs, r, done, info = step_b(env_state, action)
            step_count += 1
            last_reward = float(r[0])
            last_shape_err = float(info["shape_err"][0])
            last_shape_err_n = float(info["shape_err_n"][0])
            last_solved = float(info["solved"][0])

        pos = np.asarray(env_state.pos[0])
        target = np.asarray(env_state.target[0])
        emissions = np.asarray(env_state.last_emissions[0])
        shape_id = int(env_state.shape_id[0])
        SHAPE_NAMES = ["circle", "square", "triangle", "hexagon", "line"]
        centroid = pos.mean(axis=0)

        # Follow: EMA camera_world toward centroid (only when follow is on).
        if follow_camera:
            camera_world = camera_world + CAMERA_EMA * (centroid - camera_world)

        # Pre-transform world points to screen for the existing draw helpers.
        pos_s = _w2s(pos, camera_world, zoom, win_center)
        target_s = _w2s(target, camera_world, zoom, win_center)

        renderer.clear()
        _draw_extended_grid(renderer.surface, camera_world, zoom, win_center)
        _draw_world_border(renderer.surface, world_w, world_h,
                           camera_world, zoom, win_center)
        draw_target(renderer.surface, target_s)
        if show_ghost:
            draw_ghost(renderer.surface, target_s, pos_s.mean(axis=0))
        if show_waves:
            # Wave glow radii are world-units × amplitude — scale by zoom so
            # halos shrink/grow with the agents.
            draw_wave_glow(renderer.overlay, pos_s, emissions,
                           params.lambda_short * zoom,
                           params.lambda_long * zoom)
        draw_agents(renderer.surface, pos_s)

        renderer.draw_text("swarm-v0", 12, 10, big=True)
        renderer.draw_hud_block([
            f"shape:    {SHAPE_NAMES[shape_id]}",
            f"step:     {step_count} / {params.timestep_limit}",
            f"err:      {last_shape_err:7.2f} px   (norm {last_shape_err_n:.3f}, solve ≤ {params.solve_threshold})",
            f"reward:   {last_reward:+.4f}",
            f"solved:   {'yes' if last_solved > 0.5 else 'no'}",
            f"view:     zoom×{zoom:.2f}  {'follow' if follow_camera else 'manual'}  centroid=({centroid[0]:.0f}, {centroid[1]:.0f})",
        ], 12, 36)
        for i, line in enumerate(help_lines):
            renderer.draw_text(line, world_w - 250, 10 + i * 17, color=(110, 118, 130))
        if paused:
            renderer.draw_text("[PAUSED]", world_w - 100, world_h - 20,
                                color=(180, 60, 60), big=True)
        renderer.flip()
        clock.tick(30)

    renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["hyro", "linear", "pursuit", "gathering", "swarm"],
                        default=None)
    parser.add_argument("--run", type=Path, default=None,
                        help="If set, drive the env with a trained policy from this run dir.")
    parser.add_argument("--opp-model", type=str, default=None,
                        help="(Self-play only) override the opponent path saved "
                             "in the run's args.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Self-play checkpoints carry a "role" field. Dispatch to the dual-agent
    # viewer when we find one.
    if args.run is not None:
        with (args.run / "model.pkl").open("rb") as f:
            _peek_ckpt = pickle.load(f)
        if "role" in _peek_ckpt.get("args", {}) or "role" in _peek_ckpt:
            return run_2d_viewer_selfplay(args, _peek_ckpt)

    env_kind, action_fn = load_policy(args.run, args.env)
    if env_kind in ("pursuit", "gathering"):
        return run_2d_viewer(env_kind, action_fn, args)
    if env_kind == "swarm":
        return run_swarm_viewer(action_fn, args)

    params = default_hyro_params() if env_kind == "hyro" else default_linear_params()
    state = build_state(env_kind, params, args.seed)

    init_gl()
    fonts = {
        "hud":  GLUT_BITMAP_9_BY_15,
        "title": GLUT_BITMAP_HELVETICA_18,
        "help": GLUT_BITMAP_HELVETICA_12,
    }
    cam = OrbitCamera(target=np.asarray(state.position).copy(),
                      distance=4.5, azimuth=35.0, elevation=22.0)

    paused = False
    show_com = show_axes = show_contact = show_hud = True
    show_trail = True
    show_plot = True
    show_help = action_fn is None  # show help when manual
    trail = deque(maxlen=TRAIL_LEN)
    plot_hist = deque(maxlen=PLOT_LEN)
    step_count = 0
    dragging = False
    last_mouse = (0, 0)
    rng_seed_counter = [args.seed]

    def reset_state():
        rng_seed_counter[0] += 1
        return build_state(env_kind, params, rng_seed_counter[0])

    clock = pg.time.Clock()
    running = True
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_SPACE:
                    paused = not paused
                elif ev.key == K_BACKSPACE:
                    state = reset_state()
                    step_count = 0
                    trail.clear(); plot_hist.clear()
                elif ev.key == K_TAB:
                    env_kind = "linear" if env_kind == "hyro" else "hyro"
                    params = default_hyro_params() if env_kind == "hyro" else default_linear_params()
                    if args.run is not None:
                        # Policy is env-specific; can't switch when --run set.
                        print("[viewer] switching env without trained policy — policy was env-specific.")
                        action_fn = None
                    state = reset_state()
                    step_count = 0
                    trail.clear(); plot_hist.clear()
                elif ev.key == K_1:
                    show_com = not show_com
                elif ev.key == K_2:
                    show_axes = not show_axes
                elif ev.key == K_3:
                    show_contact = not show_contact
                elif ev.key == K_4:
                    show_hud = not show_hud
                elif ev.key == K_5:
                    show_trail = not show_trail
                    if not show_trail: trail.clear()
                elif ev.key == K_6:
                    show_plot = not show_plot
                    if not show_plot: plot_hist.clear()
                elif ev.key in (K_QUESTION, K_SLASH):
                    show_help = not show_help
            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True; last_mouse = ev.pos
                elif ev.button == 4:
                    cam.zoom(0.9)
                elif ev.button == 5:
                    cam.zoom(1.1)
            elif ev.type == MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                dx = ev.pos[0] - last_mouse[0]
                dy = ev.pos[1] - last_mouse[1]
                last_mouse = ev.pos
                cam.orbit(-dx * 0.4, dy * 0.4)

        # Decide action
        if action_fn is not None:
            if env_kind == "hyro":
                obs = hyro_obs_np(state, params)
            else:
                obs = linear_obs_np(state, params)
            action = action_fn(obs)
        else:
            action = keyboard_action(env_kind, pg.key.get_pressed())

        if not paused:
            if env_kind == "hyro":
                state = _hyro_step_j(state, jnp.asarray(action), params)
            else:
                state = _linear_step_j(state, jnp.asarray(action), params)
            step_count += 1
            pos = np.asarray(state.position)
            if show_trail:
                trail.append(pos.copy())
            if show_plot:
                if env_kind == "hyro":
                    plot_hist.append(np.sin(np.asarray(state.phi)).copy())
                else:
                    shifts = np.asarray(state.shifts)
                    plot_hist.append(shifts / float(params.radius) * 2.0 - 1.0)

        # Build adapter for legacy viz
        if env_kind == "hyro":
            adapter = _HyroAdapter(state, params)
            R = _hyro_R(adapter)
        else:
            adapter = _LinearAdapter(state, params)
            R = adapter.shifts[:, None] * adapter.U

        cam.target = adapter.position.copy()

        glClearColor(0.93, 0.94, 0.96, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cam.push()
        drawGroundGrid(extent=40.0, spacing=0.5)
        if show_trail and len(trail) >= 2:
            drawTrajectory(list(trail), line_width=3.5, min_alpha=0.2)
        glPushMatrix()
        glTranslatef(*adapter.position)
        if env_kind == "hyro":
            drawHyrosphere(adapter)
        else:
            drawLinearsphere(adapter)
        glPopMatrix()
        if show_axes:
            drawBodyAxes(adapter.position, adapter.U, length=0.5)
            drawOmegaArrow(adapter.position, adapter.Omega)
        if show_com:
            drawCOMMarker(_com_offset(adapter, R))
        if show_contact:
            drawContactAndFriction(adapter.position, adapter.radius,
                                    adapter.last_F_fric, adapter.last_in_contact)
        cam.pop()

        if show_plot and len(plot_hist) >= 2:
            arr = np.asarray(plot_hist)
            traces = [arr[:, i].tolist() for i in range(arr.shape[1])]
            margin = 16
            pw, ph = 320, 140
            px, py = WIN_W - pw - margin, margin
            drawMiniPlot(traces, x=px, y=py, width=pw, height=ph,
                          win_w=WIN_W, win_h=WIN_H, value_range=(-1.05, 1.05))
            label = "sin(phi_i)" if env_kind == "hyro" else "shifts (norm.)"
            drawText2D(label, px + 6, py + ph - 14,
                       font=fonts["help"], color=(0.1, 0.1, 0.15))

        if show_hud:
            tag = "[POLICY]" if action_fn is not None else "[MANUAL]"
            name = "HyroSphere" if env_kind == "hyro" else "LinearSphere"
            pos = adapter.position
            vel = adapter.velocity
            Om = adapter.Omega
            lines = [
                f"{name} {tag}  step {step_count}",
                f"pos:  {pos[0]:+7.3f} {pos[1]:+7.3f} {pos[2]:+7.3f}",
                f"v:    {vel[0]:+7.3f} {vel[1]:+7.3f} {vel[2]:+7.3f}   |v|={np.linalg.norm(vel):.3f}",
                f"Omega:{Om[0]:+7.3f} {Om[1]:+7.3f} {Om[2]:+7.3f}",
                f"action: [" + " ".join(f"{a:+.1f}" for a in action) + "]",
            ]
            x, y = 12, WIN_H - 24
            drawText2D(lines[0], x, y, font=fonts["title"], color=(0.05, 0.05, 0.1))
            y -= 22
            for ln in lines[1:]:
                drawText2D(ln, x, y, font=fonts["hud"], color=(0.05, 0.05, 0.1))
                y -= 18

        if show_help:
            x, y = WIN_W - 320, WIN_H - 24
            drawText2D("controls (? to hide)", x, y,
                       font=fonts["help"], color=(0.15, 0.15, 0.18))
            y -= 18
            for ln in HELP_LINES:
                drawText2D(ln, x, y, font=fonts["help"], color=(0.15, 0.15, 0.18))
                y -= 16

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()
