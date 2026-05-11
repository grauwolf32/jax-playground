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
    obs_dim = 43 if env_kind == "hyro" else 65
    act_dim = 4 if env_kind == "hyro" else 6

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["hyro", "linear"], default=None)
    parser.add_argument("--run", type=Path, default=None,
                        help="If set, drive the env with a trained policy from this run dir.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_kind, action_fn = load_policy(args.run, args.env)
    if env_kind not in ("hyro", "linear"):
        raise SystemExit(
            f"viewer.py currently only supports hyro/linear (3D OpenGL). "
            f"For {env_kind!r}, use play.py for headless eval until the 2D "
            f"renderer is ported.")
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
