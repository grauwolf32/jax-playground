"""Pygame draw helpers for the swarm env.

Drawing convention matches `render2d.py` (the surface IS the world: window
size == world size unless explicitly downscaled). All inputs are world-pixel
coords, ints or floats.

The renderer draws:
  - target shape: gray dots + thin gray polyline
  - ghost target shifted to align with current swarm centroid (cyan, dim) —
    visualises what the Chamfer reward is actually comparing against
  - agents: small filled circles, color cycles by index
  - optional wave glow: per agent, two concentric translucent disks whose
    radii follow the L2 of short / long emission amplitudes
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Palette tuned to match render2d.BG (245, 247, 250) background.
TARGET_DOT = (160, 168, 180)        # neutral gray
TARGET_LINK = (200, 206, 215)
GHOST_DOT = (90, 170, 200)          # dim cyan — centroid-aligned ghost
GHOST_LINK = (160, 205, 220)
AGENT_BASE = (40, 110, 220)         # blue
AGENT_RING = (15, 50, 130)
WAVE_SHORT = (60, 130, 230)         # blue, like agent
WAVE_LONG = (240, 150, 50)          # orange, like targets in other envs


def _agent_color(i: int, n: int) -> tuple[int, int, int]:
    """Hue cycle over swarm index. Keeps base lightness so positions are
    legible against the light background."""
    import colorsys
    h = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_target(surface, target_pts: np.ndarray) -> None:
    """target_pts: (n_agents, 2)."""
    import pygame as pg
    from pygame import gfxdraw
    n = target_pts.shape[0]
    # Connect targets in order with a thin polyline (just hint of the shape).
    if n >= 2:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in target_pts]
        for i in range(n):
            pg.draw.aaline(surface, TARGET_LINK, pts[i], pts[(i + 1) % n])
    for p in target_pts:
        cx, cy = int(round(p[0])), int(round(p[1]))
        gfxdraw.aacircle(surface, cx, cy, 4, TARGET_DOT)
        gfxdraw.filled_circle(surface, cx, cy, 3, TARGET_DOT)


def draw_ghost(surface, target_pts: np.ndarray, swarm_centroid: np.ndarray) -> None:
    """Draw target shape shifted so its centroid matches `swarm_centroid` —
    the actual reference the Chamfer reward uses."""
    import pygame as pg
    from pygame import gfxdraw
    shifted = target_pts - target_pts.mean(axis=0) + swarm_centroid
    n = shifted.shape[0]
    if n >= 2:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in shifted]
        for i in range(n):
            pg.draw.aaline(surface, GHOST_LINK, pts[i], pts[(i + 1) % n])
    for p in shifted:
        cx, cy = int(round(p[0])), int(round(p[1]))
        gfxdraw.aacircle(surface, cx, cy, 3, GHOST_DOT)


def draw_agents(surface, pos: np.ndarray, *, radius: int = 5) -> None:
    """pos: (n_agents, 2). Each agent gets a hue-cycled solid disk + dark ring."""
    from pygame import gfxdraw
    n = pos.shape[0]
    for i in range(n):
        cx, cy = int(round(pos[i, 0])), int(round(pos[i, 1]))
        c = _agent_color(i, n)
        gfxdraw.filled_circle(surface, cx, cy, radius, c)
        gfxdraw.aacircle(surface, cx, cy, radius, AGENT_RING)


def draw_wave_glow(overlay, pos: np.ndarray, emissions: np.ndarray,
                   lambda_short: float, lambda_long: float,
                   alpha_scale: float = 60.0) -> None:
    """Draw translucent halos representing each agent's emission strength.

    emissions: (n_agents, 2 bands, K). The short / long halo radius is
    proportional to the band's L2 amplitude across channels, capped at the
    band's decay scale (1/e radius). Drawn on `overlay` so per-pixel alpha
    composes nicely.
    """
    from pygame import gfxdraw
    if emissions.size == 0:
        return
    n = pos.shape[0]
    # Band layout: 0 = short, 1 = medium (skipped for clarity), -1 = long.
    short_amp = np.linalg.norm(emissions[:, 0, :], axis=-1)   # (n,)
    long_amp = np.linalg.norm(emissions[:, -1, :], axis=-1)
    # Normalise by max channel count's sqrt so amp ∈ [0, 1] approx
    K = emissions.shape[-1]
    short_amp = np.clip(short_amp / np.sqrt(K), 0.0, 1.0)
    long_amp = np.clip(long_amp / np.sqrt(K), 0.0, 1.0)
    for i in range(n):
        cx, cy = int(round(pos[i, 0])), int(round(pos[i, 1]))
        if long_amp[i] > 0.02:
            r = int(long_amp[i] * lambda_long * 0.5)
            a = int(min(180, alpha_scale * long_amp[i]))
            gfxdraw.filled_circle(overlay, cx, cy, r, (*WAVE_LONG, a))
        if short_amp[i] > 0.02:
            r = int(short_amp[i] * lambda_short)
            a = int(min(220, alpha_scale * 1.5 * short_amp[i]))
            gfxdraw.filled_circle(overlay, cx, cy, r, (*WAVE_SHORT, a))


def draw_neighbor_links(surface, pos: np.ndarray, nbr_idx: np.ndarray,
                        max_per: int = 2) -> None:
    """Optional debug overlay: thin lines from each agent to its nearest few
    neighbors. nbr_idx: (n_agents, N_NBR) ints from neighbor_vectors() ordering.
    """
    import pygame as pg
    n = pos.shape[0]
    for i in range(n):
        for k in range(min(max_per, nbr_idx.shape[1])):
            j = int(nbr_idx[i, k])
            if 0 <= j < n:
                pg.draw.aaline(surface, (210, 215, 222),
                               (int(pos[i, 0]), int(pos[i, 1])),
                               (int(pos[j, 0]), int(pos[j, 1])))
