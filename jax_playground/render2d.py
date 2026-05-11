"""Shared 2D rendering for the three env adapters.

Uses pygame.gfxdraw for anti-aliased filled polygons / circles so the agents
look clean at the new 960x720 world scale. All drawing is in world pixel
coordinates; the surface IS the world (window size == world size).

Coordinate convention: pygame screen, +x right, +y down. Vehicle heading
`phi` is in math radians but interpreted directly in screen coords, since
the vehicle dynamics already use `vy += k*sin(phi)*dt` with +y-down velocity
(see vehicle.py). So phi=0 points right, phi=pi/2 points down, etc.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---- palette ------------------------------------------------------------------
BG = (245, 247, 250)
GRID = (224, 228, 234)
GRID_BOLD = (200, 205, 212)
TEXT = (30, 32, 38)
TEXT_DIM = (110, 118, 130)
PURSUER = (220, 60, 60)       # red
EVADER = (40, 110, 220)       # blue
AGENT = (40, 110, 220)        # same as evader by default
TARGET = (245, 165, 55)       # orange
CATCH_ZONE = (220, 60, 60, 64)   # translucent red fill
CATCH_ZONE_RING = (220, 60, 60)
TRAIL_PURSUER = (220, 60, 60)
TRAIL_EVADER = (40, 110, 220)


class Renderer2D:
    """Owns the pygame window/surface, font, and a few per-agent trail deques.

    Envs construct a Renderer2D in their lazy `_init_renderer()`, then call
    `clear()`, the various `draw_*` helpers, and `flip()` once per frame.
    """

    def __init__(self, world_w: int, world_h: int, title: str):
        import pygame as pg

        pg.init()
        self.surface = pg.display.set_mode((world_w, world_h))
        pg.display.set_caption(title)

        self.world_w = world_w
        self.world_h = world_h
        self.font = pg.font.SysFont("Helvetica,Arial,Sans", 13)
        self.font_big = pg.font.SysFont("Helvetica,Arial,Sans", 18, bold=True)
        # Per-pixel-alpha overlay for translucent fills (catch zone, trails).
        # Blitted onto the main surface in `flip()`.
        self.overlay = pg.Surface((world_w, world_h), pg.SRCALPHA)

    # ---- frame primitives -----------------------------------------------------
    def clear(self) -> None:
        self.surface.fill(BG)
        self.overlay.fill((0, 0, 0, 0))

    def flip(self) -> None:
        import pygame as pg
        self.surface.blit(self.overlay, (0, 0))
        pg.display.flip()

    def close(self) -> None:
        import pygame as pg
        pg.quit()

    # ---- world overlays -------------------------------------------------------
    def draw_grid(self, spacing: int = 80, bold_every: int = 4) -> None:
        import pygame as pg
        for ix, x in enumerate(range(0, self.world_w + 1, spacing)):
            color = GRID_BOLD if ix % bold_every == 0 else GRID
            pg.draw.line(self.surface, color, (x, 0), (x, self.world_h))
        for iy, y in enumerate(range(0, self.world_h + 1, spacing)):
            color = GRID_BOLD if iy % bold_every == 0 else GRID
            pg.draw.line(self.surface, color, (0, y), (self.world_w, y))

    def draw_border(self) -> None:
        import pygame as pg
        pg.draw.rect(self.surface, GRID_BOLD,
                     pg.Rect(0, 0, self.world_w, self.world_h), width=2)

    # ---- agents / vehicles ----------------------------------------------------
    def draw_vehicle(self, x: float, y: float, phi: float, color, scale: float = 1.0) -> None:
        """Solid triangular arrow centered at (x, y) pointing along phi.

        Geometry: 22-px arrow at scale=1, body 14 wide. Anti-aliased via
        gfxdraw.aapolygon + filled_polygon (gfxdraw doesn't anti-alias the
        filled edges, but the aapolygon overlay smooths the visible silhouette).
        """
        from pygame import gfxdraw
        L_front = 18.0 * scale
        L_back = 9.0 * scale
        W_back = 9.0 * scale
        cp, sp = np.cos(phi), np.sin(phi)

        def world(lx: float, ly: float) -> tuple[int, int]:
            wx = x + lx * cp - ly * sp
            wy = y + lx * sp + ly * cp
            return int(round(wx)), int(round(wy))

        pts = [
            world(L_front, 0.0),
            world(-L_back,  W_back),
            world(-L_back, -W_back),
        ]
        gfxdraw.filled_polygon(self.surface, pts, color)
        gfxdraw.aapolygon(self.surface, pts, color)
        # Small center dot to highlight the actual position.
        gfxdraw.filled_circle(self.surface, int(round(x)), int(round(y)),
                              max(2, int(2 * scale)), (255, 255, 255))

    def draw_catch_zone(self, x: float, y: float, radius: float) -> None:
        """Translucent disk + crisp ring at the catch radius."""
        from pygame import gfxdraw
        cx, cy, r = int(round(x)), int(round(y)), int(round(radius))
        # Fill goes on the overlay so we can use real alpha blending.
        gfxdraw.filled_circle(self.overlay, cx, cy, r, CATCH_ZONE)
        # Ring on the main surface for a clean edge.
        gfxdraw.aacircle(self.surface, cx, cy, r, CATCH_ZONE_RING)

    def draw_target(self, x: float, y: float, radius: float, color=TARGET) -> None:
        from pygame import gfxdraw
        cx, cy, r = int(round(x)), int(round(y)), int(round(radius))
        # Soft outer halo.
        gfxdraw.filled_circle(self.overlay, cx, cy, r, (*color, 50))
        # Filled core.
        gfxdraw.filled_circle(self.surface, cx, cy, max(4, r // 3), color)
        gfxdraw.aacircle(self.surface, cx, cy, max(4, r // 3), color)
        gfxdraw.aacircle(self.surface, cx, cy, r, color)

    def draw_trail(self, points: Sequence[tuple[float, float]], color,
                   width: int = 3, min_alpha: int = 30, max_alpha: int = 220) -> None:
        """Draw a per-agent trail as fading-alpha line segments. Older
        segments are dim, the freshest is opaque. Caller owns the deque."""
        import pygame as pg
        n = len(points)
        if n < 2:
            return
        for i in range(1, n):
            t = i / (n - 1)
            alpha = int(min_alpha + (max_alpha - min_alpha) * t)
            pg.draw.line(self.overlay, (*color, alpha),
                         points[i - 1], points[i], width)

    # ---- text -----------------------------------------------------------------
    def draw_text(self, text: str, x: int, y: int, *, color=TEXT, big: bool = False) -> None:
        font = self.font_big if big else self.font
        surf = font.render(text, True, color)
        self.surface.blit(surf, (x, y))

    def draw_hud_block(self, lines: Sequence[str], x: int, y: int, *,
                       color=TEXT, line_h: int = 17) -> None:
        for i, line in enumerate(lines):
            self.draw_text(line, x, y + i * line_h, color=color)
