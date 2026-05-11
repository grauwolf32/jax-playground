import sys
import string
import pygame as pg 
import numpy as np

from math import sqrt
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def rotate_vec(omega, phi):
    """Rodrigues rotation, numpy. Identity for |omega| < 1e-4.

    Self-contained copy — the JAX physics has its own (jit-friendly) version
    in `jax_hyrosphere.physics.rodrigues`. This one is host-side for OpenGL.
    """
    R = np.eye(3)
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-4:
        return R
    omega = omega / omega_norm
    W = np.zeros((3, 3))
    W[0][1] = -omega[2]
    W[0][2] = omega[1]
    W[1][2] = -omega[0]
    W[1][0] = -W[0][1]
    W[2][0] = -W[0][2]
    W[2][1] = -W[1][2]
    R = R + np.sin(phi) * W + (1.0 - np.cos(phi)) * np.dot(W, W)
    return R


# Shared palette used by both the 3D visualisation (one colour per wheel /
# slider) and the 2D mini-plot. Indexed by wheel/slider number.
WHEEL_COLORS = [
    (0.1, 0.4, 0.9),  # blue
    (0.9, 0.3, 0.3),  # red
    (0.2, 0.7, 0.2),  # green
    (0.9, 0.6, 0.1),  # orange
    (0.6, 0.3, 0.8),  # purple
    (0.2, 0.7, 0.7),  # cyan
]


def drawText(position, textString, font=None):
    """Draw `textString` at 3D world `position` using a GLUT bitmap font."""
    if font is None:
        font = GLUT_BITMAP_9_BY_15
    glColor3f(0.0, 0.0, 0.0)
    glRasterPos3d(*position)
    for ch in textString:
        glutBitmapCharacter(font, ord(ch))

def drawSphere(center, radius, colors):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(colors[0],colors[1],colors[2],colors[3])
    glutSolidSphere(radius,50,50)
    glDisable(GL_BLEND)
    glPopMatrix()

def drawCircle(radius_vec, omega ,center, color):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glColor4f(color[0],color[1],color[2],color[3])

    glBegin(GL_LINES)
    c_last = radius_vec
    for i in range(0, 100):
        c = np.dot(rotate_vec(omega, 2.0*np.pi*i/(99)), radius_vec)
        glVertex3fv(c_last)
        glVertex3fv(c)
        c_last = c

    glEnd()
    glPopMatrix()


def _perpendicular_unit(v):
    """Return a unit vector perpendicular to v. Robust to v aligned with any axis."""
    v = np.asarray(v, dtype=np.float64)
    # Pick the cardinal basis vector least aligned with v, then cross with v.
    ax = np.argmin(np.abs(v))
    e = np.zeros(3)
    e[ax] = 1.0
    p = np.cross(v, e)
    n = np.linalg.norm(p)
    if n < 1e-12:
        # v is essentially zero — direction is arbitrary.
        return np.array([1.0, 0.0, 0.0])
    return p / n


def drawCylinder(start, end, radius, color, close=True, nseg=20, mseg=40):
    d = np.asarray(end) - np.asarray(start)
    p = _perpendicular_unit(d) * radius

    points = []
    curr_point = p
    alpha = 2.0*np.pi / nseg

    for i in range(0, nseg):
        points.append(curr_point)
        curr_point = np.dot(rotate_vec(d, alpha), curr_point)

    layers = []
    curr_pos = start
    step = d / (mseg - 1) 

    for i in range(0, mseg):
        layer = [point + start + i*step for point in points]
        layers.append(layer)

    quads = []
    for i in range(0, mseg-1):
        for j in range(0, nseg-1):
            quads.append([layers[i][j], layers[i+1][j], layers[i+1][j+1], layers[i][j+1]])
        
        join_layer = [layers[mseg-1][0], layers[mseg-1][nseg-1], layers[0][nseg-1], layers[0][0]]
        quads.append(join_layer)
    
    glPushMatrix()
    glBegin(GL_QUADS)
    glColor4f(color[0],color[1],color[2],color[3])

    for quad in quads:
        glVertex3fv(quad[0])
        glVertex3fv(quad[1])
        glVertex3fv(quad[2])
        glVertex3fv(quad[3])

    glEnd()

    if close:
        glBegin(GL_POLYGON)
        for point in layers[0]:
            glVertex3fv(point)
        glEnd()

        glBegin(GL_POLYGON)
        for point in layers[mseg-1]:
            glVertex3fv(point)
        glEnd()

    glPopMatrix()
        
def drawCylinders(start, end, radius, color, close=True, nseg=10, mseg=10):
    n = np.asarray(start).shape[0]
    for i in range(0, n):
        drawCylinder(start[i], end[i], radius, color, close, nseg, mseg)

def drawCone(start, end, radius, color, close, nseg=20):
    d = np.asarray(end) - np.asarray(start)
    p = _perpendicular_unit(d) * radius

    points = []
    curr_point = p
    alpha = 2.0*np.pi / nseg

    for i in range(0, nseg):
        points.append(curr_point + start)
        curr_point = np.dot(rotate_vec(d, alpha), curr_point) 

    glPushMatrix()
    glBegin(GL_TRIANGLES)
    glColor4f(color[0],color[1],color[2],color[3])

    for i in range(0, nseg-1):
        glVertex3fv(points[i])
        glVertex3fv(end)
        glVertex3fv(points[i+1])

    glVertex3fv(points[nseg-1])
    glVertex3fv(end)
    glVertex3fv(points[0])

    glEnd()

    if close:
        glBegin(GL_POLYGON)
        for point in points:
            glVertex3fv(point)
        glEnd()

    glPopMatrix()

def drawArrow(start, end, radius, color, nseg=20, mseg=20):
    d = end - start
    h = 0.7*d

    drawCylinder(start, start+h, radius, color,close=True, nseg=nseg, mseg=mseg)
    drawCone(start+h, end, radius=radius*1.4, color=color, close=True, nseg=nseg )

def drawArrows(start, end, radius, color, nseg=20, mseg=20):
    n = np.asarray(start).shape[0]
    for i in range(0, n):
        drawArrow(start[i], end[i], radius, color, nseg, mseg)

def drawLines(start, end, color, position):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor4f(color[0],color[1],color[2],color[3])
    glBegin(GL_LINES)
    for i in range(0, len(start)):
        glVertex3fv(start[i])
        glVertex3fv(end[i])

    glEnd()
    glPopMatrix()

def drawHyrosphere(hyrosphere):
    #glTranslatef(hyrosphere.position[0],hyrosphere.position[1], hyrosphere.position[2])

    U = hyrosphere.U 
    A = U * np.sqrt(1.0/8.0)

    b1 = np.cross(A[0,:], A[1,:])
    b2 = np.cross(A[1,:], A[2,:])
    b3 = np.cross(A[2,:], A[3,:])
    b4 = np.cross(A[3,:], A[0,:])

    b1 = b1 / np.linalg.norm(b1) * hyrosphere.t_len / 2.0
    b2 = b2 / np.linalg.norm(b2) * hyrosphere.t_len / 2.0
    b3 = b3 / np.linalg.norm(b3) * hyrosphere.t_len / 2.0
    b4 = b4 / np.linalg.norm(b4) * hyrosphere.t_len / 2.0

    b1 = np.dot(rotate_vec(U[0,:], hyrosphere.phi[0]), b1)
    b2 = np.dot(rotate_vec(U[1,:], hyrosphere.phi[1]), b2)
    b3 = np.dot(rotate_vec(U[2,:], hyrosphere.phi[2]), b3)
    b4 = np.dot(rotate_vec(U[3,:], hyrosphere.phi[3]), b4)

    B = np.array([b1, b2, b3, b4])
    R = A + B

    drawArrow(np.zeros(3), hyrosphere.velocity, radius=0.02, color=(1, 0, 0, 0.5))

    # Per-wheel coloured geometry. Colour i ↔ mini-plot trace i.
    mass_r = hyrosphere.t_len * np.sqrt(3.0/8.0) / 20.0
    for i in range(4):
        r, g, b = WHEEL_COLORS[i]
        line_col = (r, g, b, 0.7)
        circle_col = (r, g, b, 0.45)
        sphere_col = (r, g, b, 0.95)
        drawLines(start=[np.zeros(3)], end=[A[i]], color=line_col, position=(0, 0, 0))
        drawLines(start=[A[i]],         end=[R[i]], color=line_col, position=(0, 0, 0))
        drawCircle(B[i], A[i], A[i], color=circle_col)
        drawSphere(center=R[i], radius=mass_r, colors=sphere_col)

    drawSphere(center=(0, 0, 0),
               radius=hyrosphere.t_len * np.sqrt(3.0/8.0),
               colors=(90.0/256, 1.0, 39.0/256, 0.3))
 
    #os_vec = np.asarray([0.0,0.0,-hyrosphere.radius])
    #drawArrow(os_vec, os_vec + hyrosphere.Omega , radius=0.02, color=(1,0,0,0.5), nseg=10, mseg=10)

    text = "position : {0:.2f} {1:.2f} {2:.2f}".format(hyrosphere.position[0], hyrosphere.position[1], hyrosphere.position[2])
    drawText(position=(-1.0,0.0,1.0), textString=text)

    text = "velocity : {0:.2f} {1:.2f} {2:.2f}".format(hyrosphere.velocity[0], hyrosphere.velocity[1], hyrosphere.velocity[2])
    drawText(position=(-1.0,0.0,0.8), textString=text)

def drawLinearsphere(linearsphere):
    #glTranslatef(hyrosphere.position[0],hyrosphere.position[1], hyrosphere.position[2])

    U = linearsphere.U
    R = np.dot(np.diag(linearsphere.shifts), U)

    drawArrow(np.zeros(3), linearsphere.velocity, radius=0.02, color=(1, 0, 0, 0.5))

    # Per-slider coloured guide axes (full unit U), and per-slider mass sphere
    # at R[i] = shifts[i] · U[i]. Colour i ↔ mini-plot trace i.
    mass_r = linearsphere.radius / 20.0
    for i in range(6):
        r, g, b = WHEEL_COLORS[i]
        drawLines(start=[np.zeros(3)], end=[U[i] * linearsphere.radius],
                  color=(r, g, b, 0.35), position=(0, 0, 0))
        drawSphere(center=R[i], radius=mass_r, colors=(r, g, b, 0.95))

    drawSphere(center=(0, 0, 0), radius=linearsphere.radius,
               colors=(90.0/256, 1.0, 39.0/256, 0.3))
 
    #os_vec = np.asarray([0.0,0.0,-hyrosphere.radius])
    #drawArrow(os_vec, os_vec + hyrosphere.Omega , radius=0.02, color=(1,0,0,0.5), nseg=10, mseg=10)

    #text = "position : {0:.2f} {1:.2f} {2:.2f}".format(linearsphere.position[0], linearsphere.position[1], linearsphere.position[2])
    #drawText(position=(-1.0,0.0,1.0), textString=text)

    #text = "velocity : {0:.2f} {1:.2f} {2:.2f}".format(hyrosphere.velocity[0], hyrosphere.velocity[1], hyrosphere.velocity[2])
    #drawText(position=(-1.0,0.0,0.8), textString=text)

class CameraBase(object):
    """camera.Base camera object all other inherit from..."""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0]):
        """create the camera
           pos = position of the camera
           rotation = rotation of camera"""
        self.posx, self.posy, self.posz = pos
        self.rotx, self.roty, self.rotz = rotation

    def push(self):
        """Activate the camera - anything rendered after this uses the cameras transformations."""
        glPushMatrix()

    def pop(self):
        """Deactivate the camera - must be called after push or will raise an OpenGL error"""
        glPopMatrix()

    def get_pos(self):
        """Return the position of the camera as a tuple"""
        return self.posx, self.posy, self.posz

    def set_pos(self, pos):
        """Set the position of the camera from a tuple"""
        self.posx, self.posy, self.posz = pos

    def get_rotation(self):
        """Return the rotation of the camera as a tuple"""
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        """Transforms the matrix so that all objects are facing camera - used in Image3D (billboard sprites)"""
        pass

    def set_skybox_data(self):
        """Transforms the view only for a skybox, ie only rotation is taken into account, not position"""
        pass

class LookFromCamera(CameraBase):
    """camera.LookFromCamera is a FPS camera"""
    def __init__(self, pos=(0,0,0), rotation=(0,0,0)):
        CameraBase.__init__(self, pos, rotation)

    def push(self):
        glPushMatrix()
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def pop(self):
        glPopMatrix()

    def get_pos(self):
        return self.posx, self.posy, self.posz

    def get_rotation(self):
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(-self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)

class LookAtCamera(CameraBase):
    """camera.LookAtCamera is a third-person camera"""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0],
                 distance=0):
        """create the camera
           pos is the position the camera is looking at
           rotation is how much we are rotated around the object
           distance is how far back from the object we are"""
        CameraBase.__init__(self, pos, rotation)
        self.distance = distance

    def push(self):
        glPushMatrix()
        glTranslatef(0, 0, -self.distance)
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)


# ----------------------------------------------------------------------
# Polished-viewer helpers (used by top-level viewer.py).
# These don't replace the legacy LookAtCamera and drawHyrosphere /
# drawLinearsphere functions above — those are still what the env's
# render() method calls.
# ----------------------------------------------------------------------


class OrbitCamera(object):
    """Mouse-orbit camera that looks at a moving target point.

    `azimuth` is the yaw around +z (degrees, measured from +x toward +y).
    `elevation` is the pitch above the horizontal (degrees, clamped to (-89, 89)).
    Use gluLookAt; up vector is fixed at +z.
    """

    def __init__(self, target=(0.0, 0.0, 0.0), distance=5.0,
                 azimuth=45.0, elevation=25.0):
        self.target = np.asarray(target, dtype=np.float64)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)

    def push(self):
        glPushMatrix()
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        r = self.distance
        eye = self.target + np.array([
            r * np.cos(el) * np.cos(az),
            r * np.cos(el) * np.sin(az),
            r * np.sin(el),
        ])
        gluLookAt(eye[0], eye[1], eye[2],
                  self.target[0], self.target[1], self.target[2],
                  0.0, 0.0, 1.0)

    def pop(self):
        glPopMatrix()

    def orbit(self, d_azimuth, d_elevation):
        self.azimuth = (self.azimuth + d_azimuth) % 360.0
        self.elevation = max(-89.0, min(89.0, self.elevation + d_elevation))

    def zoom(self, factor):
        self.distance = max(0.3, min(100.0, self.distance * factor))


def drawGroundGrid(extent=10.0, spacing=0.5,
                   minor_color=(0.7, 0.7, 0.7, 0.6),
                   major_color=(0.3, 0.3, 0.3, 0.9)):
    """Draw a square grid on z=0 with thicker lines on the axes."""
    glDisable(GL_LIGHTING)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor4f(*minor_color)
    n = int(extent / spacing)
    for i in range(-n, n + 1):
        v = i * spacing
        if i == 0:
            continue
        glVertex3f(v, -extent, 0.0)
        glVertex3f(v, extent, 0.0)
        glVertex3f(-extent, v, 0.0)
        glVertex3f(extent, v, 0.0)
    glEnd()
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor4f(*major_color)
    glVertex3f(-extent, 0.0, 0.0)
    glVertex3f(extent, 0.0, 0.0)
    glVertex3f(0.0, -extent, 0.0)
    glVertex3f(0.0, extent, 0.0)
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_LIGHTING)


def drawCOMMarker(world_pos, color=(1.0, 0.2, 0.2, 1.0), radius=0.04):
    """Solid sphere at the system COM."""
    drawSphere(center=world_pos, radius=radius, colors=color)


def drawBodyAxes(origin, U_vectors, length=0.4, color=(0.1, 0.5, 1.0, 1.0)):
    """One short arrow per body-fixed axis vector U_i emanating from `origin`."""
    o = np.asarray(origin, dtype=np.float64)
    for u in U_vectors:
        drawArrow(o, o + length * np.asarray(u), radius=0.012, color=color, nseg=10, mseg=4)


def drawOmegaArrow(origin, Omega, color=(1.0, 0.55, 0.0, 1.0), scale=0.2):
    """Single arrow for the body angular velocity. Magnitude scaled for visibility."""
    o = np.asarray(origin, dtype=np.float64)
    Om = np.asarray(Omega, dtype=np.float64)
    norm = np.linalg.norm(Om)
    if norm < 1e-3:
        return
    drawArrow(o, o + scale * Om, radius=0.018, color=color, nseg=12, mseg=4)


def drawTrajectory(positions, base_color=(1.0, 0.35, 0.1, 1.0), min_alpha=0.0,
                   line_width=2.0):
    """Polyline through `positions` (oldest → newest) with alpha fading from
    `min_alpha` at the head to `base_color[3]` at the tail.

    `positions` is expected to be a sequence of (3,) world-frame points.
    """
    n = len(positions)
    if n < 2:
        return
    lighting_was = glIsEnabled(GL_LIGHTING)
    glDisable(GL_LIGHTING)
    glLineWidth(line_width)
    r, g, b, a_max = base_color
    glBegin(GL_LINE_STRIP)
    inv = 1.0 / (n - 1)
    for i in range(n):
        t = i * inv  # 0 oldest → 1 newest
        alpha = min_alpha + (a_max - min_alpha) * t
        glColor4f(r, g, b, alpha)
        glVertex3f(positions[i][0], positions[i][1], positions[i][2])
    glEnd()
    glLineWidth(1.0)
    if lighting_was:
        glEnable(GL_LIGHTING)


def drawContactAndFriction(ball_position, ball_radius, F_fric, in_contact,
                           contact_color=(1.0, 0.1, 0.1, 1.0),
                           friction_color=(0.1, 1.0, 0.3, 1.0),
                           force_scale=0.05):
    """Red dot at S; green arrow showing friction direction (scaled for visibility)."""
    if not in_contact:
        return
    S = np.asarray(ball_position, dtype=np.float64) + np.array([0.0, 0.0, -ball_radius])
    drawSphere(center=S, radius=0.035, colors=contact_color)
    F = np.asarray(F_fric, dtype=np.float64)
    if np.linalg.norm(F) > 1e-3:
        drawArrow(S, S + force_scale * F, radius=0.012, color=friction_color, nseg=10, mseg=4)


def drawMiniPlot(traces, x, y, width, height, win_w, win_h,
                 colors=None, value_range=(-1.05, 1.05),
                 bg=(0.96, 0.97, 0.99, 0.85),
                 border=(0.3, 0.3, 0.35, 0.9),
                 zero_line=True):
    """Multi-trace time-series plot in window coordinates.

    `traces` is a list of equal-length sequences (oldest first).  Each trace
    is drawn as a colored line strip across the box; values outside
    `value_range` are clipped to the box.

    `x`, `y` are the bottom-left of the plot in window pixels;
    `win_w`, `win_h` are the framebuffer size (needed to set up ortho).
    """
    if not traces:
        return
    n = len(traces[0])
    if n < 2:
        return

    if colors is None:
        colors = [(*WHEEL_COLORS[i % len(WHEEL_COLORS)], 1.0)
                  for i in range(len(traces))]

    lighting_was = glIsEnabled(GL_LIGHTING)
    depth_was = glIsEnabled(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    # Switch to 2D ortho for the duration of the overlay.
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, win_w, 0, win_h)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Translucent background and border.
    glColor4f(*bg)
    glBegin(GL_QUADS)
    glVertex2f(x, y); glVertex2f(x + width, y)
    glVertex2f(x + width, y + height); glVertex2f(x, y + height)
    glEnd()
    glColor4f(*border)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y); glVertex2f(x + width, y)
    glVertex2f(x + width, y + height); glVertex2f(x, y + height)
    glEnd()

    if zero_line:
        v_mid = 0.5 * (value_range[0] + value_range[1])
        y_mid = y + (v_mid - value_range[0]) / (value_range[1] - value_range[0]) * height
        glColor4f(0.5, 0.5, 0.55, 0.5)
        glBegin(GL_LINES)
        glVertex2f(x + 2, y_mid); glVertex2f(x + width - 2, y_mid)
        glEnd()

    v_lo, v_hi = value_range
    v_span = max(v_hi - v_lo, 1e-9)
    inv_n = 1.0 / (n - 1)
    glLineWidth(1.8)
    for trace, color in zip(traces, colors):
        glColor4f(*color)
        glBegin(GL_LINE_STRIP)
        for i, v in enumerate(trace):
            vx = x + i * inv_n * width
            vc = max(v_lo, min(v_hi, float(v)))
            vy = y + (vc - v_lo) / v_span * height
            glVertex2f(vx, vy)
        glEnd()
    glLineWidth(1.0)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    if depth_was:
        glEnable(GL_DEPTH_TEST)
    if lighting_was:
        glEnable(GL_LIGHTING)


def drawText2D(text, x, y, font=None, color=(1.0, 1.0, 1.0)):
    """Render `text` at window-space (x, y). y is measured from the bottom.

    Uses GLUT bitmap fonts (Python 3.14 has a circular-import bug in pygame.font
    that makes pg.font.SysFont unusable). `color` is (r, g, b) in [0, 1].
    `font` is one of the GLUT_BITMAP_* constants; defaults to GLUT_BITMAP_9_BY_15.
    """
    if font is None:
        font = GLUT_BITMAP_9_BY_15
    lighting_was_enabled = glIsEnabled(GL_LIGHTING)
    depth_was_enabled = glIsEnabled(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor3f(*color)
    glWindowPos2i(int(x), int(y))
    for ch in text:
        glutBitmapCharacter(font, ord(ch))
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)
    if lighting_was_enabled:
        glEnable(GL_LIGHTING)


def setupLighting():
    """Enable a single directional headlight with sane defaults."""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [3.0, 4.0, 10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.25, 0.25, 0.25, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])


def computeCOM(body):
    """System COM position in world frame for either HyroSphere or LinearSphere."""
    if hasattr(body, 'phi'):  # HyroSphere
        U = body.U
        A = U * np.sqrt(1.0 / 8.0)
        b_base = [np.cross(A[i], A[(i + 1) % 4]) for i in range(4)]
        b_base = [bi / np.linalg.norm(bi) * body.t_len / 2.0 for bi in b_base]
        B = [np.dot(rotate_vec(U[i], body.phi[i]), b_base[i]) for i in range(4)]
        R = A + np.array(B)
    else:  # LinearSphere
        R = np.dot(np.diag(body.shifts), body.U)
    total_mass = float(np.sum(body.dot_masses) + body.mass)
    offset = np.sum(body.dot_masses[:, None] * R, axis=0) / total_mass
    return body.position + offset
