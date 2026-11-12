"""
motion/wedge.py -- the "wedge-only" control for the leak question (H11). numpy only.

Question: in scenario B, `noocc` (map without obstacles) already gets most of the gain. Is that real geometry, or only WHERE
the robot stood (its camera cone leaves a known / unknown pattern that depends on the robot placement)?
Control: replace the crop by the robot's camera CONE alone in an empty world: known (free) inside the cone, unknown outside,
no walls, no shadows. If a model with this crop beats `hist` clearly, robot placement leaks the answer.

Camera (from the robot's depth_camera.xacro): horizontal FOV 1.27409 rad (73 degrees), clip far 8.0 m, two identical cameras
0.06 m apart looking forward (treated as one cone from the robot position; the mount offset of a few cm is ignored).
The robot pose in the pedestrian frame is (robot, robot_yaw) from motion/data.py.
"""
import numpy as np

from .data import RES, X0, Y0

FOV, RANGE = 1.27409, 8.0


def _centres():
    c = X0 + (np.arange(80) + 0.5) * RES
    r = Y0 + (np.arange(80) + 0.5) * RES
    X, Y = np.meshgrid(c, r)
    return X, Y


def wedge_masks(robot_xy, robot_yaw, fov=FOV, rng=RANGE, chunk=500):
    """(N,80,80) bool: cells inside the camera cone."""
    X, Y = _centres()
    out = np.zeros((len(robot_xy), 80, 80), bool)
    for i in range(0, len(robot_xy), chunk):
        sl = slice(i, i + chunk)
        dx, dy = X[None] - robot_xy[sl, 0, None, None], Y[None] - robot_xy[sl, 1, None, None]
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - robot_yaw[sl, None, None]
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        out[sl] = (np.abs(ang) <= fov / 2) & (dist <= rng)
    return out


def wedge_crops(robot_xy, robot_yaw, **kw):
    """(N,80,80) int8 crop: 0 (free) inside the cone, -1 (unknown) outside. No occupied cells."""
    return np.where(wedge_masks(robot_xy, robot_yaw, **kw), 0, -1).astype(np.int8)


def coverage(crop, wmask, usable):
    """Sanity check of the camera model against the REAL crops (usable samples):
    inside_wedge = share of the really known cells that lie inside the cone (should be close to 1 if the crop is an
                   instant view; clearly lower means the real map also contains older views or the geometry is off);
    wedge_seen   = share of the cone cells that are known in the real crop (the rest is shadow or outside the crop)."""
    known = crop[usable] != -1
    w = wmask[usable]
    inside = (known & w).sum() / max(known.sum(), 1)
    seen = (known & w).sum() / max(w.sum(), 1)
    return float(inside), float(seen)