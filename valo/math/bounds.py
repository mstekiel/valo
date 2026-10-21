"""Bounding boxes (pure numpy): box corners, and the box around points in a given frame as a box matrix."""

from __future__ import annotations

import numpy as np

from .transforms import scale, translate

__all__ = ["bounding_box", "box_corners"]


def box_corners(lo, hi) -> np.ndarray:
    """The 8 corners (8, 3) of the axis-aligned box from ``lo`` to ``hi``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def bounding_box(points, frame=None) -> np.ndarray | None:
    """The box around world ``points`` (M, 3), aligned with ``frame`` (a 4x4 model matrix; default the world axes),
    as a box matrix: it maps the unit cube [-0.5, 0.5]^3 onto the box in world space. None for no points."""
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points):
        return None
    frame = np.eye(4) if frame is None else np.asarray(frame, float)
    local = (np.linalg.inv(frame) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    lo, hi = local.min(0), local.max(0)
    return frame @ translate(*((lo + hi) / 2)) @ scale(*(hi - lo))
