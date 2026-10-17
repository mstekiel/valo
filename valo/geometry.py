"""Pure-numpy geometry: matrices (math convention, Z-up) and meshes. No GPU code here."""

from __future__ import annotations

import math

import numpy as np


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(sx, sy=None, sz=None):
    return np.diag([sx, sx if sy is None else sy, sx if sz is None else sz, 1.0]).astype(np.float32)


def rotate(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = (target - eye) / np.linalg.norm(target - eye)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m.astype(np.float32)


def perspective_reversed_z(fovy_deg, aspect, near=0.05):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1], m[2, 3], m[3, 2] = f / aspect, f, near, -1.0
    return m


def orthographic_reversed_z(height, aspect, near, far):
    """Parallel projection showing ``height`` world units vertically; reversed-Z: depth 1 at near, 0 at far."""
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = 2.0 / (height * aspect), 2.0 / height
    m[2, 2], m[2, 3], m[3, 3] = 1.0 / (far - near), far / (far - near), 1.0
    return m


def normal_matrix(model):
    n = np.eye(4, dtype=np.float32)
    n[:3, :3] = np.linalg.inv(model[:3, :3]).T
    return n


def pack_mat4(m):
    """numpy row-major math matrix → WGSL column-major mat4x4f bytes (the ONLY transpose)."""
    return np.ascontiguousarray(np.swapaxes(np.asarray(m, np.float32), -1, -2))


def instance_matrices(positions, scale=1.0, z_axis=None) -> np.ndarray:
    """(N, 4, 4) model matrices M = T · R · S, one per position.

    ``scale`` broadcasts against (N, 3): a scalar, per-axis (3,), per-instance (N, 1) or (N, 3).
    ``z_axis`` (3,) or (N, 3): R turns the mesh's +Z onto that direction (lengths are ignored; use the
    scale for them). For arrows along vectors v: ``instance_matrices(origins, np.c_[w, w, |v|], v)``.
    """
    p = np.asarray(positions, float).reshape(-1, 3)
    n = len(p)
    m = np.zeros((n, 4, 4))
    m[:, :3, :3] = rotations_z_to(np.broadcast_to(z_axis, (n, 3))) if z_axis is not None else np.eye(3)
    m[:, :3, :3] *= np.broadcast_to(np.asarray(scale, float), (n, 3))[:, None, :]  # scale the columns
    m[:, :3, 3] = p
    m[:, 3, 3] = 1.0
    return m.astype(np.float32)


def rotations_z_to(directions) -> np.ndarray:
    """(N, 3, 3) shortest-arc rotations taking +Z onto each direction (zero vectors → identity)."""
    d = np.asarray(directions, float).reshape(-1, 3)
    length = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.where(length > 1e-12, d / np.where(length > 0, length, 1), (0.0, 0.0, 1.0))
    x, y, c = d[:, 0], d[:, 1], d[:, 2]
    k = np.zeros((len(d), 3, 3))  # cross-product matrix of +Z × d = (-y, x, 0)
    k[:, 0, 2], k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = x, y, -x, -y
    flip = c < -1 + 1e-9  # straight down: any half-turn about a horizontal axis
    r = np.eye(3) + k + k @ k / np.where(flip, 1.0, 1.0 + c)[:, None, None]
    r[flip] = np.diag([1.0, -1.0, -1.0])
    return r
