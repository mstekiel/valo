"""Pure-numpy geometry: matrices (math convention, Z-up) and meshes. No GPU code here."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def normal_matrix(model):
    n = np.eye(4, dtype=np.float32)
    n[:3, :3] = np.linalg.inv(model[:3, :3]).T
    return n


def pack_mat4(m):
    """numpy row-major math matrix → WGSL column-major mat4x4f bytes (the ONLY transpose)."""
    return np.ascontiguousarray(np.swapaxes(np.asarray(m, np.float32), -1, -2))


@dataclass
class MeshData:
    vertices: np.ndarray  # (N, 6) float32: position xyz, normal xyz
    indices: np.ndarray  # (M,) uint32


def _grid_indices(nu, nv):
    idx = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = a + nv + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return np.array(idx, np.uint32)


def sphere_mesh(n_lat=48, n_lon=96, r=0.5):
    v = []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        for j in range(n_lon + 1):
            ph = 2 * math.pi * j / n_lon
            n = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            v.append((*(r * c for c in n), *n))
    return MeshData(np.array(v, np.float32), _grid_indices(n_lat, n_lon))
