"""Meshes: ``MeshData`` and the standard shapes (pure numpy, no GPU code).

Conventions: vertices are position xyz + unit normal xyz; triangles are counter-clockwise seen from
outside; Z-up; unit-sized and centred at the origin unless a function says otherwise.

Most round shapes are surfaces of revolution: a profile of (ρ, z, normal ρ, normal z) points, traversed
from the bottom of the axis to the top on the outside, swept around Z. Points repeated with a different
normal give a hard edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)  # identity hash: the renderer caches one GPU upload per MeshData
class MeshData:
    vertices: np.ndarray  # (N, 6) float32: position xyz, normal xyz
    indices: np.ndarray  # (M,) uint32

    @property
    def positions(self) -> np.ndarray:
        return self.vertices[:, :3]

    @property
    def normals(self) -> np.ndarray:
        return self.vertices[:, 3:]


def _mesh(positions, normals, indices) -> MeshData:
    v = np.concatenate([np.asarray(positions, np.float32), np.asarray(normals, np.float32)], axis=1)
    return MeshData(v.astype(np.float32), np.asarray(indices, np.uint32).ravel())


def _grid_indices(nu, nv):
    """Two triangles per cell of a (nu+1) × (nv+1) vertex grid; outward when d/du × d/dv points out."""
    idx = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = a + nv + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return np.array(idx, np.uint32)


# -- helpers


def transform(mesh: MeshData, m) -> MeshData:
    """``mesh`` with positions mapped by the 4×4 matrix ``m`` and normals by its inverse-transpose."""
    m = np.asarray(m, np.float64)
    a = m[:3, :3]
    p = mesh.positions @ a.T + m[:3, 3]
    n = mesh.normals @ np.linalg.inv(a)  # == (inv(a).T @ n.T).T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    idx = mesh.indices.reshape(-1, 3)
    if np.linalg.det(a) < 0:  # a mirror turns the triangles inside out
        idx = idx[:, ::-1]
    return _mesh(p, n, idx)


def merge(*meshes: MeshData) -> MeshData:
    """One mesh containing all of ``meshes``."""
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return MeshData(np.concatenate([m.vertices for m in meshes]),
                    np.concatenate([m.indices + o for m, o in zip(meshes, offsets, strict=True)]).astype(np.uint32))


def _quad(center, normal, u, v) -> MeshData:
    """A rectangle c ± u ± v facing ``normal`` (== u × v direction)."""
    c, u, v = (np.asarray(x, float) for x in (center, u, v))
    return _mesh([c - u - v, c + u - v, c + u + v, c - u + v], [normal] * 4, [0, 1, 2, 0, 2, 3])


def _revolve(profile, segments: int) -> MeshData:
    prof = np.asarray(profile, float)  # (K, 4): rho, z, n_rho, n_z
    a = np.linspace(0, 2 * math.pi, segments + 1)[:, None]  # seam duplicated for a clean grid
    cos, sin = np.cos(a), np.sin(a)
    grid = (segments + 1, len(prof))
    p = np.stack([prof[:, 0] * cos, prof[:, 0] * sin, np.broadcast_to(prof[:, 1], grid)], -1)
    n = np.stack([prof[:, 2] * cos, prof[:, 2] * sin, np.broadcast_to(prof[:, 3], grid)], -1)
    return _mesh(p.reshape(-1, 3), n.reshape(-1, 3), _grid_indices(segments, len(prof) - 1))


def _slope_normal(r0, z0, r1, z1):
    """Outward (n_rho, n_z) of the profile segment (r0, z0) → (r1, z1)."""
    dr, dz = r1 - r0, z1 - z0
    length = math.hypot(dr, dz)
    return dz / length, -dr / length


# -- standard meshes


def sphere(n_lat: int = 48, n_lon: int = 96, r: float = 0.5) -> MeshData:
    v = []
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat
        for j in range(n_lon + 1):
            ph = 2 * math.pi * j / n_lon
            n = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            v.append((*(r * c for c in n), *n))
    return MeshData(np.array(v, np.float32), _grid_indices(n_lat, n_lon))


def icosphere(subdivisions: int = 2, r: float = 0.5) -> MeshData:
    """A sphere of near-equal triangles: a subdivided icosahedron (20·4^subdivisions faces)."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    cache: dict[tuple[int, int], int] = {}  # edge → midpoint vertex (indices are unique across levels)

    def mid(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            p = verts[i] + verts[j]
            verts.append(p / np.linalg.norm(p))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdivisions):
        faces = [f for a, b, c in faces for f in ((a, mid(a, b), mid(c, a)), (b, mid(b, c), mid(a, b)),
                                                   (c, mid(c, a), mid(b, c)), (mid(a, b), mid(b, c), mid(c, a)))]
    n = np.array(verts)
    idx = np.array(faces)
    outward = np.einsum("ij,ij->i", np.cross(n[idx[:, 1]] - n[idx[:, 0]], n[idx[:, 2]] - n[idx[:, 0]]), n[idx[:, 0]])
    idx[outward < 0] = idx[outward < 0][:, ::-1]  # one consistent winding, whatever the table's order
    return _mesh(n * r, n, idx)


def cube(size: float = 1.0) -> MeshData:
    """Axis-aligned cube with flat normals (4 vertices per face)."""
    h = size / 2
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            u = np.roll(n, 1) * h  # u × v == n·h² for both signs
            v = np.roll(n, 2) * h * sign
            faces.append(_quad(n * h, n, u, v))
    return merge(*faces)


def plane(size: float = 1.0) -> MeshData:
    """Square in the XY plane, double-sided: a +Z face and a −Z face, so back-face culling never hides it."""
    h = size / 2
    return merge(_quad((0, 0, 0), (0, 0, 1), (h, 0, 0), (0, h, 0)),
                 _quad((0, 0, 0), (0, 0, -1), (0, h, 0), (h, 0, 0)))


def cylinder(segments: int = 32, r: float = 0.5, h: float = 1.0) -> MeshData:
    """Along Z, z ∈ [−h/2, h/2], with flat caps."""
    z0, z1 = -h / 2, h / 2
    return _revolve([(0, z0, 0, -1), (r, z0, 0, -1), (r, z0, 1, 0), (r, z1, 1, 0), (r, z1, 0, 1), (0, z1, 0, 1)],
                    segments)


def cone(segments: int = 32, r: float = 0.5, h: float = 1.0) -> MeshData:
    """Base (radius r) at z = −h/2, apex at z = +h/2."""
    z0, z1 = -h / 2, h / 2
    nr, nz = _slope_normal(r, z0, 0, z1)
    return _revolve([(0, z0, 0, -1), (r, z0, 0, -1), (r, z0, nr, nz), (0, z1, nr, nz)], segments)


def torus(major: float = 0.35, minor: float = 0.15, nu: int = 48, nv: int = 24) -> MeshData:
    """Ring in the XY plane: tube of radius ``minor`` around a circle of radius ``major``."""
    a = np.linspace(0, 2 * math.pi, nu + 1)[:, None]  # around Z
    b = np.linspace(0, 2 * math.pi, nv + 1)[None, :]  # around the tube
    n = np.stack(np.broadcast_arrays(np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)), -1)
    p = n * minor + np.stack(np.broadcast_arrays(major * np.cos(a), major * np.sin(a), 0 * b), -1)
    return _mesh(p.reshape(-1, 3), n.reshape(-1, 3), _grid_indices(nu, nv))


def capsule(r: float = 0.25, length: float = 1.0, segments: int = 32, rings: int = 8) -> MeshData:
    """Along Z, total length ``length`` including the two hemispherical ends."""
    c = max(length / 2 - r, 0.0)  # centre of each hemisphere
    def cap(z, th0, th1):  # polar angle th measured from +Z
        return [(r * math.sin(t), z + r * math.cos(t), math.sin(t), math.cos(t))
                for t in np.linspace(th0, th1, rings + 1)]

    return _revolve(cap(-c, math.pi, math.pi / 2) + cap(c, math.pi / 2, 0), segments)


def arrow(shaft_r: float = 0.15, head_r: float = 0.5, head_len: float = 0.35, segments: int = 24) -> MeshData:
    """Tail at the origin, tip at z = 1, head diameter 1 (= 2·head_r): scale (w, w, length) to fit a vector."""
    zh = 1.0 - head_len
    nr, nz = _slope_normal(head_r, zh, 0, 1)
    return _revolve([(0, 0, 0, -1), (shaft_r, 0, 0, -1), (shaft_r, 0, 1, 0), (shaft_r, zh, 1, 0),
                     (shaft_r, zh, 0, -1), (head_r, zh, 0, -1), (head_r, zh, nr, nz), (0, 1, nr, nz)], segments)


STANDARD = {"sphere": sphere, "icosphere": icosphere, "cube": cube, "cylinder": cylinder, "cone": cone,
            "torus": torus, "capsule": capsule, "arrow": arrow, "plane": plane}
