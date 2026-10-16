"""Meshes: ``MeshData`` and the standard shapes (pure numpy, no GPU code).

Conventions: vertices are position xyz + unit normal xyz; triangles are counter-clockwise seen from
outside; Z-up; unit-sized and centred at the origin unless a function says otherwise.

Builders for new shapes: ``revolve`` (a (ρ, z) profile swept around Z), ``parametric`` (f(u, v) → xyz),
``heightfield`` (z = f(x, y)), plus ``transform`` and ``merge``. ``check`` reports what's wrong with a mesh.
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


def _profile_normals(prof: np.ndarray) -> np.ndarray:
    """(K, 2) outward (n_rho, n_z) for a (ρ, z) profile: neighbouring segments averaged; a point repeated
    twice in a row is a hard edge (each copy takes the normal of its own side)."""
    d = np.diff(prof, axis=0)
    length = np.linalg.norm(d, axis=1)
    seg = np.zeros_like(d)
    seg[length > 0] = np.c_[d[:, 1], -d[:, 0]][length > 0] / length[length > 0, None]
    n = np.zeros_like(prof)
    n[:-1] += seg  # segment k touches points k and k + 1
    n[1:] += seg
    norm = np.linalg.norm(n, axis=1)
    good = np.flatnonzero(norm > 0)
    if not len(good):
        raise ValueError("profile has no segment of non-zero length")
    nearest = good[np.abs(np.arange(len(n))[:, None] - good[None, :]).argmin(1)]  # e.g. a trailing repeat
    return n[nearest] / norm[nearest, None]


def revolve(profile, segments: int = 32) -> MeshData:
    """Sweep a profile around Z. ``profile`` is (K, 2) of (ρ, z), or (K, 4) with explicit (n_ρ, n_z).

    Go from the bottom of the axis to the top on the outside (start and end at ρ = 0 for a closed
    solid). Normals of a (K, 2) profile are computed; repeat a point to make a hard edge there.
    """
    prof = np.asarray(profile, float)
    if prof.shape[1] == 2:
        prof = np.c_[prof, _profile_normals(prof)]
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
    return revolve([(0, z0, 0, -1), (r, z0, 0, -1), (r, z0, 1, 0), (r, z1, 1, 0), (r, z1, 0, 1), (0, z1, 0, 1)],
                    segments)


def cone(segments: int = 32, r: float = 0.5, h: float = 1.0) -> MeshData:
    """Base (radius r) at z = −h/2, apex at z = +h/2."""
    z0, z1 = -h / 2, h / 2
    nr, nz = _slope_normal(r, z0, 0, z1)
    return revolve([(0, z0, 0, -1), (r, z0, 0, -1), (r, z0, nr, nz), (0, z1, nr, nz)], segments)


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

    return revolve(cap(-c, math.pi, math.pi / 2) + cap(c, math.pi / 2, 0), segments)


def arrow(shaft_r: float = 0.15, head_r: float = 0.5, head_len: float = 0.35, segments: int = 24) -> MeshData:
    """Tail at the origin, tip at z = 1, head diameter 1 (= 2·head_r): scale (w, w, length) to fit a vector."""
    zh = 1.0 - head_len
    nr, nz = _slope_normal(head_r, zh, 0, 1)
    return revolve([(0, 0, 0, -1), (shaft_r, 0, 0, -1), (shaft_r, 0, 1, 0), (shaft_r, zh, 1, 0),
                     (shaft_r, zh, 0, -1), (head_r, zh, 0, -1), (head_r, zh, nr, nz), (0, 1, nr, nz)], segments)


STANDARD = {"sphere": sphere, "icosphere": icosphere, "cube": cube, "cylinder": cylinder, "cone": cone,
            "torus": torus, "capsule": capsule, "arrow": arrow, "plane": plane}


# -- builders for new shapes


def parametric(f, nu: int = 64, nv: int = 64, u=(0.0, 1.0), v=(0.0, 1.0), flip: bool = False) -> MeshData:
    """Surface P = f(u, v) over a grid; ``f`` takes and returns numpy arrays (xyz stacked on the last axis,
    or a tuple of three arrays). Outward is ∂P/∂u × ∂P/∂v; pass ``flip`` if your surface faces the other way.
    """
    def evaluate(uu, vv):
        p = f(uu, vv)
        return np.stack(np.broadcast_arrays(*p), -1) if isinstance(p, tuple) else np.asarray(p, float)

    uu, vv = np.meshgrid(np.linspace(*u, nu + 1), np.linspace(*v, nv + 1), indexing="ij")
    hu, hv = 1e-4 * (u[1] - u[0]), 1e-4 * (v[1] - v[0])

    def normals(uu, vv):
        du = evaluate(uu + hu, vv) - evaluate(uu - hu, vv)
        dv = evaluate(uu, vv + hv) - evaluate(uu, vv - hv)
        return np.cross(du, dv)

    n = normals(uu, vv)
    flat = np.linalg.norm(n, axis=-1) < 1e-12 * max(hu * hv, 1e-300)  # poles: step a little inwards
    if flat.any():
        su = np.where(uu < (u[0] + u[1]) / 2, 1, -1) * 1e-3 * (u[1] - u[0])
        sv = np.where(vv < (v[0] + v[1]) / 2, 1, -1) * 1e-3 * (v[1] - v[0])
        n[flat] = normals(uu + su, vv + sv)[flat]
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    idx = _grid_indices(nu, nv).reshape(-1, 3)
    if flip:
        n, idx = -n, idx[:, ::-1]
    return _mesh(evaluate(uu, vv).reshape(-1, 3), n.reshape(-1, 3), idx)


def heightfield(f, size: float = 1.0, n: int = 64) -> MeshData:
    """Terrain z = f(x, y) over the square |x|, |y| ≤ size/2, facing +Z (one-sided)."""
    h = size / 2
    return parametric(lambda x, y: (x, y, f(x, y)), n, n, (-h, h), (-h, h))


# -- diagnostics


@dataclass
class MeshReport:
    vertices: int
    triangles: int
    degenerate: int  # zero-area triangles (harmless seams and poles, but wasted work)
    bad_normals: int  # not unit length
    flipped: int  # wound against their vertex normals (shows up as holes: back faces are culled)
    watertight: bool  # closed and consistently wound, once vertices are welded by position
    volume: float | None  # enclosed volume, when watertight

    @property
    def ok(self) -> bool:
        return self.bad_normals == 0 and self.flipped == 0

    def lines(self) -> list[str]:
        out = [f"{self.vertices} vertices, {self.triangles} triangles ({self.degenerate} degenerate)",
               f"normals: {'ok' if not self.bad_normals else f'{self.bad_normals} not unit length'}",
               f"winding: {'ok' if not self.flipped else f'{self.flipped} triangles flipped'}"]
        out.append(f"watertight, volume {self.volume:.4g}" if self.watertight else "open (not watertight)")
        return out


def check(mesh: MeshData) -> MeshReport:
    """What's wrong with ``mesh``: the checks the tests and the MeshSculptor panel share."""
    p = mesh.positions.astype(float)
    n = mesh.normals.astype(float)
    t = mesh.indices.reshape(-1, 3).astype(np.int64)
    bad_normals = int((~(np.abs(np.linalg.norm(n, axis=1) - 1) <= 1e-3)).sum())  # NaN counts as bad
    if not len(t):
        return MeshReport(len(p), 0, 0, bad_normals, 0, False, None)

    face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    scale = float(np.ptp(p, axis=0).max()) or 1.0
    real = np.linalg.norm(face, axis=1) > 1e-10 * scale**2
    flipped = int((np.einsum("ij,ij->i", face, n[t].mean(1))[real] <= 0).sum())

    # Weld vertices by position (seams and hard edges duplicate them), then every edge of a closed,
    # consistently wound surface appears exactly once in each direction.
    _, weld = np.unique(np.round(p / scale, 6), axis=0, return_inverse=True)
    w = weld.ravel()[t]
    distinct = (w[:, 0] != w[:, 1]) & (w[:, 1] != w[:, 2]) & (w[:, 2] != w[:, 0])
    w = w[distinct & real]
    edges = np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]])
    key = edges[:, 0] * (len(p) + 1) + edges[:, 1]
    rev = edges[:, 1] * (len(p) + 1) + edges[:, 0]
    keys, counts = np.unique(key, return_counts=True)
    watertight = bool(len(w)) and (counts == 1).all() and np.isin(rev, keys).all()
    volume = None
    if watertight:
        q = p[t[distinct & real]]
        volume = float(np.einsum("ij,ij->i", q[:, 0], np.cross(q[:, 1], q[:, 2])).sum() / 6)
    return MeshReport(len(p), len(t), int((~real).sum()), bad_normals, flipped, watertight, volume)
