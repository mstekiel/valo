"""Builders for new shapes: ``revolve`` (a (ρ, z) profile swept around Z), ``parametric`` (P(u, v)),
``heightfield`` (z = f(x, y)) and ``tube`` (a circle swept along a 3D path). Shape files use these."""

from __future__ import annotations

import math

import numpy as np

from .data import MeshData, _grid_indices, _mesh


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


def _path_frames(path: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit tangents and a rotation-minimising frame (normal, binormal) at each point of ``path`` (K, 3), by the
    double-reflection method; for a closed path the leftover twist is spread evenly along it."""
    k = len(path)
    if closed:
        t = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    else:
        t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    helper = np.eye(3)[np.argmin(np.abs(t[0]))]  # the axis least parallel to the first tangent
    n = np.zeros_like(path)
    n[0] = np.cross(t[0], helper)
    n[0] /= np.linalg.norm(n[0])
    for i in range(k - 1):
        v1 = path[i + 1] - path[i]
        c1 = v1 @ v1
        nl = n[i] - (2 / c1) * (v1 @ n[i]) * v1 if c1 > 0 else n[i]  # reflect along the chord
        tl = t[i] - (2 / c1) * (v1 @ t[i]) * v1 if c1 > 0 else t[i]
        v2 = t[i + 1] - tl
        c2 = v2 @ v2
        n[i + 1] = nl - (2 / c2) * (v2 @ nl) * v2 if c2 > 1e-24 else nl  # then onto the next tangent
    if closed:  # the frame carried once round doesn't meet itself: untwist by a share of the gap per step
        v1 = path[0] - path[-1]
        c1 = v1 @ v1
        nl = n[-1] - (2 / c1) * (v1 @ n[-1]) * v1 if c1 > 0 else n[-1]
        tl = t[-1] - (2 / c1) * (v1 @ t[-1]) * v1 if c1 > 0 else t[-1]
        v2 = t[0] - tl
        c2 = v2 @ v2
        end = nl - (2 / c2) * (v2 @ nl) * v2 if c2 > 1e-24 else nl
        gap = math.atan2(np.cross(end, n[0]) @ t[0], end @ n[0])
        share = gap * np.arange(k)[:, None] / k
        b = np.cross(t, n)
        n = np.cos(share) * n + np.sin(share) * b
    n -= (n * t).sum(1, keepdims=True) * t  # keep exactly perpendicular
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return t, n, np.cross(t, n)


def _cap(center, ring: np.ndarray, normal) -> MeshData:
    """A flat disc closing ``ring`` (M, 3), facing ``normal``: a fan of triangles around ``center``."""
    m = len(ring)
    idx = np.array([[0, 1 + j, 1 + (j + 1) % m] for j in range(m)])
    p = np.vstack([center, ring])
    a, b, c = p[idx[0]]
    if np.cross(b - a, c - a) @ normal < 0:
        idx = idx[:, ::-1]
    return _mesh(p, np.broadcast_to(normal, p.shape), idx)


def tube(path, radius=0.1, segments: int = 16, closed: bool = False, caps: bool = True) -> MeshData:
    """A tube (a generalised cylinder) of ``radius`` around the polyline ``path`` (K, 3), K ≥ 2.

    ``radius`` is one value or one per point (K,). The circle's frame is rotation-minimising, so the tube doesn't
    twist along curves. ``closed`` joins the last point back to the first (a ring; no caps), otherwise ``caps``
    closes the ends with flat discs. Smooth curves need enough points: a straight line needs only two.
    """
    path = np.asarray(path, float).reshape(-1, 3)
    if len(path) < 2:
        raise ValueError("a tube needs a path of at least 2 points")
    if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) == 0):
        raise ValueError("the path repeats a point; remove the duplicate")
    if closed and np.allclose(path[0], path[-1]):
        path = path[:-1]  # the loop is closed for you
    k = len(path)
    r = np.broadcast_to(np.asarray(radius, float), (k,)).copy()
    if np.any(r <= 0):
        raise ValueError("radius must be > 0")
    t, n, b = _path_frames(path, closed)
    ring = np.r_[np.arange(k), 0] if closed else np.arange(k)  # vertex rows (the loop repeats its first)
    if closed:
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[path, path[:1]], axis=0), axis=1))]
        dr = np.gradient(np.r_[r, r[:1]], s)[:-1] if k > 2 else np.zeros(k)
    else:
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        dr = np.gradient(r, s) if k > 2 else np.full(k, (r[-1] - r[0]) / s[-1])
    a = np.linspace(0, 2 * math.pi, segments + 1)[:, None, None]  # seam duplicated for a clean grid
    radial = np.cos(a) * n[ring] + np.sin(a) * b[ring]  # (segments + 1, rows, 3)
    p = path[ring] + r[ring, None] * radial
    normal = radial - dr[ring, None] * t[ring]  # a widening tube faces backwards a little
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    side = _mesh(p.reshape(-1, 3), normal.reshape(-1, 3), _grid_indices(segments, len(ring) - 1))
    if closed or not caps:
        return side
    from .ops import merge

    start = _cap(path[0], p[:-1, 0], -t[0])
    end = _cap(path[-1], p[:-1, -1], t[-1])
    return merge(side, start, end)


def heightfield(f, size: float = 1.0, n: int = 64) -> MeshData:
    """Terrain z = f(x, y) over the square |x|, |y| ≤ size/2, facing +Z (one-sided)."""
    h = size / 2
    return parametric(lambda x, y: (x, y, f(x, y)), n, n, (-h, h), (-h, h))
