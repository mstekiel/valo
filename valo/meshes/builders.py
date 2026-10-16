"""Builders for new shapes: ``revolve`` (a (ρ, z) profile swept around Z), ``parametric`` (P(u, v)) and
``heightfield`` (z = f(x, y)). Sculpt files use these."""

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


def heightfield(f, size: float = 1.0, n: int = 64) -> MeshData:
    """Terrain z = f(x, y) over the square |x|, |y| ≤ size/2, facing +Z (one-sided)."""
    h = size / 2
    return parametric(lambda x, y: (x, y, f(x, y)), n, n, (-h, h), (-h, h))
