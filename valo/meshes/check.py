"""Diagnostics: ``check(mesh)`` reports what is wrong with a mesh (shared by the tests and MeshSculptor)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MeshData


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
