"""MeshData: vertices (position + unit normal) and triangle indices. Conventions for every mesh in valo:
triangles counter-clockwise seen from outside, Z-up, unit-sized and centred at the origin unless stated."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)  # identity hash: the renderer caches one GPU upload per MeshData
class MeshData:
    """A triangle mesh: per-vertex position and unit normal, and uint32 triangle indices."""
    vertices: np.ndarray  # (N, 6) float32: position xyz, normal xyz
    indices: np.ndarray  # (M,) uint32

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) vertex positions (a view into ``vertices``)."""
        return self.vertices[:, :3]

    @property
    def normals(self) -> np.ndarray:
        """(N, 3) unit vertex normals (a view into ``vertices``)."""
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
