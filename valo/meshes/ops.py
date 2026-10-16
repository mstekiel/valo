"""Operations on meshes: MeshData in, MeshData out."""

from __future__ import annotations

import numpy as np

from .data import MeshData, _mesh


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
