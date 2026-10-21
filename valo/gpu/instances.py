"""Instanced draws: ``Instances`` (one mesh, one material, N model matrices in a storage buffer), ``Spheres``, and the
``Batch`` a scene's ``batches()`` returns to the renderer."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import wgpu

from ..math import instance_matrices, pack_mat4
from .material import Material
from .mesh import GpuMesh

__all__ = ["Batch", "Instances", "Spheres"]


class Batch(NamedTuple):
    """What a scene's ``batches()`` yields: one instanced draw. ``source`` is the Instances (for picking/info)."""

    material: Material
    mesh: GpuMesh
    bind_group: object
    count: int
    source: object = None


class Instances:
    """Instances of one mesh sharing one material: a grow-only storage buffer of model matrices.

    ``batch()`` is the Batch the renderer draws. ``label`` names the object in the Info panel, and
    ``info(i) -> dict`` (optional) adds scene-specific lines for instance ``i``. ``owner`` (optional) is the scene
    object the instances belong to: a click on any of them selects, and highlights, the whole object.
    """

    def __init__(self, device: wgpu.GPUDevice, instance_layout, mesh: GpuMesh, material: Material,
                 label: str | None = None, info: Callable[[int], dict] | None = None):
        self.device, self.layout, self.mesh, self.material = device, instance_layout, mesh, material
        self.label, self.info, self.owner = label, info, None
        self.count = self.capacity = 0  # capacity: instances the buffer can hold
        self.buffer = self.bind_group = None
        self.set(np.zeros((0, 4, 4)))

    def set(self, matrices) -> None:
        """Upload (N, 4, 4) math-convention model matrices, e.g. from ``geometry.instance_matrices``."""
        matrices = np.asarray(matrices, np.float32).reshape(-1, 4, 4)
        self.matrices = matrices.copy()  # CPU copy: picking, info and centring read it
        self.count = len(matrices)
        data = np.zeros((max(self.count, 1), 4, 4), np.float32)  # buffers can't be empty
        data[: self.count] = pack_mat4(matrices)
        if self.buffer is None or self.capacity < len(data):  # grow only; shrinking reuses the buffer
            self.capacity = max(len(data), 2 * self.capacity)
            self.buffer = self.device.create_buffer(
                size=self.capacity * 64, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            self.bind_group = self.device.create_bind_group(
                layout=self.layout, entries=[{"binding": 0, "resource": {"buffer": self.buffer}}])
        self.device.queue.write_buffer(self.buffer, 0, data)

    def batch(self) -> Batch:
        """The Batch a scene's ``batches()`` returns for these instances."""
        return Batch(self.material, self.mesh, self.bind_group, self.count, self)


class Spheres(Instances):
    """Instances of the unit-diameter sphere, placed by centre and radius."""

    def set(self, centers, radii=0.5) -> None:
        """Upload centres (N, 3) and radii (one value or N values)."""
        centers = np.asarray(centers, np.float32).reshape(-1, 3)
        diameters = 2 * np.broadcast_to(np.asarray(radii, np.float32), len(centers))[:, None]
        super().set(instance_matrices(centers, diameters))
