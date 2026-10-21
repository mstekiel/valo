"""MeshData on the GPU: vertex and index buffers, and the vertex layout every pipeline reads."""

from __future__ import annotations

import numpy as np
import wgpu

from ..meshes.data import MeshData

__all__ = ["VERTEX_LAYOUT", "GpuMesh"]


VERTEX_LAYOUT = {
    "array_stride": 24, "step_mode": "vertex",
    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0},
                   {"format": "float32x3", "offset": 12, "shader_location": 1}],
}


class GpuMesh:
    """A MeshData uploaded to vertex and index buffers."""
    def __init__(self, device, data: MeshData):
        self.vb = device.create_buffer_with_data(data=data.vertices, usage=wgpu.BufferUsage.VERTEX)
        self.ib = device.create_buffer_with_data(data=data.indices, usage=wgpu.BufferUsage.INDEX)
        self.count = len(data.indices)
        self.radius = float(np.linalg.norm(data.positions, axis=1).max()) if len(data.positions) else 0.0
        p = data.positions if len(data.positions) else np.zeros((1, 3))
        self.bounds = (p.min(0).astype(float), p.max(0).astype(float))  # mesh-space box: (min xyz, max xyz)

    def draw(self, rpass, instances: int = 1, first_instance: int = 0):
        """Draw instances ``first_instance`` .. ``first_instance + instances - 1`` in the current render pass."""
        rpass.set_vertex_buffer(0, self.vb)
        rpass.set_index_buffer(self.ib, wgpu.IndexFormat.uint32)
        rpass.draw_indexed(self.count, instances, 0, 0, first_instance)
