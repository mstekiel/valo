"""valo.gpu: the draw engine (wgpu). Below the scene model: it knows meshes, materials and batches, not scenes.

    shaders.py    all WGSL (vertex, shading models, pick, highlight, post)
    material.py   Material (uniform buffer + parameter schema), SCHEMA, SHADINGS, colour conversion
    mesh.py       GpuMesh (vertex / index buffers)
    instances.py  Instances, Spheres, Batch (one instanced draw)
    renderer.py   Renderer (pipelines, targets, passes, pick), Pick, the shared device and renderer()
"""

from .instances import Batch, Instances, Spheres
from .material import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .mesh import GpuMesh
from .renderer import Pick, Renderer, get_device, renderer

__all__ = ["SCHEMA", "SHADINGS", "Batch", "GpuMesh", "Instances", "Material", "Pick", "Renderer", "Spheres",
           "get_device", "linear_to_srgb", "renderer", "srgb_to_linear"]
