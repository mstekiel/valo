"""valo: an interactive scientific 3D rendering engine (wgpu + glfw + Dear ImGui).

An app builds a scene (anything with ``batches()``) from the Viewer's renderer and shows it::

    viewer = Viewer()
    gold = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    balls = viewer.renderer.spheres(gold)
    balls.set(centers, radii=0.4)
    arrows = viewer.renderer.instances(meshes.arrow(), gold)
    arrows.set(instance_matrices(origins, np.c_[w, w, lengths], z_axis=vectors))
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch(), arrows.batch()]), Editor())
    viewer.run()

See ``examples/`` for complete apps.
"""

from . import library, meshes
from .app import Viewer
from .geometry import instance_matrices
from .materials import SCHEMA, SHADINGS, Material
from .meshes import MeshData
from .render import Instances, OrbitCamera, Renderer, Spheres
from .sculpt import Sculpt
from .ui import Editor
from .undo import Command, UndoStack

__all__ = ["SCHEMA", "SHADINGS", "Command", "Editor", "Instances", "Material", "MeshData", "OrbitCamera",
           "Renderer", "Sculpt", "Spheres", "UndoStack", "Viewer", "instance_matrices", "library", "meshes"]
