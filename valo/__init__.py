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

from . import meshes
from .app import Viewer
from .geometry import instance_matrices
from .lighting import Lighting
from .materials import SCHEMA, SHADINGS, Material
from .meshes import MeshData, Sculpt, library
from .render import Instances, OrbitCamera, Renderer, Spheres
from .ui import Editor
from .ui_style import ICON, mono_font
from .undo import Command, UndoStack

__all__ = ["ICON", "SCHEMA", "SHADINGS", "Command", "Editor", "Instances", "Lighting", "Material", "MeshData",
           "OrbitCamera", "Renderer", "Sculpt", "Spheres", "UndoStack", "Viewer", "instance_matrices", "library",
           "meshes", "mono_font"]
