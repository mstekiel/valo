"""valo: an interactive scientific 3D rendering engine (wgpu + glfw + Dear ImGui).

An app builds a scene (anything with ``batches()``) from the Viewer's renderer and shows it::

    viewer = Viewer()
    gold = viewer.renderer.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    balls = viewer.renderer.spheres(gold)
    balls.set(centers, radii=0.4)
    viewer.show(SimpleNamespace(batches=lambda: [balls.batch()]), Editor())
    viewer.run()

See ``examples/`` for complete apps.
"""

from .app import Viewer
from .materials import SCHEMA, SHADINGS, Material
from .render import OrbitCamera, Renderer, Spheres
from .ui import Editor
from .undo import Command, UndoStack

__all__ = ["SCHEMA", "SHADINGS", "Command", "Editor", "Material", "OrbitCamera", "Renderer", "Spheres",
           "UndoStack", "Viewer"]
