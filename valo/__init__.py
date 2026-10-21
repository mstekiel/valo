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

Or, at a higher level, a ``valo.Scene``: a tree of named objects, each a shape function with parameters, a
transform and a material from the scene's library (see ``valo/scene.py``).

See ``examples/`` for complete apps.
"""

from . import meshes
from .app import Viewer
from .config import rc, rc_context
from .geometry import instance_matrices
from .lighting import Lighting
from .materials import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .meshes import (
           MeshData,
           Sculpt,  # noqa: F401 - deprecated alias of ShapeFile, removed at the end of the refactor
           ShapeFile,
           library,
)
from .offline import plain, py_literal, render, save_png
from .render import (
           Instances,
           OrbitCamera,  # noqa: F401 - deprecated name of Camera, removed at the end of the refactor
           Renderer,
           Spheres,
           renderer,
)
from .scene import (
           Camera,
           Group,
           InstancedObject,
           MeshObject,
           ProceduralGroup,
           Scene,
           Shape,
           Transform,
           node_from_settings,
)
from .scene_editor import SceneEditor
from .studio import StudioEditor
from .ui import Editor
from .ui_style import ICON, mono_font
from .undo import Command, UndoStack

__all__ = ["ICON", "SCHEMA", "SHADINGS", "Camera", "Command", "Editor", "Group", "InstancedObject", "Instances",
           "Lighting", "Material", "MeshData", "MeshObject", "ProceduralGroup", "Renderer", "Scene", "SceneEditor",
           "StudioEditor", "ShapeFile", "Shape", "Spheres", "Transform", "UndoStack", "Viewer", "instance_matrices",
           "library", "linear_to_srgb", "meshes", "mono_font", "node_from_settings", "plain", "py_literal", "rc",
           "rc_context", "render", "renderer", "save_png", "srgb_to_linear"]
