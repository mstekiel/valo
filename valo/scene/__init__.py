"""valo.scene: the scene data model (Blender's "data"): a tree of nodes with shapes, transforms and materials.

    camera.py  Camera (the 3D view: orbit, projection, fitting), PROJECTIONS, VIEWS
    shape.py   Shape (a function + parameters → MeshData), resolve, reference
    nodes.py   Transform, Node, Group, ProceduralGroup, MeshObject, InstancedObject
    scene.py   Scene (the tree + material library, batches(), settings()), node_from_settings, unique_name
"""

from .camera import Camera
from .nodes import Group, InstancedObject, MeshObject, Node, ProceduralGroup, Transform
from .scene import Scene, node_from_settings, unique_name
from .shape import Shape, reference, resolve

__all__ = ["Camera", "Group", "InstancedObject", "MeshObject", "Node", "ProceduralGroup", "Scene", "Shape", "Transform",
           "node_from_settings", "reference", "resolve", "unique_name"]
