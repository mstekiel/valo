"""The scene: a tree of objects, each with a shape (a function and its parameters), a transform and a material.

    scene = Scene()
    gold = scene.material("Gold", "pbr", base_color=(1.0, 0.71, 0.29), metallic=1.0)
    shapes = scene.add(Group("Shapes"))
    scene.add(MeshObject("Ball", Shape("valo.meshes.primitives:sphere", {"r": 0.4}), "Gold"), parent=shapes)
    scene.add(InstancedObject("Arrows", Shape(meshes.arrow), "Gold", instances=matrices))
    viewer.show(scene, SceneEditor())

- **Node types.** ``Group`` (passes its transform and visibility to its children), ``ProceduralGroup`` (a group whose
  children a builder function regenerates from the group's parameters), ``MeshObject`` (one mesh at the node's world
  transform) and ``InstancedObject`` (one mesh at world · instance_i for each instance matrix).
- **World transform.** world = parent world · own ``Transform`` (scale, then rotate x, y, z in degrees, then
  translate). A node is drawn only when it and all its ancestors are visible.
- **Shapes.** A ``Shape`` is a function (a callable, a ``"module:function"`` path, or a shape-file path) plus its
  parameters; ``meshes.params.shape_params`` reads the editable ones from the signature.
- **Materials.** The scene owns a library of named materials; objects refer to one by name, so several can share it.
- **Names.** Node and material names are unique within a scene (a " 2" suffix is added when needed).
- **Drawing.** The renderer draws ``scene.batches()``; each object owns a ``valo.Instances`` whose ``owner`` is it.
"""
from __future__ import annotations

import numpy as np

from ..gpu.instances import Instances
from ..gpu.material import Material
from ..gpu.renderer import Renderer
from ..gpu.renderer import renderer as shared_renderer
from .nodes import Group, InstancedObject, MeshObject, Node, ProceduralGroup, Transform
from .shape import Shape


def unique_name(name: str, taken) -> str:
    """``name``, or ``name 2``, ``name 3``, ... whichever is not in ``taken``."""
    if name not in taken:
        return name
    i = 2
    while f"{name} {i}" in taken:
        i += 1
    return f"{name} {i}"


class Scene:
    """A tree of nodes under ``root`` and a library of named materials; draws its visible objects."""

    def __init__(self, renderer: Renderer | None = None):
        """``renderer`` owns the GPU objects (default: the shared ``valo.renderer()``)."""
        self.renderer = renderer or shared_renderer()
        self.materials: dict[str, Material] = {}
        self.root = Group("Scene")
        self.root.scene = self
        self.overlays: list[Instances] = []  # drawn after the objects, not in the tree, not saved (editor helpers)

    # -- materials
    def material(self, name: str, shading: str = "pbr", **params) -> Material:
        """Add a new material to the library (the name is made unique) and return it."""
        name = unique_name(name, self.materials)
        self.materials[name] = self.renderer.material(name, shading, **params)
        return self.materials[name]

    def add_material(self, material: Material) -> Material:
        """Put an existing material into the library under a unique name."""
        material.name = unique_name(material.name, self.materials)
        self.materials[material.name] = material
        return material

    def users(self, material: str) -> list[MeshObject]:
        """The objects using material ``material``."""
        return [n for n in self.objects() if n.material == material]

    # -- tree
    def walk(self):
        """Every node below the root, depth first."""
        for child in self.root.children:
            yield from child.walk()

    def objects(self) -> list[MeshObject]:
        """Every mesh / instanced object."""
        return [n for n in self.walk() if isinstance(n, MeshObject)]

    def find(self, name: str) -> Node:
        """The node called ``name`` (KeyError if none)."""
        for n in self.walk():
            if n.name == name:
                return n
        raise KeyError(f"no node {name!r} in the scene")

    def add(self, node: Node, parent: Node | None = None, index: int | None = None) -> Node:
        """Insert ``node`` (and its subtree) under ``parent`` (default: the root); returns it. If a mesh or a
        procedural group fails to build, the error is raised and the scene stays as it was."""
        parent = parent or self.root
        if not isinstance(parent, Group):
            raise TypeError(f"{parent!r} can't have children: only groups can")
        taken = {n.name for n in self.walk()}
        for n in node.walk():
            n.name = unique_name(n.name, taken)
            taken.add(n.name)
            if isinstance(n, MeshObject) and n.material not in self.materials:
                raise KeyError(f"{n.name}: no material {n.material!r} in the scene; add it with scene.material()")
        for n in node.walk():  # build the meshes first: a shape that fails never reaches the tree
            if isinstance(n, MeshObject) and n.instances is None:
                n.attach(self)
        node.parent = parent
        parent.children.insert(len(parent.children) if index is None else index, node)
        for n in node.walk():
            n.scene = self
        self.refresh(node)
        try:
            for n in [n for n in node.walk() if isinstance(n, ProceduralGroup) and not n.children]:
                n.rebuild()
        except Exception:
            self.remove(node)
            raise
        return node

    def remove(self, node: Node) -> None:
        """Take ``node`` (and its subtree) out of the scene."""
        node.parent.children.remove(node)
        node.parent = None
        for n in node.walk():
            n.scene = None

    def refresh(self, node: Node | None = None) -> None:
        """Re-upload world matrices of ``node`` and everything below it (after a transform change)."""
        for n in (node or self.root).walk():
            n.refresh()

    # -- saving
    def settings(self) -> dict:
        """The whole scene as plain data: the material library and the node tree (``from_settings`` rebuilds it)."""
        return {"materials": {name: m.to_dict(digits=6) for name, m in self.materials.items()},
                "nodes": [n.settings() for n in self.root.children]}

    @classmethod
    def from_settings(cls, d: dict, renderer: Renderer | None = None) -> Scene:
        """A scene rebuilt from ``settings()`` data; shape functions and builders are imported by path."""
        scene = cls(renderer)
        for name, params in d.get("materials", {}).items():
            params = dict(params)
            scene.material(name, params.pop("shading"), **params)
        for nd in d.get("nodes", []):
            scene.add(node_from_settings(nd))
        return scene

    # -- what the renderer and the camera need
    def batches(self):
        """One batch per visible object with at least one instance, then the overlays."""
        return ([n.instances.batch() for n in self.objects() if n.shown() and n.instances.count]
                + [o.batch() for o in self.overlays if o.count])

    def _spheres(self) -> tuple[np.ndarray, np.ndarray]:
        centres, radii = [], []
        for n in self.objects():
            if n.shown() and n.instances.count:
                m = n.instances.matrices
                centres.append(m[:, :3, 3])
                radii.append(n.instances.mesh.radius * np.linalg.norm(m[:, :3, :3], axis=1).max(1))
        if not centres:
            return np.zeros((1, 3)), np.ones(1)
        return np.concatenate(centres), np.concatenate(radii)

    def center(self) -> np.ndarray:
        """Centre of the bounding box of everything shown (world xyz)."""
        c, r = self._spheres()
        return ((c - r[:, None]).min(0) + (c + r[:, None]).max(0)) / 2

    def bounding_radius(self) -> float:
        """Radius around ``center()`` containing everything shown."""
        c, r = self._spheres()
        return float((np.linalg.norm(c - self.center(), axis=1) + r).max())


def node_from_settings(d: dict) -> Node:
    """A node (with its subtree) from ``Node.settings()`` data; add it to a scene to draw it."""
    kw = {"visible": d.get("visible", True), "transform": Transform(**d.get("transform", {})),
          "locked": d.get("locked", False)}
    kind = d["kind"]
    if kind == "group":
        node = Group(d["name"], **kw)
    elif kind == "procedural":
        return ProceduralGroup(d["name"], d["builder"], d.get("params"), ranges=d.get("ranges"), **kw)
    elif kind == "mesh":
        node = MeshObject(d["name"], Shape(d["shape"]["func"], d["shape"].get("params")), d["material"], **kw)
    elif kind == "instances":
        node = InstancedObject(d["name"], Shape(d["shape"]["func"], d["shape"].get("params")), d["material"],
                               d.get("instances"), **kw)
    else:
        raise ValueError(f"unknown node kind {kind!r} (group, procedural, mesh or instances)")
    for c in d.get("children", []):
        child = node_from_settings(c)
        child.parent = node
        node.children.append(child)
    return node
