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
- **Shapes.** A ``Shape`` is a function (a callable, a ``"module:function"`` path, or a sculpt-file path) plus its
  parameters; ``meshes.params.shape_params`` reads the editable ones from the signature.
- **Materials.** The scene owns a library of named materials; objects refer to one by name, so several can share it.
- **Names.** Node and material names are unique within a scene (a " 2" suffix is added when needed).
- **Drawing.** The renderer draws ``scene.batches()``; each object owns a ``valo.Instances`` whose ``owner`` is it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .geometry import transform_matrix
from .materials import Material
from .meshes.data import MeshData
from .meshes.params import ShapeParam, shape_params
from .render import Instances, Renderer
from .render import renderer as shared_renderer


def resolve(ref):
    """A callable from a callable or a ``"package.module:function"`` path."""
    if callable(ref):
        return ref
    module, _, name = str(ref).partition(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def reference(func) -> str:
    """The ``"module:qualname"`` path of an importable function (what ``resolve`` turns back into it)."""
    if isinstance(func, str):
        return func
    path = f"{func.__module__}:{func.__qualname__}"
    if "<" in path or func.__module__ == "__main__":
        raise ValueError(f"{path} is not importable by name; use a module-level function to save the scene")
    return path


def unique_name(name: str, taken) -> str:
    """``name``, or ``name 2``, ``name 3``, ... whichever is not in ``taken``."""
    if name not in taken:
        return name
    i = 2
    while f"{name} {i}" in taken:
        i += 1
    return f"{name} {i}"


class Transform:
    """Translate (xyz), rotate (degrees about x, then y, then z) and scale (xyz); ``matrix()`` = T · R · S."""

    def __init__(self, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        self.translate = tuple(float(c) for c in translate)
        self.rotate_deg = tuple(float(c) for c in rotate_deg)
        self.scale = tuple(float(c) for c in scale)

    def matrix(self) -> np.ndarray:
        """The 4×4 model matrix (math convention)."""
        return transform_matrix(self.translate, self.rotate_deg, self.scale)

    def settings(self) -> dict:
        """As plain data."""
        return {"translate": list(self.translate), "rotate_deg": list(self.rotate_deg), "scale": list(self.scale)}

    def __eq__(self, other):
        return isinstance(other, Transform) and self.settings() == other.settings()

    def __repr__(self):
        return f"Transform({self.translate}, {self.rotate_deg}, {self.scale})"


class Shape:
    """A mesh made by a function from parameters: ``build()`` returns its MeshData.

    ``source`` is a callable, a ``"module:function"`` path or the path of a sculpt file (``.py`` with ``mesh(...)``,
    whose own TRANSFORM is baked in, as in the library). ``specs`` lists the editable parameters.
    """

    def __init__(self, source, params: dict | None = None):
        self.source = source
        self.sculpt = None
        self._load()
        known = {p.name for p in self.specs}
        unknown = set(params or {}) - known
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)} for {self.name}; it takes {sorted(known)}")
        self.params = {**{p.name: p.default for p in self.specs}, **(params or {})}

    def _load(self) -> None:
        if isinstance(self.source, (str, Path)) and str(self.source).endswith(".py"):
            from .meshes.sculpt import Sculpt

            self.sculpt = Sculpt.load(self.source)
            self.func, self.specs = self.sculpt.build, self.sculpt.params
        else:
            self.func = resolve(self.source)
            self.specs: list[ShapeParam] = shape_params(self.func)

    @property
    def name(self) -> str:
        """A short label: the function name or the sculpt file's stem."""
        return Path(str(self.source)).stem if self.sculpt else getattr(self.func, "__name__", str(self.source))

    def build(self) -> MeshData:
        """The mesh for the current parameters."""
        data = self.func(self.params) if self.sculpt else self.func(**self.params)
        if not isinstance(data, MeshData):
            raise TypeError(f"{self.name} returned {type(data).__name__}, expected MeshData")
        return data

    def reload(self) -> None:
        """Re-read a sculpt file; parameters that still exist (same kind) keep their values."""
        old = {p.name: p.kind for p in self.specs}
        values = self.params
        self._load()
        self.params = {p.name: values[p.name] if old.get(p.name) == p.kind else p.default for p in self.specs}

    def settings(self) -> dict:
        """As plain data: the function reference (or file) and the parameters."""
        return {"func": str(self.source) if self.sculpt else reference(self.source), "params": dict(self.params)}


class Node:
    """A node of the scene tree: name, visibility, a transform relative to its parent, and children."""

    kind = "node"

    def __init__(self, name: str, *, visible: bool = True, transform: Transform | None = None):
        self.name, self.visible = name, visible
        self.transform = transform or Transform()
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.scene: Scene | None = None

    def world_matrix(self) -> np.ndarray:
        """parent's world matrix · own transform."""
        m = self.transform.matrix()
        return m if self.parent is None else self.parent.world_matrix() @ m

    def shown(self) -> bool:
        """Visible, and so are all ancestors."""
        return self.visible and (self.parent is None or self.parent.shown())

    def walk(self):
        """This node and all its descendants, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def refresh(self) -> None:
        """Re-upload what depends on the world transform (objects override)."""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class Group(Node):
    """A node that only carries a transform and visibility for its children."""

    kind = "group"

    def add(self, node: Node) -> Node:
        """Add ``node`` as a child (through the scene, so names stay unique and GPU objects get made)."""
        if self.scene is None:
            raise RuntimeError("add the group to a scene before adding children to it")
        return self.scene.add(node, parent=self)


class ProceduralGroup(Group):
    """A group whose children ``builder(group, **params)`` creates; changing a parameter rebuilds them all.

    ``builder`` is a callable or a ``"module:function"`` path; its parameters after the first (the group) are the
    group's editable parameters, read like shape parameters. The children are not saved with the scene: they are
    rebuilt from the parameters.
    """

    kind = "procedural"

    def __init__(self, name: str, builder, params: dict | None = None, **kw):
        super().__init__(name, **kw)
        self.builder = builder
        self.func = resolve(builder)
        self.specs = shape_params(self.func, skip=1)
        unknown = set(params or {}) - {p.name for p in self.specs}
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)} for builder {self.func.__name__}")
        self.params = {**{p.name: p.default for p in self.specs}, **(params or {})}

    def rebuild(self) -> None:
        """Remove the children and let the builder make them again from the current parameters."""
        for child in list(self.children):
            self.scene.remove(child)
        self.func(self, **self.params)


class MeshObject(Node):
    """One mesh (from ``shape``) with a material, at the node's world transform."""

    kind = "mesh"

    def __init__(self, name: str, shape: Shape, material: str, **kw):
        super().__init__(name, **kw)
        self.shape, self.material = shape, material
        self.instances: Instances | None = None  # made when the node joins a scene

    def attach(self, scene: Scene) -> None:
        """Make the GPU instances (called by Scene.add)."""
        r = scene.renderer
        self.instances = r.instances(self.shape.build(), scene.materials[self.material], label=self.name,
                                     info=self.info)
        self.instances.owner = self

    def local_instances(self) -> np.ndarray:
        """Instance matrices relative to the node: one identity for a plain mesh."""
        return np.eye(4, dtype=np.float32)[None]

    def refresh(self) -> None:
        """Upload world · instance matrices."""
        if self.instances is not None:
            self.instances.set(self.world_matrix() @ self.local_instances())

    def rebuild_mesh(self) -> None:
        """Build the mesh again from the shape's current parameters."""
        if self.instances is not None:
            self.instances.mesh = self.scene.renderer.mesh(self.shape.build())

    def set_material(self, name: str) -> None:
        """Use the scene material ``name``."""
        self.material = name
        if self.instances is not None:
            self.instances.material = self.scene.materials[name]

    def info(self, i: int) -> dict:
        """Info rows: the object, its shape and parameters."""
        params = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.shape.params.items())
        return {"Object": self.name, "Type": self.kind, "Shape": f"{self.shape.name}({params})"}


class InstancedObject(MeshObject):
    """One mesh drawn at world · M_i for each instance matrix M_i (data: positions, orientations, sizes)."""

    kind = "instances"

    def __init__(self, name: str, shape: Shape, material: str, instances=None, **kw):
        super().__init__(name, shape, material, **kw)
        self.matrices = np.asarray(np.eye(4)[None] if instances is None else instances, np.float32).reshape(-1, 4, 4)

    def local_instances(self) -> np.ndarray:
        """The instance matrices, relative to the node."""
        return self.matrices

    def set_instances(self, matrices) -> None:
        """Replace the instance matrices (N, 4, 4) and upload."""
        self.matrices = np.asarray(matrices, np.float32).reshape(-1, 4, 4)
        self.refresh()


class Scene:
    """A tree of nodes under ``root`` and a library of named materials; draws its visible objects."""

    def __init__(self, renderer: Renderer | None = None):
        """``renderer`` owns the GPU objects (default: the shared ``valo.renderer()``)."""
        self.renderer = renderer or shared_renderer()
        self.materials: dict[str, Material] = {}
        self.root = Group("Scene")
        self.root.scene = self

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
        """Insert ``node`` (and its subtree) under ``parent`` (default: the root); returns it."""
        parent = parent or self.root
        if not isinstance(parent, Group):
            raise TypeError(f"{parent!r} can't have children: only groups can")
        taken = {n.name for n in self.walk()}
        for n in node.walk():
            n.name = unique_name(n.name, taken)
            taken.add(n.name)
            if isinstance(n, MeshObject) and n.material not in self.materials:
                raise KeyError(f"{n.name}: no material {n.material!r} in the scene; add it with scene.material()")
        node.parent = parent
        parent.children.insert(len(parent.children) if index is None else index, node)
        for n in node.walk():
            n.scene = self
            if isinstance(n, MeshObject) and n.instances is None:
                n.attach(self)
        self.refresh(node)
        for n in [n for n in node.walk() if isinstance(n, ProceduralGroup) and not n.children]:
            n.rebuild()
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

    # -- what the renderer and the camera need
    def batches(self):
        """One batch per visible object with at least one instance."""
        return [n.instances.batch() for n in self.objects() if n.shown() and n.instances.count]

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
