"""The nodes of the scene tree: ``Transform`` and ``Node`` with its kinds ``Group``, ``ProceduralGroup``, ``MeshObject``
and ``InstancedObject``. World = parent world · own transform; a node is drawn only when it and its ancestors are
visible. See ``valo.scene`` for the whole model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gpu.instances import Instances
from ..math import bounding_box, box_corners, scale, transform_matrix, translate
from ..meshes.params import shape_params
from .shape import Shape, reference, resolve

if TYPE_CHECKING:
    from .scene import Scene


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


class Node:
    """A node of the scene tree: name, visibility, a transform relative to its parent, and children.

    A ``locked`` node is part of an app's fixed structure: the editor can hide it but not delete, rename, duplicate
    or move it.
    """

    kind = "node"

    def __init__(self, name: str, *, visible: bool = True, transform: Transform | None = None,
                 locked: bool = False):
        self.name, self.visible, self.locked = name, visible, locked
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

    def bounding_box(self) -> np.ndarray | None:
        """The box around everything shown at or below this node, in this node's frame (it turns and scales with
        it), as a box matrix mapping the unit cube [-0.5, 0.5]^3 onto it in world space; None if nothing is shown."""
        corners = []
        for n in self.walk():
            if isinstance(n, MeshObject) and n.instances is not None and n.instances.count and n.shown():
                local = np.c_[box_corners(*n.instances.mesh.bounds), np.ones(8)]  # (8, 4)
                corners.append(np.einsum("nij,kj->nki", n.instances.matrices, local)[..., :3].reshape(-1, 3))
        return bounding_box(np.concatenate(corners), self.world_matrix()) if corners else None

    def settings(self) -> dict:
        """The node and its subtree as plain data (``node_from_settings`` rebuilds it)."""
        d = {"kind": self.kind, "name": self.name, "visible": self.visible, "transform": self.transform.settings()}
        if self.locked:
            d["locked"] = True
        if self.children:
            d["children"] = [c.settings() for c in self.children]
        return d

    def editable(self) -> dict:
        """The properties a user can edit and a builder can set: visibility and transform."""
        return {"visible": self.visible, "transform": self.transform.settings()}

    def _take(self, key: str, fresh: Node) -> None:
        setattr(self, key, getattr(fresh, key))

    def adopt(self, fresh: Node) -> None:
        """Become ``fresh``, this node's rebuilt twin, except for the edits the user made: an ``editable()`` property
        stays as it is only when it differs from what the builder gave last time."""
        self.locked = fresh.locked
        built, mine, theirs = getattr(self, "_built", {}), self.editable(), fresh.editable()
        for key, value in theirs.items():
            if key in built and mine[key] != built[key]:
                continue  # edited by the user: keep it
            if mine[key] != value:
                self._take(key, fresh)
        self._built = theirs

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
    """A group whose children ``builder(group, **params)`` creates; changing a parameter rebuilds them.

    ``builder`` is a callable or a ``"module:function"`` path; its parameters after the first (the group) are the
    group's editable parameters, read like shape parameters (``ranges`` overrides their slider ranges).
    ``context`` is any object the builder needs besides the parameters (``group.context``, e.g. an app's data
    source); a group with one can't be saved. The children are not saved: they are rebuilt from the parameters.

    A rebuild keeps the children it makes again: a new child with the same name path and type as an old one is
    replaced by that old node, which keeps the user's edits (visibility, material, shape, transform) and takes only
    what the builder computes (``Node.adopt``: instance matrices, info). So selection and undo history stay valid.
    """

    kind = "procedural"

    def __init__(self, name: str, builder, params: dict | None = None, *, ranges: dict | None = None,
                 context=None, **kw):
        super().__init__(name, **kw)
        self.builder, self.context, self.ranges = builder, context, dict(ranges or {})
        self.func = resolve(builder)
        self.specs = shape_params(self.func, self.ranges, skip=1)
        unknown = set(params or {}) - {p.name for p in self.specs}
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)} for builder {self.func.__name__}")
        self.params = {**{p.name: p.default for p in self.specs}, **(params or {})}

    def settings(self) -> dict:
        """The group, its builder and parameters; not the children (they are rebuilt)."""
        if self.context is not None:
            raise ValueError(f"{self.name} builds from a context object, so it can't be saved as plain data")
        d = super().settings()
        d.pop("children", None)
        d = {**d, "builder": reference(self.builder), "params": dict(self.params)}
        if self.ranges:
            d["ranges"] = {k: list(v) for k, v in self.ranges.items()}
        return d

    def _paths(self, node: Node, prefix: tuple = ()):
        for c in node.children:
            yield prefix + (c.name,), c
            yield from self._paths(c, prefix + (c.name,))

    def rebuild(self) -> None:
        """Let the builder make the children again from the current parameters, keeping the ones it remakes."""
        old, before = dict(self._paths(self)), list(self.children)
        for child in before:
            self.scene.remove(child)
        try:
            self.func(self, **self.params)
        except Exception:  # put the old children back, so a failed rebuild changes nothing
            for child in list(self.children):
                self.scene.remove(child)
            for child in before:
                self.scene.add(child, self)
            raise
        self._reconcile(self, (), old)
        self.scene.refresh(self)

    def _reconcile(self, parent: Node, prefix: tuple, old: dict) -> None:
        for i, fresh in enumerate(list(parent.children)):
            path = prefix + (fresh.name,)
            twin = old.get(path)
            node = fresh
            if twin is None or type(twin) is not type(fresh):
                fresh._built = fresh.editable()  # what the builder gave: later rebuilds compare edits to it
            else:
                twin.adopt(fresh)
                twin.children, twin.parent, twin.scene = fresh.children, parent, parent.scene
                for c in twin.children:
                    c.parent = twin
                parent.children[i] = node = twin
            self._reconcile(node, path, old)


class MeshObject(Node):
    """One mesh (from ``shape``) with a material, at the node's world transform."""

    kind = "mesh"

    def __init__(self, name: str, shape: Shape, material: str, *, info=None, **kw):
        """``info(i) -> dict`` (optional): extra Info rows for instance ``i``, after the object rows."""
        super().__init__(name, **kw)
        self.shape, self.material, self.info_hook = shape, material, info
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

    def settings(self) -> dict:
        """The object, its shape and material name."""
        return {**super().settings(), "shape": self.shape.settings(), "material": self.material}

    def info(self, i: int) -> dict:
        """Info rows: the object, its shape and parameters."""
        params = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.shape.params.items())
        rows = {"Object": self.name, "Type": self.kind, "Shape": f"{self.shape.name}({params})"}
        return {**rows, **self.info_hook(i)} if self.info_hook else rows

    def editable(self) -> dict:
        """Visibility, transform, material and shape (function and parameters)."""
        source = self.shape.source if isinstance(self.shape.source, str) else self.shape.func
        return {**super().editable(), "material": self.material, "shape": (source, dict(self.shape.params))}

    def _take(self, key: str, fresh: Node) -> None:
        if key == "material":
            self.set_material(fresh.material)
        elif key == "shape":
            self.shape = fresh.shape
            self.instances.mesh = fresh.instances.mesh
        else:
            super()._take(key, fresh)

    def adopt(self, fresh: Node) -> None:
        """Also take the rebuilt twin's info hook."""
        super().adopt(fresh)
        self.info_hook = fresh.info_hook


class InstancedObject(MeshObject):
    """One mesh drawn once per instance matrix M_i (data: positions, orientations, sizes), at parent world · M_i · T.

    Like Blender's instancing, the object's own ``transform`` T acts on the mesh of each instance: it turns, scales
    and moves every copy in place, and the copies stay where their M_i put them. ``world_matrix()`` is therefore the
    parent's (the frame the M_i live in).
    """

    kind = "instances"

    def __init__(self, name: str, shape: Shape, material: str, instances=None, **kw):
        super().__init__(name, shape, material, **kw)
        self.matrices = np.asarray(np.eye(4)[None] if instances is None else instances, np.float32).reshape(-1, 4, 4)

    def world_matrix(self) -> np.ndarray:
        """The frame of the instance matrices: the parent's world matrix (the own transform acts per instance)."""
        return np.eye(4) if self.parent is None else self.parent.world_matrix()

    def local_instances(self) -> np.ndarray:
        """The instance matrices with the object's transform applied to each instance's mesh: M_i · T."""
        return self.matrices @ self.transform.matrix().astype(np.float32)

    def instance_boxes(self) -> np.ndarray:
        """The box of every instance around its mesh, in that instance's frame (it turns, scales and mirrors with
        the instance), as box matrices (N, 4, 4) mapping the unit cube [-0.5, 0.5]^3 onto it in world space."""
        if self.instances is None:
            return np.zeros((0, 4, 4))
        lo, hi = (np.asarray(b, float) for b in self.instances.mesh.bounds)
        return self.instances.matrices.astype(float) @ translate(*((lo + hi) / 2)) @ scale(*(hi - lo))

    def instance_box(self, i: int) -> np.ndarray:
        """The box of instance ``i`` around its mesh (see ``instance_boxes``)."""
        return self.instance_boxes()[i]

    def settings(self) -> dict:
        """The object and its instance matrices (as nested lists)."""
        return {**super().settings(), "instances": self.matrices.tolist()}

    def adopt(self, fresh: Node) -> None:
        """Take the rebuilt twin's instance matrices too."""
        super().adopt(fresh)
        self.matrices = fresh.matrices

    def set_instances(self, matrices) -> None:
        """Replace the instance matrices (N, 4, 4) and upload."""
        self.matrices = np.asarray(matrices, np.float32).reshape(-1, 4, 4)
        self.refresh()
