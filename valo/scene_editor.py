"""SceneEditor: the editor for a ``valo.Scene``. Every edit of the tree, an object or a material is undoable.

    scene = Scene()
    ...
    viewer.show(scene, SceneEditor(scene))

The actions (``add``, ``delete``, ``duplicate``, ``rename``, ``reparent``, ``set_visible``, ``set_transform``,
``set_shape_param``, ``set_object_material``, ``set_procedural_param``, ``new_material``, ``duplicate_material``,
``delete_material``) are what the Scene, Object and Materials windows call, so scripts and tests can do the same.
Selection (``select``) is not an edit and is not recorded.
"""

from __future__ import annotations

from .scene import Group, MeshObject, Node, ProceduralGroup, Scene, node_from_settings, unique_name
from .ui import Editor
from .undo import Command

TRANSFORM_FIELDS = ("translate", "rotate_deg", "scale")


class SceneEditor(Editor):
    """Editor of a Scene's objects and material library, with undo for every action."""

    title = "Scene"

    def __init__(self, scene: Scene):
        super().__init__()
        self.scene = scene
        self.selected: Node | None = None
        for m in scene.materials.values():
            self.watch(m)

    def _push(self, label: str, undo, redo, merge_key=None) -> None:
        self.undo_stack.push(Command(label, undo, redo, merge_key=merge_key))

    # -- selection (not an edit)
    def select(self, node: Node | None) -> None:
        """Make ``node`` the selected one (shown in the Object window, highlighted whole in the 3D view)."""
        self.selected = node
        if self.viewer is not None:
            objects = [] if node is None else [n for n in node.walk() if isinstance(n, MeshObject)]
            self.viewer.highlighted = [o.instances for o in objects]
            sel = self.viewer.selection
            if sel is not None and getattr(sel.source, "owner", None) is not node:
                self.viewer.selection = None  # a 3D pick of something else no longer applies

    # -- the tree
    def _insert(self, node: Node, parent: Group, index: int) -> None:
        self.scene.add(node, parent, index)

    def _take(self, node: Node) -> tuple[Group, int]:
        parent = node.parent
        index = parent.children.index(node)
        self.scene.remove(node)
        if self.selected is not None and (self.selected is node or self.selected in node.walk()):
            self.select(None)
        return parent, index

    def add(self, node: Node, parent: Group | None = None) -> Node:
        """Add ``node`` (with its subtree) at the end of ``parent`` (default: the root), select it; returns it."""
        parent = parent or self.scene.root
        self.scene.add(node, parent)
        index = parent.children.index(node)
        self._push(f"Add {node.name}", lambda: self._take(node), lambda: self._insert(node, parent, index))
        self.select(node)
        return node

    def delete(self, node: Node) -> None:
        """Remove ``node`` and everything below it."""
        parent, index = self._take(node)
        self._push(f"Delete {node.name}", lambda: self._insert(node, parent, index), lambda: self._take(node))

    def duplicate(self, node: Node) -> Node:
        """Add a copy of ``node`` (and its subtree) right after it, select it; returns the copy."""
        copy = node_from_settings(node.settings())
        parent = node.parent
        index = parent.children.index(node) + 1
        self.scene.add(copy, parent, index)
        self._push(f"Duplicate {node.name}", lambda: self._take(copy), lambda: self._insert(copy, parent, index))
        self.select(copy)
        return copy

    def rename(self, node: Node, name: str) -> None:
        """Rename ``node`` (made unique within the scene)."""
        old = node.name
        new = unique_name(name, {n.name for n in self.scene.walk() if n is not node})
        if new == old:
            return

        def apply(value):
            def f():
                node.name = value
                if isinstance(node, MeshObject):
                    node.instances.label = value
            return f

        apply(new)()
        self._push(f"Rename {old} → {new}", apply(old), apply(new))

    def reparent(self, node: Node, parent: Group, index: int | None = None) -> None:
        """Move ``node`` under group ``parent`` (at ``index``, default last); its own transform is kept, so it
        moves with its new parent."""
        if not isinstance(parent, Group):
            raise TypeError(f"{parent!r} can't have children: only groups can")
        if parent in node.walk():
            raise ValueError(f"can't move {node.name} into itself or its own subtree")
        old_parent, old_index = node.parent, node.parent.children.index(node)
        selected = self.selected

        def move(to, i):
            def f():
                self.scene.remove(node)
                self.scene.add(node, to, i)
                self.select(selected)
            return f

        new_index = index if index is not None else len(parent.children) - (old_parent is parent)
        move(parent, new_index)()
        self._push(f"Move {node.name} to {parent.name}", move(old_parent, old_index), move(parent, new_index))

    def set_visible(self, node: Node, visible: bool) -> None:
        """Show or hide ``node`` (and so everything below it)."""
        old = node.visible
        if old == visible:
            return

        def apply(v):
            def f():
                node.visible = v
            return f

        apply(visible)()
        self._push(f"{'Show' if visible else 'Hide'} {node.name}", apply(old), apply(visible))

    # -- one node's properties
    def set_transform(self, node: Node, field: str, xyz) -> None:
        """Set ``node.transform.<field>`` (translate, rotate_deg in degrees, or scale) to the 3 values ``xyz``."""
        if field not in TRANSFORM_FIELDS:
            raise KeyError(f"unknown transform field {field!r}; use one of {TRANSFORM_FIELDS}")
        old, new = getattr(node.transform, field), tuple(float(c) for c in xyz)
        if old == new:
            return

        def apply(v):
            def f():
                setattr(node.transform, field, v)
                self.scene.refresh(node)
            return f

        apply(new)()
        self._push(f"{node.name}: {field}", apply(old), apply(new), merge_key=(id(node), field))

    def set_shape_param(self, obj: MeshObject, key: str, value) -> None:
        """Set shape parameter ``key`` of ``obj`` and rebuild its mesh."""
        if key not in obj.shape.params:
            raise KeyError(f"{obj.shape.name} has no parameter {key!r}")
        old = obj.shape.params[key]
        if old == value:
            return

        def apply(v):
            def f():
                obj.shape.params[key] = v
                obj.rebuild_mesh()
            return f

        apply(value)()
        self._push(f"{obj.name}: {key}", apply(old), apply(value), merge_key=(id(obj), key))

    def set_object_material(self, obj: MeshObject, name: str) -> None:
        """Draw ``obj`` with the scene material ``name``."""
        old = obj.material
        if old == name:
            return
        if name not in self.scene.materials:
            raise KeyError(f"no material {name!r} in the scene")
        obj.set_material(name)
        self._push(f"{obj.name}: material {name}", lambda: obj.set_material(old), lambda: obj.set_material(name))

    def set_procedural_param(self, group: ProceduralGroup, key: str, value) -> None:
        """Set builder parameter ``key`` of ``group`` and rebuild its children."""
        if key not in group.params:
            raise KeyError(f"{group.name} has no parameter {key!r}")
        old = group.params[key]
        if old == value:
            return

        def apply(v):
            def f():
                group.params[key] = v
                group.rebuild()
                if self.selected is not None and self.selected.scene is None:
                    self.select(group)  # the selected child was rebuilt away
            return f

        apply(value)()
        self._push(f"{group.name}: {key}", apply(old), apply(value), merge_key=(id(group), key))

    # -- the material library
    def _put(self, m, index: int) -> None:
        items = list(self.scene.materials.items())
        items.insert(index, (m.name, m))
        self.scene.materials.clear()
        self.scene.materials.update(items)

    def _drop(self, m) -> None:
        del self.scene.materials[m.name]

    def new_material(self, name: str = "Material", shading: str = "pbr", **params):
        """Add a material to the library (name made unique); returns it."""
        m = self.scene.material(name, shading, **params)
        self.watch(m)
        index = len(self.scene.materials) - 1
        self._push(f"New material {m.name}", lambda: self._drop(m), lambda: self._put(m, index))
        return m

    def duplicate_material(self, name: str):
        """Add a copy of material ``name`` to the library; returns it."""
        params = self.scene.materials[name].to_dict(digits=6)
        return self.new_material(name, params.pop("shading"), **params)

    def delete_material(self, name: str) -> None:
        """Remove material ``name`` from the library; only an unused one can go."""
        users = self.scene.users(name)
        if users:
            raise ValueError(f"material {name!r} is used by {', '.join(u.name for u in users)}")
        m = self.scene.materials[name]
        index = list(self.scene.materials).index(name)
        self._drop(m)
        self._push(f"Delete material {name}", lambda: self._put(m, index), lambda: self._drop(m))
