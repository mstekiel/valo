"""SceneEditor: the editor for a ``valo.Scene``. Every edit of the tree, an object or a material is undoable.

    scene = Scene()
    ...
    viewer.show(scene, SceneEditor(scene))

Windows: **Scene** (the tree: name, type, visibility; click to select, right-click to rename / duplicate / delete,
drag a row onto a group to move it there, "+ Add" for a shape or a group), **Object** (the selection's shape
parameters, transform, material and mesh diagnostics) and **Materials** (the library). A click in the 3D view
selects the object under the pointer.

The actions (``add``, ``add_shape``, ``delete``, ``duplicate``, ``rename``, ``reparent``, ``set_visible``,
``set_transform``, ``set_shape``, ``set_shape_param``, ``set_object_material``, ``set_procedural_param``,
``new_material``, ``duplicate_material``, ``delete_material``) are what the windows call, so scripts
and tests can do the same.
Selection (``select``) is not an edit and is not recorded. Locked nodes (an app's fixed structure) can be hidden and
edited but not deleted, renamed, duplicated or moved, and nothing can be added inside a procedural group (its
builder owns those children).
"""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from ..config import rc
from ..scene import (
    Group,
    MeshObject,
    Node,
    ProceduralGroup,
    Scene,
    Shape,
    node_from_settings,
    unique_name,
)
from .editor import Editor
from .undo import Command
from .windows.materials import MaterialsWindow
from .windows.object import ObjectWindow
from .windows.outliner import OutlinerWindow

TRANSFORM_FIELDS = ("translate", "rotate_deg", "scale")


def shape_error(trace: str, path) -> str:
    """A traceback cut to what concerns shape file ``path``: from its last frame (or the error line) to the end."""
    lines = trace.rstrip("\n").splitlines()
    name = Path(path).name
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("File ") and name in line]
    return "\n".join(line.removeprefix("  ") for line in (lines[starts[-1]:] if starts else lines[-1:]))


class SceneEditor(OutlinerWindow, ObjectWindow, MaterialsWindow, Editor):
    """Editor of a Scene's objects and material library, with undo for every action."""

    title = "Scene"
    dock_bottom = ("Object", "Materials")

    def __init__(self, scene: Scene):
        super().__init__()
        self.scene = scene
        self.selected: Node | None = None
        self.material_selected: str | None = next(iter(scene.materials), None)
        self.uniform_scale = True
        self._last_pick = None  # the viewer's selection last frame: a new 3D click selects its object
        self._renaming: tuple[Node, str] | None = None
        self._reports: dict[int, tuple] = {}  # id(object) → (gpu mesh, MeshReport), made again after a change
        self.errors: dict[str, str] = {}  # object name → traceback of its shape file's last failed reload
        self._polled = 0.0
        for m in scene.materials.values():
            self.watch(m)

    def _push(self, label: str, undo, redo, merge_key=None) -> None:
        self.undo_stack.push(Command(label, undo, redo, merge_key=merge_key))

    @contextmanager
    def one_step(self, label: str):
        """Record every action inside the ``with`` block as one undo step called ``label``; if the block raises,
        the actions it already did are undone and the error passes on."""
        stack = self.undo_stack
        start = len(stack.done)
        try:
            yield
        except Exception:
            while len(stack.done) > start:
                stack.undo()
            stack.undone.clear()
            raise
        finally:
            steps = stack.done[start:]
            del stack.done[start:]
            if len(steps) == 1:
                stack.done.append(steps[0])
            elif steps:
                stack.push(Command(label, lambda: [c.undo() for c in reversed(steps)],
                                   lambda: [c.redo() for c in steps]))

    # -- selection (not an edit)
    def select(self, node: Node | None) -> None:
        """Make ``node`` the selected one (shown in the Object window; its bounding box in the 3D view)."""
        self.selected = node
        if self.viewer is not None:
            self.viewer.outlined = [] if node is None else [node]
            sel = self.viewer.selection
            if sel is not None and getattr(sel.source, "owner", None) is not node:
                self.viewer.selection = None  # a 3D pick of something else no longer applies

    # -- the tree
    @staticmethod
    def _check_free(node: Node, action: str) -> None:
        locked = next((n for n in node.walk() if n.locked), None)
        if locked is not None:
            raise ValueError(f"{locked.name} is part of a fixed structure: it can be hidden, not {action}")

    @staticmethod
    def _check_can_hold(parent: Node) -> None:
        if not isinstance(parent, Group):
            raise TypeError(f"{parent!r} can't have children: only groups can")
        p = parent
        while p is not None:
            if isinstance(p, ProceduralGroup):
                raise ValueError(f"{parent.name} is built by {p.name}'s builder: nothing can be added inside it")
            p = p.parent

    def _insert(self, node: Node, parent: Group, index: int) -> None:
        self.scene.add(node, parent, index)

    def _take(self, node: Node) -> tuple[Group, int]:
        parent = node.parent
        index = parent.children.index(node)
        self.scene.remove(node)
        if self.selected is not None and (self.selected is node or self.selected in node.walk()):
            self.select(None)
        return parent, index

    def add_shape(self, source, parent: Group | None = None, material: str | None = None) -> MeshObject:
        """Add a MeshObject made by ``source`` (a callable, ``"module:function"`` or shape file) with default
        parameters and ``material`` (default: the library's first, or a new one); returns it."""
        shape = Shape(source)
        with self.one_step(f"Add {shape.name.capitalize()}"):
            if material is None:
                material = next(iter(self.scene.materials), None) or self.new_material().name
            return self.add(MeshObject(shape.name.capitalize(), shape, material), parent)

    def add(self, node: Node, parent: Group | None = None) -> Node:
        """Add ``node`` (with its subtree) at the end of ``parent`` (default: the root), select it; returns it."""
        parent = parent or self.scene.root
        self._check_can_hold(parent)
        self.scene.add(node, parent)
        index = parent.children.index(node)
        self._push(f"Add {node.name}", lambda: self._take(node), lambda: self._insert(node, parent, index))
        self.select(node)
        return node

    def delete(self, node: Node) -> None:
        """Remove ``node`` and everything below it."""
        self._check_free(node, "deleted")
        parent, index = self._take(node)
        self._push(f"Delete {node.name}", lambda: self._insert(node, parent, index), lambda: self._take(node))

    def duplicate(self, node: Node) -> Node:
        """Add a copy of ``node`` (and its subtree) right after it, select it; returns the copy."""
        self._check_free(node, "duplicated")
        copy = node_from_settings(node.settings())
        parent = node.parent
        index = parent.children.index(node) + 1
        self.scene.add(copy, parent, index)
        self._push(f"Duplicate {node.name}", lambda: self._take(copy), lambda: self._insert(copy, parent, index))
        self.select(copy)
        return copy

    def rename(self, node: Node, name: str) -> None:
        """Rename ``node`` (made unique within the scene)."""
        if node.locked:
            self._check_free(node, "renamed")
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
        self._check_free(node, "moved")
        self._check_can_hold(parent)
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

    def set_shape(self, obj: MeshObject, source) -> None:
        """Make ``obj`` from another shape ``source`` (callable, ``"module:function"`` or shape file), with its
        default parameters; instances, material and transform stay."""
        old, new = obj.shape, Shape(source)

        def apply(shape):
            def f():
                obj.shape = shape
                obj.rebuild_mesh()
            return f

        apply(new)()
        self._push(f"{obj.name}: shape {new.name}", apply(old), apply(new))

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
        """Set builder parameter ``key`` of ``group`` and rebuild its children. If the builder fails, the parameter
        and the children stay as they were, the error is kept in ``errors`` (shown in the Object window) and nothing
        is recorded."""
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

        try:
            apply(value)()
        except Exception:
            group.params[key] = old
            self.errors[group.name] = traceback.format_exc(limit=-3)
            return
        self.errors.pop(group.name, None)
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

    def can_export(self) -> bool:
        """True unless a procedural group builds from a context object (then the scene can't be saved)."""
        return super().can_export() and not any(getattr(n, "context", None) is not None for n in self.scene.walk())

    # -- windows
    def windows(self):
        """Scene (the tree), Object (the selection) and Materials (the library)."""
        return {"Scene": self.scene_window, "Object": self.object_window, "Materials": self.materials_window}

    def gui(self) -> None:
        """Follow 3D clicks (a click selects the clicked object, the background deselects), then draw the frame."""
        if self.viewer is not None and self.viewer.selection is not self._last_pick:
            self._last_pick = pick = self.viewer.selection
            owner = getattr(pick.source, "owner", None) if pick is not None else None
            if pick is None or owner is not None:
                self.select(owner)
                self.viewer.selection = pick  # keep the clicked instance for the Info window
        self.poll()
        super().gui()

    # -- hot reload (a file change is not an edit: not recorded for undo)
    def poll(self) -> None:
        """Every ``rc.studio.poll_s`` seconds, reload the shapes whose shape files changed on disk."""
        now = time.monotonic()
        if now - self._polled < rc.studio.poll_s:
            return
        self._polled = now
        for obj in self.scene.objects():
            if obj.shape.stale():
                self.reload_shape(obj)

    def reload_shape(self, obj: MeshObject) -> None:
        """Re-read ``obj``'s shape file and rebuild its mesh; on an error keep the last good mesh and remember the
        traceback in ``errors`` (shown in the Object window)."""
        try:
            obj.shape.reload()
            obj.rebuild_mesh()
        except Exception:
            obj.shape.mtime = Path(obj.shape.source).stat().st_mtime  # retry only after the next change
            self.errors[obj.name] = shape_error(traceback.format_exc(), obj.shape.source)
            return
        self.errors.pop(obj.name, None)


