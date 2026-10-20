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

from imgui_bundle import imgui

from . import meshes
from .config import rc
from .meshes import library
from .meshes.check import check
from .meshes.params import ShapeParam
from .scene import (
    Group,
    InstancedObject,
    MeshObject,
    Node,
    ProceduralGroup,
    Scene,
    Shape,
    Transform,
    node_from_settings,
    unique_name,
)
from .ui import Editor
from .ui_style import ICON
from .undo import Command

TRANSFORM_FIELDS = ("translate", "rotate_deg", "scale")
KIND_ICONS = {"group": ICON.FOLDER, "procedural": ICON.GEARS, "mesh": ICON.CUBE, "instances": ICON.CUBES}
DRAG_TYPE = "valo-node"


def shape_error(trace: str, path) -> str:
    """A traceback cut to what concerns shape file ``path``: from its last frame (or the error line) to the end."""
    lines = trace.rstrip("\n").splitlines()
    name = Path(path).name
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("File ") and name in line]
    return "\n".join(line.removeprefix("  ") for line in (lines[starts[-1]:] if starts else lines[-1:]))


class SceneEditor(Editor):
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
        self.errors: dict[str, str] = {}  # object name → traceback of its sculpt file's last failed reload
        self._polled = 0.0
        for m in scene.materials.values():
            self.watch(m)

    def _push(self, label: str, undo, redo, merge_key=None) -> None:
        self.undo_stack.push(Command(label, undo, redo, merge_key=merge_key))

    @contextmanager
    def one_step(self, label: str):
        """Record every action inside the ``with`` block as one undo step called ``label``."""
        stack = self.undo_stack
        start = len(stack.done)
        try:
            yield
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
        """Make ``node`` the selected one (shown in the Object window, highlighted whole in the 3D view)."""
        self.selected = node
        if self.viewer is not None:
            objects = [] if node is None else [n for n in node.walk() if isinstance(n, MeshObject)]
            self.viewer.highlighted = [o.instances for o in objects]
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
        """Add a MeshObject made by ``source`` (a callable, ``"module:function"`` or sculpt file) with default
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
        """Make ``obj`` from another shape ``source`` (callable, ``"module:function"`` or sculpt file), with its
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
        """Every ``rc.sculptor.poll_s`` seconds, reload the shapes whose sculpt files changed on disk."""
        now = time.monotonic()
        if now - self._polled < rc.sculptor.poll_s:
            return
        self._polled = now
        for obj in self.scene.objects():
            if obj.shape.stale():
                self.reload_shape(obj)

    def reload_shape(self, obj: MeshObject) -> None:
        """Re-read ``obj``'s sculpt file and rebuild its mesh; on an error keep the last good mesh and remember the
        traceback in ``errors`` (shown in the Object window)."""
        try:
            obj.shape.reload()
            obj.rebuild_mesh()
        except Exception:
            obj.shape.mtime = Path(obj.shape.source).stat().st_mtime  # retry only after the next change
            self.errors[obj.name] = shape_error(traceback.format_exc(), obj.shape.source)
            return
        self.errors.pop(obj.name, None)

    def _insert_parent(self) -> Group:
        """Where "+ Add" puts a node: into the selected group, next to the selected object, or at the root."""
        sel = self.selected
        parent = sel if isinstance(sel, Group) else (sel.parent if sel is not None else self.scene.root)
        try:
            self._check_can_hold(parent)
        except (TypeError, ValueError):
            return self.scene.root
        return parent

    def _add_menu(self) -> None:
        if imgui.begin_menu(f"{ICON.CUBE} Shape"):
            for name, func in meshes.STANDARD.items():
                if imgui.menu_item_simple(name):
                    self.add_shape(func, self._insert_parent())
            imgui.end_menu()
        names = library.names()
        if imgui.begin_menu(f"{ICON.BOOK} Library sculpt", bool(names)):
            for name in names:
                if imgui.menu_item_simple(name):
                    self.add_shape(library.path(name), self._insert_parent())
            imgui.end_menu()
        if imgui.menu_item_simple(f"{ICON.FOLDER} Group"):
            self.add(Group("Group"), self._insert_parent())

    def scene_window(self) -> None:
        """The object tree: name, type and visibility per row; select, edit, drag to reparent, add."""
        if imgui.button(f"{ICON.PLUS} Add"):
            imgui.open_popup("add")
        if imgui.begin_popup("add"):
            self._add_menu()
            imgui.end_popup()
        flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
        if imgui.begin_table("tree", 3, flags):
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("Type", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("Shown", imgui.TableColumnFlags_.width_fixed)
            for child in list(self.scene.root.children):
                self._tree_row(child)
            imgui.end_table()
        self._rename_popup()

    def _tree_row(self, node: Node) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.push_id(str(id(node)))
        flags = (imgui.TreeNodeFlags_.span_avail_width | imgui.TreeNodeFlags_.open_on_arrow
                 | imgui.TreeNodeFlags_.default_open)
        if not isinstance(node, Group) or not node.children:
            flags |= imgui.TreeNodeFlags_.leaf
        if node is self.selected:
            flags |= imgui.TreeNodeFlags_.selected
        dim = not node.shown()
        if dim:
            imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
        is_open = imgui.tree_node_ex(node.name, flags)
        if dim:
            imgui.pop_style_color()
        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            self.select(node)
            self._focus.append("Object")
        if imgui.begin_popup_context_item("row"):
            self._row_menu(node)
            imgui.end_popup()
        if not any(n.locked for n in node.walk()) and imgui.begin_drag_drop_source():
            imgui.set_drag_drop_payload_py_id(DRAG_TYPE, id(node))
            imgui.text(node.name)
            imgui.end_drag_drop_source()
        if self._holds(node) and imgui.begin_drag_drop_target():
            payload = imgui.accept_drag_drop_payload_py_id(DRAG_TYPE)
            if payload is not None:
                dragged = next((n for n in self.scene.walk() if id(n) == payload.data_id), None)
                if dragged is not None and node not in dragged.walk():
                    self.reparent(dragged, node)
            imgui.end_drag_drop_target()
        imgui.table_next_column()
        imgui.text_disabled(f"{KIND_ICONS.get(node.kind, '')} {node.kind}")
        imgui.table_next_column()
        if imgui.small_button(ICON.EYE if node.visible else ICON.EYE_SLASH):
            self.set_visible(node, not node.visible)
        imgui.pop_id()
        if is_open:
            for child in list(getattr(node, "children", [])):
                self._tree_row(child)
            imgui.tree_pop()

    def _holds(self, node: Node) -> bool:
        try:
            self._check_can_hold(node)
        except (TypeError, ValueError):
            return False
        return True

    def _row_menu(self, node: Node) -> None:
        if any(n.locked for n in node.walk()):
            imgui.text_disabled(f"{ICON.LOCK} part of a fixed structure: hide it with the eye")
            return
        if imgui.menu_item_simple(f"{ICON.PEN} Rename"):
            self._renaming = (node, node.name)
        if imgui.menu_item_simple(f"{ICON.COPY} Duplicate"):
            self.duplicate(node)
        if node.parent is not self.scene.root and imgui.menu_item_simple(f"{ICON.ARROW_UP} Move to top level"):
            self.reparent(node, self.scene.root)
        if imgui.menu_item_simple(f"{ICON.TRASH} Delete"):
            self.delete(node)

    def _rename_popup(self) -> None:
        if self._renaming is None:
            return
        if not imgui.is_popup_open("Rename"):
            imgui.open_popup("Rename")
        if imgui.begin_popup_modal("Rename", None, imgui.WindowFlags_.always_auto_resize)[0]:
            node, text = self._renaming
            if imgui.is_window_appearing():
                imgui.set_keyboard_focus_here()
            done, text = imgui.input_text("##name", text, imgui.InputTextFlags_.enter_returns_true)
            self._renaming = (node, text)
            if imgui.button("OK") or done:
                if text.strip():
                    self.rename(node, text.strip())
                self._renaming = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel"):
                self._renaming = None
                imgui.close_current_popup()
            imgui.end_popup()

    @staticmethod
    def _param_widget(p: ShapeParam, value) -> tuple[bool, object]:
        if p.kind == "float":
            return imgui.slider_float(p.name, float(value), p.lo, p.hi, "%.4g")
        if p.kind == "int":
            return imgui.slider_int(p.name, int(value), int(p.lo), int(p.hi))
        return imgui.checkbox(p.name, bool(value))

    def object_window(self) -> None:
        """The selected node: its shape or builder parameters, transform, material and mesh diagnostics."""
        node = self.selected
        if node is None:
            imgui.text_disabled("Select an object in the Scene window or the 3D view")
            return
        imgui.text(f"{KIND_ICONS.get(node.kind, '')} {node.name}")
        imgui.same_line()
        imgui.text_disabled(node.kind)
        if node.name in self.errors:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.45, 0.4, 1.0))
            imgui.text_wrapped(self.errors[node.name])
            imgui.pop_style_color()
            imgui.text_disabled("showing the last good result")
        if isinstance(node, MeshObject):
            imgui.separator_text("Shape")
            self._shape_combo(node)
            for p in node.shape.specs:
                changed, v = self._param_widget(p, node.shape.params[p.name])
                if changed:
                    self.set_shape_param(node, p.name, v)
        elif isinstance(node, ProceduralGroup):
            imgui.separator_text(f"Builder: {node.func.__name__}")
            for p in node.specs:
                changed, v = self._param_widget(p, node.params[p.name])
                if changed:
                    self.set_procedural_param(node, p.name, v)
        self._transform_section(node)
        if isinstance(node, MeshObject):
            imgui.separator_text("Material")
            if imgui.begin_combo("Material", node.material):
                for name in self.scene.materials:
                    if imgui.selectable(name, name == node.material)[0]:
                        self.set_object_material(node, name)
                imgui.end_combo()
            if imgui.small_button(f"{ICON.PALETTE} Edit in Materials"):
                self.material_selected = node.material
                self._focus.append("Materials")
            imgui.separator_text("Mesh")
            for line in self.diagnostics(node):
                imgui.text_disabled(line)

    def _shape_combo(self, obj: MeshObject) -> None:
        if imgui.begin_combo("Shape", obj.shape.name):
            for name, func in meshes.STANDARD.items():
                if imgui.selectable(name, func is obj.shape.func)[0]:
                    self.set_shape(obj, func)
            for name in library.names():
                if imgui.selectable(f"{ICON.BOOK} {name}", obj.shape.name == name and obj.shape.sculpt)[0]:
                    self.set_shape(obj, library.path(name))
            imgui.end_combo()

    def _transform_section(self, node: Node) -> None:
        imgui.separator_text("Transform")
        t = node.transform
        changed, v = imgui.drag_float3("Translate", list(t.translate), 0.01)
        if changed:
            self.set_transform(node, "translate", v)
        changed, v = imgui.drag_float3("Rotate xyz (deg)", list(t.rotate_deg), 0.5, -360.0, 360.0, "%.1f")
        if changed:
            self.set_transform(node, "rotate_deg", v)
        _, self.uniform_scale = imgui.checkbox("Uniform scale", self.uniform_scale)
        if self.uniform_scale:
            changed, k = imgui.drag_float("Scale", t.scale[0], 0.01, 0.01, 100.0)
            if changed:
                self.set_transform(node, "scale", (k, k, k))
        else:
            changed, v = imgui.drag_float3("Scale xyz", list(t.scale), 0.01, 0.01, 100.0)
            if changed:
                self.set_transform(node, "scale", v)
        if imgui.small_button(f"{ICON.ARROW_ROTATE_LEFT} Reset transform"):
            for field, value in Transform().settings().items():
                self.set_transform(node, field, value)

    def diagnostics(self, obj: MeshObject) -> list[str]:
        """``meshes.check`` lines for ``obj``'s current mesh (computed again only after the mesh changed)."""
        gpu, report = self._reports.get(id(obj), (None, None))
        if gpu is not obj.instances.mesh:
            report = check(obj.shape.build())
            self._reports[id(obj)] = (obj.instances.mesh, report)
        lines = report.lines()
        if isinstance(obj, InstancedObject):
            lines.insert(0, f"{obj.instances.count} instances")
        return lines

    def materials_window(self) -> None:
        """The material library: select one to edit it; New, Duplicate, and Delete for unused materials."""
        mats = self.scene.materials
        if self.material_selected not in mats:
            self.material_selected = next(iter(mats), None)
        if imgui.button(f"{ICON.PLUS} New"):
            self.material_selected = self.new_material().name
        name = self.material_selected
        imgui.same_line()
        imgui.begin_disabled(name is None)
        if imgui.button(f"{ICON.COPY} Duplicate"):
            self.material_selected = self.duplicate_material(name).name
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(name is None or bool(self.scene.users(name)))
        if imgui.button(f"{ICON.TRASH} Delete"):
            self.delete_material(name)
        imgui.end_disabled()
        if imgui.begin_list_box("##materials", imgui.ImVec2(-1, 5 * imgui.get_text_line_height_with_spacing())):
            for n in mats:
                if imgui.selectable(f"{n}  ({len(self.scene.users(n))} users)", n == self.material_selected)[0]:
                    self.material_selected = n
            imgui.end_list_box()
        if self.material_selected in mats:
            self.material_section(mats[self.material_selected])
