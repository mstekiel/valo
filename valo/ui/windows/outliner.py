"""The Scene window (Blender's Outliner): the node tree with type and visibility, select, rename / duplicate / delete,
drag onto a group to move, and "+ Add"."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import meshes
from ...meshes import library
from ...scene import (
    Group,
    Node,
)
from ..style import ICON

KIND_ICONS = {"group": ICON.FOLDER, "procedural": ICON.GEARS, "mesh": ICON.CUBE, "instances": ICON.CUBES}


DRAG_TYPE = "valo-node"


class OutlinerWindow:
    """SceneEditor part: the Scene window (the tree)."""

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
        if imgui.begin_menu(f"{ICON.BOOK} Library shape", bool(names)):
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
