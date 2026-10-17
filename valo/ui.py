"""The editor base class: undo history, a docking Dear ImGui layout, and schema-driven material widgets.

An app subclasses ``Editor``: every user action is a method that pushes a ``Command`` for undo, and
``windows()`` returns the app's dockable windows (title → draw function); their widgets only call those
methods, so scripts and tests can do everything the UI can. The base class draws the menu bar (app menus
from ``menus()``, then Edit and View), the dockspace whose empty central node shows the 3D scene, the
History window, and the default layout (app windows tabbed in a right column, History below them).
"""

from __future__ import annotations

from collections.abc import Callable

from imgui_bundle import imgui
from imgui_bundle.imgui import internal

from .materials import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .ui_style import ICON, apply_style
from .undo import Command, UndoStack

UI_SCALES = (0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)


class Editor:
    title = "valo"
    right_column = 0.3  # share of the window width for the right dock column
    dock_bottom = ("History",)  # windows docked under the right column; the others are tabs in it
    bottom_share = 0.25  # height share of that lower part

    def __init__(self):
        self.undo_stack = UndoStack()
        self.auto_rotate = False
        self.ui_scale = 1.0
        self.visible: dict[str, bool] = {}
        self.scene_rect: tuple[float, float, float, float] | None = None  # central dock node, logical pixels
        self._focus: str | None = None  # window to focus next frame
        self._layout = "auto"  # next frame: "auto" = default layout unless a saved one was restored, "force"

    # -- material actions (material edits reach the undo stack through watch())
    def watch(self, m: Material) -> None:
        """Record every edit of ``m`` on the undo stack."""
        m.listeners.append(self._on_material_event)

    def _on_material_event(self, m, key, old, new) -> None:
        self.undo_stack.push(Command(f"{m.name}: {key}", lambda: m.set(key, old), lambda: m.set(key, new),
                                     merge_key=(id(m), key)))

    def set_param(self, m: Material, key, value) -> None:
        m.set(key, value)

    def reset(self, m: Material) -> None:
        for s in SCHEMA[m.shading]:
            self.set_param(m, s.key, s.default)

    # -- what an app provides
    def windows(self) -> dict[str, Callable[[], None]]:
        """The app's dockable windows, title → function drawing the contents. Default: one ``panel()`` window."""
        return {self.title: self.panel}

    def panel(self) -> None:
        """Contents of the default single window (override ``windows()`` instead for several)."""

    def menus(self) -> dict[str, Callable[[], None]]:
        """Extra menus before Edit and View, name → function drawing the items (e.g. a File menu)."""
        return {}

    # -- shared widgets
    def material_section(self, m: Material) -> None:
        changed, si = imgui.combo("Shading", SHADINGS.index(m.shading), SHADINGS)
        if changed:
            self.set_param(m, "shading", SHADINGS[si])

        imgui.separator_text(f"Material parameters ({m.shading})")
        for s in SCHEMA[m.shading]:  # the schema decides the widgets
            if s.kind == "color":
                srgb = [linear_to_srgb(c) for c in m.get(s.key)]
                changed, col = imgui.color_edit3(s.label, srgb, imgui.ColorEditFlags_.picker_hue_wheel)
                if changed:
                    self.set_param(m, s.key, [srgb_to_linear(c) for c in col])
            elif s.kind == "float":
                fl = imgui.SliderFlags_.logarithmic if s.log else 0
                changed, v = imgui.slider_float(s.label, m.get(s.key), s.lo, s.hi, "%.0f" if s.log else "%.3f", fl)
                if changed:
                    self.set_param(m, s.key, v)
            else:
                changed, v = imgui.slider_int(s.label, int(m.get(s.key)), int(s.lo), int(s.hi))
                if changed:
                    self.set_param(m, s.key, v)
        if imgui.button(f"{ICON.ARROW_ROTATE_LEFT} Reset material parameters"):
            self.reset(m)

    def history_window(self) -> None:
        stack = self.undo_stack
        imgui.begin_disabled(not stack.done)
        if imgui.button(f"{ICON.ROTATE_LEFT} Undo"):
            stack.undo()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not stack.undone)
        if imgui.button(f"{ICON.ROTATE_RIGHT} Redo"):
            stack.redo()
        imgui.end_disabled()
        for cmd in reversed(stack.undone):  # newest redo at the top, greyed out
            imgui.text_disabled(f"  {cmd.label}")
        for i, cmd in enumerate(reversed(stack.done)):
            imgui.text(f"{ICON.CARET_RIGHT if i == 0 else ' '} {cmd.label}")
        if not stack.done and not stack.undone:
            imgui.text_disabled("No edits yet")

    # -- frame
    def all_windows(self) -> dict[str, Callable[[], None]]:
        return {**self.windows(), "History": self.history_window}

    def gui(self) -> None:
        """Called by the Viewer's ImguiRenderer every frame."""
        io = imgui.get_io()
        if io.key_ctrl and not io.want_text_input:
            if imgui.is_key_pressed(imgui.Key.z):
                self.undo_stack.undo()
            if imgui.is_key_pressed(imgui.Key.y):
                self.undo_stack.redo()

        wins = self.all_windows()
        self._menu_bar(wins)
        dock_id = imgui.dock_space_over_viewport(0, None, imgui.DockNodeFlags_.passthru_central_node)
        focus, self._focus = self._focus, None  # requested by last frame's layout build
        if self._layout:
            node = internal.dock_builder_get_node(dock_id)
            if self._layout == "force" or node is None or not node.is_split_node():  # nothing restored from file
                self.build_layout(dock_id, list(wins))
                self._focus = next(iter(wins))  # new windows grab focus: give it to the app's first, next frame
            self._layout = None
        central = internal.dock_builder_get_central_node(dock_id)
        self.scene_rect = (central.pos.x, central.pos.y, central.size.x, central.size.y)

        for name, draw in wins.items():
            if not self.visible.setdefault(name, True):
                continue
            if name == focus:
                imgui.set_next_window_focus()
            expanded, still_open = imgui.begin(name, True)
            if expanded:
                draw()
            imgui.end()
            self.visible[name] = bool(still_open)

    def build_layout(self, dock_id: int, names: list[str]) -> None:
        """The default layout: a right column of tabbed app windows, ``dock_bottom`` windows below it."""
        internal.dock_builder_remove_node(dock_id)
        internal.dock_builder_add_node(
            dock_id, imgui.DockNodeFlags_.passthru_central_node | internal.DockNodeFlagsPrivate_.dock_space)
        viewport = imgui.get_main_viewport()
        internal.dock_builder_set_node_size(dock_id, viewport.work_size)
        _, right, _ = internal.dock_builder_split_node_py(dock_id, imgui.Dir.right, self.right_column)
        _, bottom, top = internal.dock_builder_split_node_py(right, imgui.Dir.down, self.bottom_share)
        for name in names:
            internal.dock_builder_dock_window(name, bottom if name in self.dock_bottom else top)
        internal.dock_builder_finish(dock_id)

    def reset_layout(self) -> None:
        self.visible.clear()
        self._layout = "force"

    def _menu_bar(self, wins: dict) -> None:
        if not imgui.begin_main_menu_bar():
            return
        for name, draw in self.menus().items():
            if imgui.begin_menu(name):
                draw()
                imgui.end_menu()
        if imgui.begin_menu("Edit"):
            stack = self.undo_stack
            last = f" {stack.done[-1].label}" if stack.done else ""
            nxt = f" {stack.undone[-1].label}" if stack.undone else ""
            if imgui.menu_item(f"{ICON.ROTATE_LEFT} Undo{last}", "Ctrl+Z", False, bool(stack.done))[0]:
                stack.undo()
            if imgui.menu_item(f"{ICON.ROTATE_RIGHT} Redo{nxt}", "Ctrl+Y", False, bool(stack.undone))[0]:
                stack.redo()
            imgui.end_menu()
        if imgui.begin_menu("View"):
            for name in wins:
                if imgui.menu_item(name, "", self.visible.get(name, True))[0]:
                    self.visible[name] = not self.visible.get(name, True)
            imgui.separator()
            self.auto_rotate = imgui.menu_item("Auto-rotate camera", "", self.auto_rotate)[1]
            if imgui.begin_menu("UI scale"):
                for s in UI_SCALES:
                    if imgui.menu_item(f"{s:.0%}", "", s == self.ui_scale)[0]:
                        self.set_ui_scale(s)
                imgui.end_menu()
            if imgui.menu_item(f"{ICON.TABLE_COLUMNS} Reset layout")[0]:
                self.reset_layout()
            imgui.end_menu()
        imgui.text_disabled("   Left-drag: orbit   Wheel: zoom")
        imgui.end_main_menu_bar()

    def set_ui_scale(self, scale: float) -> None:
        self.ui_scale = scale
        apply_style(scale)
