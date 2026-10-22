"""The editor base class: undo history, a docking Dear ImGui layout, and schema-driven material widgets.

An app subclasses ``Editor``: every user action is a method that pushes a ``Command`` for undo, and
``windows()`` returns the app's dockable windows (title → draw function); their widgets only call those
methods, so scripts and tests can do everything the UI can. The base class draws the menu bar (app menus
from ``menus()``, then Edit and View), the dockspace whose empty central node shows the 3D scene, the
History window, and the default layout (app windows tabbed in a right column, History below them).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from imgui_bundle import imgui
from imgui_bundle.imgui import internal

from ..config import rc
from ..gpu.material import SCHEMA, Material
from ..offline import image_settings
from .messages import Messages
from .overlays import Overlays
from .style import ICON
from .undo import Command, UndoStack
from .widgets import Widgets
from .windows.camera import CameraWindow
from .windows.console import ConsoleWindow
from .windows.history import HistoryWindow
from .windows.info import InfoWindow
from .windows.lighting import LightingWindow
from .windows.preferences import PreferencesWindow
from .windows.render import RenderWindow

UI_SCALES = (0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
SHARED_WINDOWS = ("Info", "Camera", "Lighting", "History", "Console")  # by the base class, in the lower part


class Editor(CameraWindow, LightingWindow, InfoWindow, HistoryWindow, ConsoleWindow, RenderWindow, PreferencesWindow,
             Overlays, Widgets, Messages):
    """Base class for an app's editor: undoable actions, dockable windows and the shared UI."""
    title = "valo"
    right_column: float | None = None  # share of the window width for the right dock column (None: rc.ui)
    dock_bottom: tuple[str, ...] = ()  # app windows docked under the right column, before the shared ones
    bottom_share: float | None = None  # height share of that lower part (None: rc.ui)
    dock_left: tuple[str, ...] = ()  # app windows docked in a left column (e.g. a code editor), tabbed
    left_column: float | None = None  # share of the window width for that left column (None: rc.ui)

    def __init__(self):
        self.undo_stack = UndoStack()
        self.auto_rotate = False
        self.viewer = None  # set by Viewer.show(): camera, lighting and scene
        self._ground_before_flat: tuple | None = None  # restored when Flat ambient is switched off
        self.ui_scale = rc.ui.scale
        self.show_preferences = False
        self.show_render = False
        self.render_options = {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in image_settings(None).items()}  # from rc.render
        self.render_status = ""
        self.visible: dict[str, bool] = {}
        self.scene_rect: tuple[float, float, float, float] | None = None  # central dock node, logical pixels
        self._focus: list[str] = []  # windows to focus next frame (the first tab of each dock node)
        self._layout = "auto"  # next frame: "auto" = default layout unless a saved one was restored, "force"

    # -- material actions (material edits reach the undo stack through watch())
    def watch(self, m: Material) -> None:
        """Record every edit of ``m`` on the undo stack."""
        m.listeners.append(self._on_material_event)

    def _on_material_event(self, m, key, old, new) -> None:
        self.undo_stack.push(Command(f"{m.name}: {key}", lambda: m.set(key, old), lambda: m.set(key, new),
                                     merge_key=(id(m), key)))

    def set_param(self, m: Material, key, value) -> None:
        """Set material parameter ``key`` (undoable once the material is watched)."""
        m.set(key, value)

    def reset(self, m: Material) -> None:
        """Reset every parameter of ``m``'s shading model to its default, as one undo step."""
        old = {s.key: m.get(s.key) for s in SCHEMA[m.shading]}
        new = {s.key: s.default for s in SCHEMA[m.shading]}

        def apply(values):
            def f():
                for key, value in values.items():
                    m.set(key, value)
            return f

        stack = self.undo_stack
        stack.applying = True  # the per-parameter events would each push a Command; record one instead
        try:
            apply(new)()
        finally:
            stack.applying = False
        if {k: m.get(k) for k in old} != old:
            stack.push(Command(f"{m.name}: reset", apply(old), apply(new)))

    # -- what an app provides
    def windows(self) -> dict[str, Callable[[], None]]:
        """The app's dockable windows, title → function drawing the contents. Default: one ``panel()`` window."""
        return {self.title: self.panel}

    def panel(self) -> None:
        """Contents of the default single window (override ``windows()`` instead for several)."""

    def compass_axes(self) -> list[tuple[str, tuple, tuple]]:
        """Axes of the corner compass, (label, xyz direction, sRGB 0-1 colour[, length share]); apps add their own."""
        return [("x", (1.0, 0.0, 0.0), (0.93, 0.33, 0.31)), ("y", (0.0, 1.0, 0.0), (0.45, 0.80, 0.35)),
                ("z", (0.0, 0.0, 1.0), (0.35, 0.55, 0.95))]


    def menus(self) -> dict[str, Callable[[], None]]:
        """Extra menus before Edit and View, name → function drawing the items (e.g. a File menu)."""
        return {}

    # -- shared widgets

    # -- camera and lighting actions (navigation with the mouse is not undoable; these settings are)
    def _record(self, label: str, obj, keys: tuple[str, ...], mutate: Callable[[], None], merge_key=None) -> None:
        """Run ``mutate`` and push a Command that restores / re-applies ``obj``'s ``keys``."""
        old = {k: copy.copy(getattr(obj, k)) for k in keys}
        mutate()
        new = {k: copy.copy(getattr(obj, k)) for k in keys}
        if new == old:
            return

        def restore(state):
            def f():
                for k, v in state.items():
                    setattr(obj, k, copy.copy(v))
            return f

        self.undo_stack.push(Command(label, restore(old), restore(new), merge_key=merge_key))


    # -- shared windows


    # -- frame
    def all_windows(self) -> dict[str, Callable[[], None]]:
        """The app's windows plus the shared Info, Camera, Lighting, History and Console windows."""
        shared = ({"Info": self.info_window, "Camera": self.camera_window, "Lighting": self.lighting_window}
                  if self.viewer else {})
        return {**self.windows(), **shared, "History": self.history_window, "Console": self.console_window}

    def _in_bottom(self, name: str) -> bool:
        return name in self.dock_bottom or name in SHARED_WINDOWS

    def gui(self) -> None:
        """Called by the Viewer's ImguiRenderer every frame."""
        self.apply_pending_style()
        io = imgui.get_io()
        if io.key_ctrl and not io.want_text_input:
            if imgui.is_key_pressed(imgui.Key.z):
                self.undo_stack.undo()
            if imgui.is_key_pressed(imgui.Key.y):
                self.undo_stack.redo()

        wins = self.all_windows()
        self._menu_bar(wins)
        dock_id = imgui.dock_space_over_viewport(0, None, imgui.DockNodeFlags_.passthru_central_node)
        focus, self._focus = self._focus, []  # requested by last frame's layout build
        if self._layout:
            node = internal.dock_builder_get_node(dock_id)
            if self._layout == "force" or node is None or not node.is_split_node():  # nothing restored from file
                self.build_layout(dock_id, list(wins))
                bottom = [n for n in wins if self._in_bottom(n)]  # new windows grab focus in submission
                self._focus = [n for n in wins if not self._in_bottom(n)][:1] + bottom[:1]  # order: fix next frame
            self._layout = None
        central = internal.dock_builder_get_central_node(dock_id)
        self.scene_rect = (central.pos.x, central.pos.y, central.size.x, central.size.y)
        self.draw_compass()
        self.draw_boxes()

        for name, draw in wins.items():
            if not self.visible.setdefault(name, True):
                continue
            if name in focus:
                imgui.set_next_window_focus()
            expanded, still_open = imgui.begin(name, True)
            if expanded:
                draw()
            imgui.end()
            self.visible[name] = bool(still_open)
        if self.show_render:
            imgui.set_next_window_size(imgui.ImVec2(420, 0), imgui.Cond_.first_use_ever)
            expanded, self.show_render = imgui.begin("Render", True)
            if expanded:
                self.render_window()
            imgui.end()
        if self.show_preferences:  # a floating window, not part of the dock layout
            imgui.set_next_window_size(imgui.ImVec2(460, 560), imgui.Cond_.first_use_ever)
            expanded, self.show_preferences = imgui.begin("Preferences", True)
            if expanded:
                self.preferences_window()
            imgui.end()
        self.draw_toast()

    def build_layout(self, dock_id: int, names: list[str]) -> None:
        """The default layout: a right column of tabbed app windows, ``dock_bottom`` and shared windows below them,
        and ``dock_left`` windows (if any) in a left column."""
        internal.dock_builder_remove_node(dock_id)
        internal.dock_builder_add_node(
            dock_id, imgui.DockNodeFlags_.passthru_central_node | internal.DockNodeFlagsPrivate_.dock_space)
        viewport = imgui.get_main_viewport()
        internal.dock_builder_set_node_size(dock_id, viewport.work_size)
        column, lower = self.right_column or rc.ui.right_column, self.bottom_share or rc.ui.bottom_share
        left, rest = None, dock_id
        if any(name in self.dock_left for name in names):
            share = self.left_column or rc.ui.left_column
            _, left, rest = internal.dock_builder_split_node_py(dock_id, imgui.Dir.left, share)
            column = min(column / (1 - share), 0.9)  # the right column's share of what is left
        _, right, _ = internal.dock_builder_split_node_py(rest, imgui.Dir.right, column)
        _, bottom, top = internal.dock_builder_split_node_py(right, imgui.Dir.down, lower)
        for name in names:
            target = left if name in self.dock_left else (bottom if self._in_bottom(name) else top)
            internal.dock_builder_dock_window(name, target)
        internal.dock_builder_finish(dock_id)

    def reset_layout(self) -> None:
        """Restore the default dock layout and show every window again."""
        self.visible.clear()
        self._layout = "force"

    def _menu_bar(self, wins: dict) -> None:
        if not imgui.begin_main_menu_bar():
            return
        menus = self.menus()
        app_file = menus.pop("File", None)
        if imgui.begin_menu("File"):  # the app's File items, then Render
            if app_file:
                app_file()
                imgui.separator()
            if imgui.menu_item(f"{ICON.IMAGE} Render image / export script...", "", self.show_render)[0]:
                self.show_render = not self.show_render
            imgui.end_menu()
        for name, draw in menus.items():
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
            if imgui.begin_menu("UI scale"):
                for s in UI_SCALES:
                    if imgui.menu_item(f"{s:.0%}", "", s == self.ui_scale)[0]:
                        self.set_ui_scale(s)
                imgui.end_menu()
            if imgui.menu_item_simple(f"{ICON.TABLE_COLUMNS} Reset layout"):
                self.reset_layout()
            imgui.separator()
            if imgui.menu_item(f"{ICON.GEAR} Preferences", "", self.show_preferences)[0]:
                self.show_preferences = not self.show_preferences
            imgui.end_menu()
        imgui.text_disabled("   Drag: orbit   Middle/Shift-drag: pan   Wheel: zoom   Click: select   F: frame")
        imgui.end_main_menu_bar()

    # -- offline render and export (they read the scene; they don't edit it, so no undo)


    # -- preferences (configuration, not scene editing: not undoable)


