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

from .lighting import MODES
from .materials import SCHEMA, SHADINGS, Material, linear_to_srgb, srgb_to_linear
from .render import PROJECTIONS, VIEWS
from .ui_style import ICON, apply_style
from .undo import Command, UndoStack

UI_SCALES = (0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
SHARED_WINDOWS = ("Camera", "Lighting", "History")  # added by the base class, docked in the lower part


class Editor:
    """Base class for an app's editor: undoable actions, dockable windows and the shared UI."""
    title = "valo"
    right_column = 0.3  # share of the window width for the right dock column
    dock_bottom: tuple[str, ...] = ()  # app windows docked under the right column, before the shared ones
    bottom_share = 0.42  # height share of that lower part (Camera, Lighting, History live there)

    def __init__(self):
        self.undo_stack = UndoStack()
        self.auto_rotate = False
        self.viewer = None  # set by Viewer.show(): camera, lighting and scene
        self.ui_scale = 1.0
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

    def menus(self) -> dict[str, Callable[[], None]]:
        """Extra menus before Edit and View, name → function drawing the items (e.g. a File menu)."""
        return {}

    # -- shared widgets
    def material_section(self, m: Material) -> None:
        """Shading picker and schema-driven parameter widgets for material ``m``."""
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

    def set_projection(self, projection: str) -> None:
        """Switch the camera between perspective and orthographic (undoable)."""
        cam = self.viewer.camera
        self._record(f"camera: {projection}", cam, ("projection", "distance", "ortho_height"),
                     lambda: cam.set_projection(projection))

    def set_fov(self, degrees: float) -> None:
        """Set the perspective field of view in degrees (undoable)."""
        cam = self.viewer.camera
        self._record("camera: field of view", cam, ("fovy",), lambda: setattr(cam, "fovy", float(degrees)),
                     merge_key=("camera", "fovy"))

    def set_view(self, name: str) -> None:
        """Look from a preset side: one of ``render.VIEWS`` (+X ... -Z, Iso)."""
        cam = self.viewer.camera

        def look():
            cam.azimuth, cam.elevation = VIEWS[name]

        self._record(f"camera: view {name}", cam, ("azimuth", "elevation"), look)

    def fit_view(self) -> None:
        """Move the camera so the whole scene fits (undoable)."""
        cam, scene = self.viewer.camera, self.viewer.scene
        if hasattr(scene, "bounding_radius"):
            self._record("camera: fit", cam, ("distance", "ortho_height"), lambda: cam.fit(scene.bounding_radius()))

    def set_light(self, key: str, value) -> None:
        """Any ``Lighting`` field except ``mode``: azimuth, elevation, color, intensity, sky, ground, ambient."""
        light = self.viewer.lighting
        value = tuple(float(c) for c in value) if isinstance(value, (tuple, list)) else float(value)
        self._record(f"light: {key}", light, (key,), lambda: setattr(light, key, value), merge_key=("light", key))

    def set_light_mode(self, mode: str) -> None:
        """Switch the light between headlight and fixed without moving it (undoable)."""
        light, cam = self.viewer.lighting, self.viewer.camera
        self._record(f"light: {mode}", light, ("mode", "azimuth", "elevation"), lambda: light.set_mode(mode, cam))

    def set_flat_ambient(self, rgb) -> None:
        """One ambient colour from every side (sky = ground)."""
        light, rgb = self.viewer.lighting, tuple(float(c) for c in rgb)

        def flat():
            light.sky = light.ground = rgb

        self._record("light: ambient colour", light, ("sky", "ground"), flat, merge_key=("light", "flat"))

    # -- shared windows
    def camera_window(self) -> None:
        """Projection, field of view, view presets, fit and auto-rotate."""
        cam = self.viewer.camera
        for p in PROJECTIONS:
            if imgui.radio_button(p.capitalize(), cam.projection == p):
                self.set_projection(p)
            imgui.same_line()
        imgui.new_line()
        if cam.projection == "perspective":
            changed, v = imgui.slider_float("Field of view", cam.fovy, 10.0, 90.0, "%.0f deg")
            if changed:
                self.set_fov(v)
        imgui.text_disabled("View from")
        for i, name in enumerate(VIEWS):
            if i:
                imgui.same_line()
            if imgui.button(name):
                self.set_view(name)
        if imgui.button(f"{ICON.EXPAND} Fit"):
            self.fit_view()
        imgui.same_line()
        _, self.auto_rotate = imgui.checkbox("Auto-rotate", self.auto_rotate)

    def _color(self, label: str, rgb) -> tuple[bool, list[float]]:
        """sRGB colour swatch for a linear colour."""
        changed, col = imgui.color_edit3(label, [linear_to_srgb(c) for c in rgb], imgui.ColorEditFlags_.no_inputs)
        return changed, [srgb_to_linear(c) for c in col]

    def lighting_window(self) -> None:
        """Light mode, direction, colour and intensity, and the global ambient."""
        light = self.viewer.lighting
        for m in MODES:
            if imgui.radio_button(m.capitalize(), light.mode == m):
                self.set_light_mode(m)
            imgui.same_line()
        imgui.new_line()
        imgui.text_disabled("moves with the camera" if light.mode == "headlight" else "stays put in the world")
        for key, lo, hi in (("azimuth", -180.0, 180.0), ("elevation", -90.0, 90.0)):
            changed, v = imgui.slider_float(key.capitalize(), getattr(light, key), lo, hi, "%.0f deg")
            if changed:
                self.set_light(key, v)
        changed, col = self._color("##light colour", light.color)
        if changed:
            self.set_light("color", col)
        imgui.same_line()
        changed, v = imgui.slider_float("Light", light.intensity, 0.0, 3.0, "%.2f")
        if changed:
            self.set_light("intensity", v)

        imgui.separator_text("Ambient (all objects)")
        flat = tuple(light.sky) == tuple(light.ground)
        changed, col = self._color("Sky" if not flat else "Colour", light.sky)
        if changed:
            if flat:
                self.set_flat_ambient(col)
            else:
                self.set_light("sky", col)
        if not flat:
            imgui.same_line()
            changed, col = self._color("Ground", light.ground)
            if changed:
                self.set_light("ground", col)
        imgui.same_line()
        if imgui.checkbox("Flat", flat)[0] and not flat:
            self.set_flat_ambient(light.sky)
        changed, v = imgui.slider_float("Ambient", light.ambient, 0.0, 1.5, "%.2f")
        if changed:
            self.set_light("ambient", v)

    def history_window(self) -> None:
        """Undo / redo buttons and the list of recorded edits."""
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
        """The app's windows plus the shared Camera, Lighting and History windows."""
        shared = {"Camera": self.camera_window, "Lighting": self.lighting_window} if self.viewer else {}
        return {**self.windows(), **shared, "History": self.history_window}

    def _in_bottom(self, name: str) -> bool:
        return name in self.dock_bottom or name in SHARED_WINDOWS

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

    def build_layout(self, dock_id: int, names: list[str]) -> None:
        """The default layout: a right column of tabbed app windows; ``dock_bottom`` and shared windows below."""
        internal.dock_builder_remove_node(dock_id)
        internal.dock_builder_add_node(
            dock_id, imgui.DockNodeFlags_.passthru_central_node | internal.DockNodeFlagsPrivate_.dock_space)
        viewport = imgui.get_main_viewport()
        internal.dock_builder_set_node_size(dock_id, viewport.work_size)
        _, right, _ = internal.dock_builder_split_node_py(dock_id, imgui.Dir.right, self.right_column)
        _, bottom, top = internal.dock_builder_split_node_py(right, imgui.Dir.down, self.bottom_share)
        for name in names:
            internal.dock_builder_dock_window(name, bottom if self._in_bottom(name) else top)
        internal.dock_builder_finish(dock_id)

    def reset_layout(self) -> None:
        """Restore the default dock layout and show every window again."""
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
        """Scale text and spacing of the whole UI."""
        self.ui_scale = scale
        apply_style(scale)
