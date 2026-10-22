"""The Viewer: a rendercanvas canvas (glfw window, or offscreen for tests) + device + Renderer + camera
+ Dear ImGui with docking, and mouse navigation. Apps hand it a scene (anything with ``batches()``) and an
Editor; the scene is drawn into the editor's empty central dock node."""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from imgui_bundle import imgui
from imgui_bundle.imgui import internal
from wgpu.utils.imgui import ImguiRenderer

from ..config import rc
from ..gpu import get_device, renderer
from ..math import bounding_box, box_corners
from ..scene.camera import Camera
from ..scene.lighting import Lighting
from .editor import Editor
from .native import DEFAULT_ICON, set_icon, set_title_bar
from .navigation import Navigator
from .style import apply_style, load_fonts

EMPTY_SCENE = SimpleNamespace(batches=list)


class Viewer:
    """A window (or offscreen canvas) that renders a scene with an Editor's dockable windows around it."""
    def __init__(self, canvas=None, title: str = "valo", size: tuple[int, int] = (1360, 800), layout_file="auto",
                 icon=None):
        """``layout_file``: where the dock layout is saved; "auto" = ``~/.valo/<title>.ini`` for a window and
        nothing for a canvas passed in (tests, embedding); None = never saved. ``icon``: an image file for the
        window's icon (a square PNG; None: valo's)."""
        if layout_file == "auto":
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "valo"
            layout_file = Path.home() / ".valo" / f"{slug}.ini" if canvas is None else None
        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(title=title, size=size, update_mode="continuous", max_fps=60)
            set_icon(canvas, icon or DEFAULT_ICON)
            set_title_bar(canvas)
        self.canvas = canvas
        self.adapter, self.device = get_device()
        # Plain (non-sRGB) surface: ImGui's colours are already sRGB, the post pass encodes the scene.
        self.context = canvas.get_wgpu_context()
        surface_format = self.context.get_preferred_format(self.adapter).removesuffix("-srgb")
        self.context.configure(device=self.device, format=surface_format)

        self.renderer = renderer()  # shared: scenes built for scripting can be shown here, and back
        self.camera = Camera(fovy=rc.camera.fov)
        self.lighting = Lighting()
        self.imgui = ImguiRenderer(self.device, canvas)  # uses the format configured above
        io = imgui.get_io()  # of the context the renderer just made current
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        if layout_file:
            Path(layout_file).parent.mkdir(parents=True, exist_ok=True)
        self.layout_file = Path(layout_file) if layout_file else None
        io.set_ini_filename(str(layout_file) if layout_file else None)  # (don't read it back: NULL crashes)
        load_fonts()
        apply_style(rc.ui.scale)
        self.scene, self.editor = EMPTY_SCENE, None
        self.viewport = (0, 0, 1, 1)  # (x, y, w, h) of the scene in framebuffer pixels
        self.navigator = Navigator(self)  # moves self.camera: mouse, keys, presets, transitions
        self.pointer = None  # last pointer position, logical px
        self.hover = None  # Pick under the pointer (gpu.Pick) or None
        self.selection = None  # Pick chosen by a click, or None
        self.outlined: list = []  # selected things to box (anything with bounding_box(), e.g. a scene editor's node)
        self._last_click = None  # (time, x, y) of the last click, for double-click
        self.gui_error: str | None = None  # traceback of the last exception in the editor's gui (it was recovered)
        self._last_frame = time.perf_counter()
        canvas.add_event_handler(self.on_pointer, "pointer_down", "pointer_move", "pointer_up", "wheel")
        canvas.add_event_handler(self._on_close, "close")
        canvas.add_event_handler(self.on_key, "key_down")
        canvas.request_draw(self.draw)

    def show(self, scene, editor: Editor | None = None) -> None:
        """Draw ``scene`` from now on, with ``editor``'s windows (if any) docked around it."""
        self.scene, self.editor = scene, editor
        if editor:
            editor.viewer = self
            self.imgui.set_gui(self._gui)
        if hasattr(scene, "bounding_radius"):
            self.camera.fit(scene.bounding_radius())

    def _gui(self) -> None:
        """The editor's gui, recovered if it raises: the error is printed and kept in ``gui_error``, and ImGui's
        unfinished windows / menus are closed, so one broken widget doesn't freeze every later frame."""
        state = internal.ErrorRecoveryState()
        internal.error_recovery_store_state(state)
        try:
            self.editor.gui()
        except Exception:
            error = traceback.format_exc()
            if error != self.gui_error:
                print(error)
                self.editor.notify(error.strip().splitlines()[-1], "error", error)
            self.gui_error = error
            io = imgui.get_io()
            saved = io.config_error_recovery_enable_assert, io.config_error_recovery_enable_debug_log
            io.config_error_recovery_enable_assert = io.config_error_recovery_enable_debug_log = False  # printed above
            try:
                internal.error_recovery_try_to_recover_state(state)
            finally:
                io.config_error_recovery_enable_assert, io.config_error_recovery_enable_debug_log = saved

    def ui_click(self, x: float, y: float, button: int = 0) -> None:
        """Click the ImGui layer at (x, y) logical pixels: press and release, one frame each (tests, demos)."""
        imgui.set_current_context(self.imgui.imgui_context)
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)
        for down in (True, False):
            io.add_mouse_button_event(button, down)
            self.canvas.draw()

    def run(self) -> None:
        """Open the window and run the event loop until it is closed."""
        from rendercanvas.glfw import loop

        print(f"adapter: {self.adapter.info['device']} ({self.adapter.info['backend_type']})")
        loop.run()

    def draw(self) -> None:
        """Render one frame: the scene into the central dock node, then the ImGui windows on top."""
        texture = self.context.get_current_texture()
        ed = self.editor
        now = time.perf_counter()
        dt, self._last_frame = min(now - self._last_frame, 0.1), now  # seconds, capped after a stall
        self.navigator.step(dt)
        if ed and ed.auto_rotate:
            self.camera.azimuth = (self.camera.azimuth + rc.camera.auto_rotate_speed * dt) % 360
        for _ in range(3):  # the layout is known only after the ImGui frame: redraw if the scene rect moved
            self.viewport = self._scene_viewport(texture)
            self.renderer.render(texture, self.viewport, self.scene, self.camera, self.lighting)
            self.hover = self._pick_under_pointer()
            if not ed:
                return
            self.imgui.render()  # windows on top of the same texture
            if self._scene_viewport(texture) == self.viewport:
                return

    @staticmethod
    def _pick_box(pick) -> tuple[object, np.ndarray | None]:
        """(what the pick stands for, its box matrix): the owner's box when the picked instances belong to an object
        with ``bounding_box()``, else the box of the picked instance (its mesh bounds, turned with it)."""
        src = pick.source
        owner = getattr(src, "owner", None)
        if owner is not None and hasattr(owner, "bounding_box"):
            return owner, owner.bounding_box()
        if src is None or pick.instance_index >= len(getattr(src, "matrices", ())):
            return None, None
        m = src.matrices[pick.instance_index]
        corners = (m @ np.c_[box_corners(*src.mesh.bounds), np.ones(8)].T).T[:, :3]
        return (id(src), pick.instance_index), bounding_box(corners, m)

    def boxes(self) -> list[tuple[np.ndarray, str]]:
        """The bounding boxes to draw, (box matrix, "selected" | "hover"): the ``outlined`` things (else the clicked
        selection), and the object under the pointer unless it is already selected (``rc.highlight.enabled``)."""
        out, selected = [], set()
        sources = [(n, n.bounding_box()) for n in self.outlined] if self.outlined else (
            [self._pick_box(self.selection)] if self.selection is not None else [])
        for key, box in sources:
            if box is not None:
                out.append((box, "selected"))
                selected.add(id(key) if not isinstance(key, tuple) else key)
        if rc.highlight.enabled and self.hover is not None:
            key, box = self._pick_box(self.hover)
            if box is not None and (id(key) if not isinstance(key, tuple) else key) not in selected:
                out.append((box, "hover"))
        return out

    def _pick_under_pointer(self):
        """The Pick under the mouse pointer, if picking is on and the pointer is over the scene."""
        if not (rc.picking.enabled and self.pointer and self.editor):
            return None
        r = self.canvas.get_pixel_ratio()
        return self.renderer.pick(self.viewport, self.scene, self.camera, self.pointer[0] * r, self.pointer[1] * r)

    def _scene_viewport(self, texture) -> tuple[int, int, int, int]:
        """The editor's central dock node in framebuffer pixels (the whole window before the first layout)."""
        w, h = texture.size[:2]
        if not (self.editor and self.editor.scene_rect):
            return 0, 0, w, h
        r = self.canvas.get_pixel_ratio()
        x, y, rw, rh = (int(round(c * r)) for c in self.editor.scene_rect)
        return x, y, max(rw, 1), max(rh, 1)

    def snapshot(self) -> np.ndarray:
        """Draw two frames offscreen (ImGui lays out on the first) and return the scene region, (h, w, 3) uint8;
        message boxes are left out (they are about the app, not part of the picture)."""
        self.capturing = True
        try:
            for _ in range(2):
                img = np.asarray(self.canvas.draw())[:, :, :3]
        finally:
            self.capturing = False
        return self.crop(img).copy()

    def crop(self, image):
        """The scene part of a full-window ``image`` (e.g. from an offscreen ``canvas.draw()``)."""
        x, y, w, h = self.viewport
        return image[y : y + h, x : x + w]

    def _on_close(self, event: dict) -> None:
        """ImGui saves the layout only every few seconds: save it once more on close."""
        if self.layout_file:
            imgui.set_current_context(self.imgui.imgui_context)
            imgui.save_ini_settings_to_disk(str(self.layout_file))

    def on_pointer(self, event: dict) -> None:
        """Pointer events from the canvas: over the 3D view they navigate (``navigator.pointer``: orbit, pan, zoom),
        a click selects and a quick second click centres on the point; over a window or the menu ImGui has them."""
        imgui.set_current_context(self.imgui.imgui_context)
        kind = event["event_type"]
        if kind == "pointer_move":
            self.pointer = (event["x"], event["y"])
        if imgui.get_io().want_capture_mouse and not self.navigator.dragging:  # a window or the menu has the mouse
            self.pointer = None
            return
        rect = self.editor.scene_rect if self.editor else None
        if kind in ("pointer_down", "wheel") and rect:
            x, y, w, h = rect
            if not (x <= event["x"] < x + w and y <= event["y"] < y + h):
                return
        if self.navigator.pointer(event):
            self._click(event["x"], event["y"])

    def _click(self, x: float, y: float) -> None:
        """Select what is under the pointer (or deselect on the background); a second click soon after centres."""
        now = time.perf_counter()
        hit = self.hover
        last, self._last_click = self._last_click, (now, x, y)
        if last and (now - last[0]) * 1000 < rc.picking.double_click_ms and abs(x - last[1]) + abs(y - last[2]) < 6:
            if hit is not None:
                self.navigator.center_on(hit.world_point)
            self._last_click = None
            return
        self.selection = hit

    def on_key(self, event: dict) -> None:
        """Key presses from the canvas (unless a widget is typing): navigation keys go to the navigator, the
        deselect key (``rc.keys.deselect``) clears the selection."""
        imgui.set_current_context(self.imgui.imgui_context)
        if imgui.get_io().want_capture_keyboard:  # typing into a widget
            return
        key, mods = event.get("key", ""), event.get("modifiers", ())
        if key.lower() == rc.keys.deselect.lower():
            self.selection = None
        else:
            self.navigator.key(key, mods)

    def frame_selection(self) -> None:
        """Centre on the selected instance and zoom so it fits (eased); without a selection, fit everything."""
        self.navigator.frame_selection()

    def fit_all(self) -> None:
        """Centre on the scene and zoom so all of it fits (eased)."""
        self.navigator.fit_all()
