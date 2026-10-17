"""The Viewer: a rendercanvas canvas (glfw window, or offscreen for tests) + device + Renderer + camera
+ Dear ImGui with docking, and mouse navigation. Apps hand it a scene (anything with ``batches()``) and an
Editor; the scene is drawn into the editor's empty central dock node."""

from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wgpu
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from .config import rc
from .lighting import Lighting
from .render import VIEWS, OrbitCamera, Renderer
from .ui import Editor
from .ui_style import apply_style, load_fonts

EMPTY_SCENE = SimpleNamespace(batches=list)


def get_device() -> tuple[wgpu.GPUAdapter, wgpu.GPUDevice]:
    """The high-performance GPU adapter and a device on it."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter, adapter.request_device_sync()


class Viewer:
    """A window (or offscreen canvas) that renders a scene with an Editor's dockable windows around it."""
    def __init__(self, canvas=None, title: str = "valo", size: tuple[int, int] = (1360, 800), layout_file="auto"):
        """``layout_file``: where the dock layout is saved; "auto" = ``~/.valo/<title>.ini`` for a window and
        nothing for a canvas passed in (tests, embedding); None = never saved."""
        if layout_file == "auto":
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "valo"
            layout_file = Path.home() / ".valo" / f"{slug}.ini" if canvas is None else None
        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(title=title, size=size, update_mode="continuous", max_fps=60)
        self.canvas = canvas
        self.adapter, self.device = get_device()
        # Plain (non-sRGB) surface: ImGui's colours are already sRGB, the post pass encodes the scene.
        self.context = canvas.get_wgpu_context()
        surface_format = self.context.get_preferred_format(self.adapter).removesuffix("-srgb")
        self.context.configure(device=self.device, format=surface_format)

        self.renderer = Renderer(self.device, surface_format)
        self.camera = OrbitCamera(fovy=rc.camera.fov)
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
        self._drag = None
        self.pointer = None  # last pointer position, logical px
        self.hover = None  # Pick under the pointer (render.Pick) or None
        self.selection = None  # Pick chosen by a click, or None
        self._last_click = None  # (time, x, y) of the last click, for double-click
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
            self.imgui.set_gui(editor.gui)
        if hasattr(scene, "bounding_radius"):
            self.camera.fit(scene.bounding_radius())

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
        self.camera.step(dt)
        if ed and ed.auto_rotate:
            self.camera.azimuth = (self.camera.azimuth + rc.camera.auto_rotate_speed * dt) % 360
        for _ in range(3):  # the layout is known only after the ImGui frame: redraw if the scene rect moved
            self.viewport = self._scene_viewport(texture)
            self.renderer.render(texture, self.viewport, self.scene, self.camera, self.lighting, self._highlights())
            self.hover = self._pick_under_pointer()
            if not ed:
                return
            self.imgui.render()  # windows on top of the same texture
            if self._scene_viewport(texture) == self.viewport:
                return

    def _highlights(self) -> list:
        """(batch, instance, rgb) tints for the renderer: the selection, and the hovered object (last frame's pick)."""
        h = rc.highlight
        if not h.enabled:
            return []
        out = []
        for pick, strength in ((self.selection, h.selected_strength), (self.hover, h.strength)):
            key = pick and (pick.batch_index, pick.instance_index)
            if pick is not None and key not in [o[:2] for o in out]:
                out.append((*key, tuple(c * strength for c in h.color)))
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
        """Draw two frames offscreen (ImGui lays out on the first) and return the scene region, (h, w, 3) uint8."""
        for _ in range(2):
            img = np.asarray(self.canvas.draw())[:, :, :3]
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
        """Orbit (left-drag), pan (middle-drag or Shift+left-drag), zoom (wheel), select (click), centre
        (double-click), when the pointer is over the scene."""
        imgui.set_current_context(self.imgui.imgui_context)
        kind = event["event_type"]
        if kind == "pointer_move":
            self.pointer = (event["x"], event["y"])
        if imgui.get_io().want_capture_mouse and not self._drag:  # a window or the menu has the mouse
            self.pointer = None
            return
        rect = self.editor.scene_rect if self.editor else None
        if kind in ("pointer_down", "wheel") and rect:
            x, y, w, h = rect
            if not (x <= event["x"] < x + w and y <= event["y"] < y + h):
                return
        cam = self.camera
        if kind == "pointer_down" and event.get("button") in (1, 3):
            pan = event.get("button") == 3 or "Shift" in event.get("modifiers", ())
            self._drag = ("pan" if pan else "orbit", event["x"], event["y"], cam.azimuth, cam.elevation, cam.target)
        elif kind == "pointer_up" and self._drag:
            mode, x0, y0, *_ = self._drag
            self._drag = None
            if mode == "orbit" and abs(event["x"] - x0) + abs(event["y"] - y0) < 4:
                self._click(event["x"], event["y"])
        elif kind == "pointer_move" and self._drag:
            mode, x0, y0, az0, el0, target0 = self._drag
            dx, dy = event["x"] - x0, event["y"] - y0
            if mode == "orbit":
                speed = rc.camera.orbit_speed
                cam.azimuth = az0 - speed * dx
                cam.elevation = float(np.clip(el0 + speed * dy, -85, 85))
            else:
                r = self.canvas.get_pixel_ratio()
                upp = cam.world_per_pixel(self.viewport[3]) * r * rc.camera.pan_speed
                cam.target = target0
                cam.pan(dx * upp, dy * upp)
        elif kind == "wheel":
            cam.zoom(1.0015 ** (event["dy"] * rc.camera.zoom_speed))

    def _click(self, x: float, y: float) -> None:
        """Select what is under the pointer (or deselect on the background); a second click soon after centres."""
        now = time.perf_counter()
        hit = self.hover
        last, self._last_click = self._last_click, (now, x, y)
        if last and (now - last[0]) * 1000 < rc.picking.double_click_ms and abs(x - last[1]) + abs(y - last[2]) < 6:
            if hit is not None:
                self.camera.center_on(hit.world_point)
            self._last_click = None
            return
        self.selection = hit

    def on_key(self, event: dict) -> None:
        """Keyboard navigation (``rc.keys``): frame, fit, view presets, projection, orbit/pan steps, deselect."""
        imgui.set_current_context(self.imgui.imgui_context)
        if imgui.get_io().want_capture_keyboard:  # typing into a widget
            return
        key, mods = event.get("key", ""), event.get("modifiers", ())
        k = rc.keys

        def pressed(binding: str) -> bool:
            return key.lower() == binding.lower()

        cam, ctrl, shift = self.camera, "Control" in mods, "Shift" in mods
        if pressed(k.frame_selection):
            self.frame_selection()
        elif pressed(k.fit_all):
            self.fit_all()
        elif pressed(k.view_front):
            self._view("+Y" if ctrl else "-Y")
        elif pressed(k.view_right):
            self._view("-X" if ctrl else "+X")
        elif pressed(k.view_top):
            self._view("-Z" if ctrl else "+Z")
        elif pressed(k.toggle_ortho):
            other = "perspective" if cam.projection == "orthographic" else "orthographic"
            self.editor.set_projection(other) if self.editor else cam.set_projection(other)
        elif pressed(k.deselect):
            self.selection = None
        else:
            steps = {k.orbit_left: (-1, 0), k.orbit_right: (1, 0), k.orbit_up: (0, 1), k.orbit_down: (0, -1)}
            for binding, (sx, sy) in steps.items():
                if pressed(binding):
                    if shift:  # pan by a tenth of the view
                        step = cam.world_per_pixel(self.viewport[3]) * self.viewport[3] / 10
                        cam.pan(-sx * step, sy * step)
                    else:
                        cam.azimuth = (cam.azimuth + sx * rc.camera.orbit_step) % 360
                        cam.elevation = float(np.clip(cam.elevation + sy * rc.camera.orbit_step, -89.99, 89.99))
                    break

    def _view(self, name: str) -> None:
        if self.editor:
            self.editor.set_view(name)
        else:
            self.camera.azimuth, self.camera.elevation = VIEWS[name]

    def frame_selection(self) -> None:
        """Centre on the selected instance and zoom so it fits (animated); without a selection, fit everything."""
        sel = self.selection
        if sel is None or sel.source is None:
            return self.fit_all()
        m = sel.source.matrices[sel.instance_index]
        radius = sel.source.mesh.radius * float(np.abs(np.linalg.norm(m[:3, :3], axis=0)).max())
        self.camera.center_on(m[:3, 3], radius=max(radius, 1e-3))

    def fit_all(self) -> None:
        """Centre on the scene and zoom so all of it fits (animated)."""
        scene = self.scene
        center = scene.center() if hasattr(scene, "center") else (0.0, 0.0, 0.0)
        radius = scene.bounding_radius() if hasattr(scene, "bounding_radius") else None
        self.camera.center_on(center, radius=radius)
