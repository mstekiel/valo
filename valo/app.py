"""The Viewer: a rendercanvas canvas (glfw window, or offscreen for tests) + device + Renderer + camera
+ Dear ImGui with docking, and mouse navigation. Apps hand it a scene (anything with ``batches()``) and an
Editor; the scene is drawn into the editor's empty central dock node."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wgpu
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from .lighting import Lighting
from .render import OrbitCamera, Renderer
from .ui import Editor
from .ui_style import apply_style, load_fonts

EMPTY_SCENE = SimpleNamespace(batches=list)


def get_device() -> tuple[wgpu.GPUAdapter, wgpu.GPUDevice]:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter, adapter.request_device_sync()


class Viewer:
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
        self.camera = OrbitCamera()
        self.lighting = Lighting()
        self.imgui = ImguiRenderer(self.device, canvas)  # uses the format configured above
        io = imgui.get_io()  # of the context the renderer just made current
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        if layout_file:
            Path(layout_file).parent.mkdir(parents=True, exist_ok=True)
        self.layout_file = Path(layout_file) if layout_file else None
        io.set_ini_filename(str(layout_file) if layout_file else None)  # (don't read it back: NULL crashes)
        load_fonts()
        apply_style()
        self.scene, self.editor = EMPTY_SCENE, None
        self.viewport = (0, 0, 1, 1)  # (x, y, w, h) of the scene in framebuffer pixels
        self._drag = None
        canvas.add_event_handler(self.on_pointer, "pointer_down", "pointer_move", "pointer_up", "wheel")
        canvas.add_event_handler(self._on_close, "close")
        canvas.request_draw(self.draw)

    def show(self, scene, editor: Editor | None = None) -> None:
        """Draw ``scene`` from now on, with ``editor``'s windows (if any) docked around it."""
        self.scene, self.editor = scene, editor
        if editor:
            self.imgui.set_gui(editor.gui)
        if hasattr(scene, "bounding_radius"):
            self.camera.fit(scene.bounding_radius())

    def run(self) -> None:
        from rendercanvas.glfw import loop

        print(f"adapter: {self.adapter.info['device']} ({self.adapter.info['backend_type']})")
        loop.run()

    def draw(self) -> None:
        texture = self.context.get_current_texture()
        ed = self.editor
        if ed and ed.auto_rotate:
            self.camera.azimuth = (self.camera.azimuth + 0.4) % 360
        for _ in range(3):  # the layout is known only after the ImGui frame: redraw if the scene rect moved
            self.viewport = self._scene_viewport(texture)
            self.renderer.render(texture, self.viewport, self.scene, self.camera, self.lighting)
            if not ed:
                return
            self.imgui.render()  # windows on top of the same texture
            if self._scene_viewport(texture) == self.viewport:
                return

    def _scene_viewport(self, texture) -> tuple[int, int, int, int]:
        """The editor's central dock node in framebuffer pixels (the whole window before the first layout)."""
        w, h = texture.size[:2]
        if not (self.editor and self.editor.scene_rect):
            return 0, 0, w, h
        r = self.canvas.get_pixel_ratio()
        x, y, rw, rh = (int(round(c * r)) for c in self.editor.scene_rect)
        return x, y, max(rw, 1), max(rh, 1)

    @property
    def viewport_width(self) -> int:
        return self.viewport[2]

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
        imgui.set_current_context(self.imgui.imgui_context)
        if imgui.get_io().want_capture_mouse:  # a window or the menu has the mouse
            return
        rect = self.editor.scene_rect if self.editor else None
        if event["event_type"] in ("pointer_down", "wheel") and rect:
            x, y, w, h = rect
            if not (x <= event["x"] < x + w and y <= event["y"] < y + h):
                return
        cam, kind = self.camera, event["event_type"]
        if kind == "pointer_down" and event.get("button") == 1:
            self._drag = (event["x"], event["y"], cam.azimuth, cam.elevation)
        elif kind == "pointer_up":
            self._drag = None
        elif kind == "pointer_move" and self._drag:
            x0, y0, az0, el0 = self._drag
            cam.azimuth = az0 - 0.4 * (event["x"] - x0)
            cam.elevation = float(np.clip(el0 + 0.3 * (event["y"] - y0), -85, 85))
        elif kind == "wheel":
            cam.zoom(1.0015 ** event["dy"])
