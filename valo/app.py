"""The Viewer: a rendercanvas canvas (glfw window, or offscreen for tests) + device + Renderer + camera
+ Dear ImGui, with mouse navigation. Apps hand it a scene (anything with ``batches()``) and an Editor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import wgpu
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from .render import OrbitCamera, Renderer
from .ui import PANEL_W, Editor
from .ui_style import apply_style, load_fonts

EMPTY_SCENE = SimpleNamespace(batches=list)


def get_device() -> tuple[wgpu.GPUAdapter, wgpu.GPUDevice]:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter, adapter.request_device_sync()


class Viewer:
    def __init__(self, canvas=None, title: str = "valo", size: tuple[int, int] = (1360, 800)):
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
        self.imgui = ImguiRenderer(self.device, canvas)  # uses the format configured above
        load_fonts()  # into the ImGui context the renderer just made current
        apply_style()
        self.scene, self.editor = EMPTY_SCENE, None
        self.viewport = (0, 0, 1, 1)  # (x, y, w, h) of the scene in framebuffer pixels
        self._drag = None
        canvas.add_event_handler(self.on_pointer, "pointer_down", "pointer_move", "pointer_up", "wheel")
        canvas.request_draw(self.draw)

    def show(self, scene, editor: Editor | None = None) -> None:
        """Draw ``scene`` from now on, with ``editor``'s panel (if any) on the right."""
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
        panel_px = int(PANEL_W * self.canvas.get_pixel_ratio()) if ed and ed.docked else 0
        w, h = texture.size[:2]
        self.viewport = (0, 0, max(w - panel_px, 1), h)
        self.renderer.render(texture, self.viewport, self.scene, self.camera)
        if ed:
            self.imgui.render()  # panel on top of the same texture

    @property
    def viewport_width(self) -> int:
        return self.viewport[2]

    def crop(self, image):
        """The scene part of a full-window ``image`` (e.g. from an offscreen ``canvas.draw()``)."""
        x, y, w, h = self.viewport
        return image[y : y + h, x : x + w]

    def on_pointer(self, event: dict) -> None:
        if imgui.get_io().want_capture_mouse:  # the panel has the mouse
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
            cam.distance = float(np.clip(cam.distance * 1.0015 ** event["dy"], 2.0, 120.0))
