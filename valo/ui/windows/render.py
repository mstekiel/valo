"""The shared Render window: offline images of the current view and runnable render scripts."""

from __future__ import annotations

import time
from pathlib import Path

from imgui_bundle import imgui

from ...gpu.material import linear_to_srgb, srgb_to_linear
from ...offline import export_script, image_settings, plain, render
from ..style import ICON


class RenderWindow:
    """Editor part: Render image / Export script (they read the scene, they don't edit it: no undo)."""

    def render_settings(self) -> dict:
        """Camera, lighting and image options as plain data, plus the scene's ``settings()`` when it has them."""
        viewer = self.viewer
        out = {"camera": viewer.camera.settings(), "lighting": viewer.lighting.settings(),
               "image": image_settings(None, **self.render_options)}
        if self.can_export():
            out = {"scene": viewer.scene.settings(), **out}
        return plain(out)

    def _output(self, suffix: str) -> Path:
        name = type(self.viewer.scene).__name__.lower()
        return Path.cwd() / f"render_{name}_{time.strftime('%Y%m%d-%H%M%S')}{suffix}"

    def render_image(self, path=None) -> Path:
        """Render the scene offline as it is on screen (with the Render window's image options) and save a PNG."""
        path = Path(path) if path else self._output(".png")
        settings = self.render_settings()
        render(self.viewer.scene, settings, path=path)
        self.notify(f"rendered {path.name}", detail=str(path))
        return path

    def export_script(self, path=None) -> Path:
        """Write a runnable script that re-renders this picture offline (needs a scene with ``settings()``)."""
        path = Path(path) if path else self._output(".py")
        export_script(path, self.viewer.scene, self.render_settings())
        self.notify(f"exported {path.name}", detail=str(path))
        return path

    def can_export(self) -> bool:
        """True when the scene can describe itself (``settings()`` / ``from_settings()``)."""
        scene = self.viewer.scene if self.viewer else None
        return hasattr(scene, "settings") and hasattr(type(scene), "from_settings")

    def render_window(self) -> None:
        """Image options for offline rendering, and the Render / Export buttons."""
        opt = self.render_options
        changed, size = imgui.input_int2("Size (px)", list(opt["size"]))
        if changed:
            opt["size"] = [max(16, min(16384, int(c))) for c in size]
        _, opt["samples"] = imgui.slider_int("Supersampling", int(opt["samples"]), 1, 4, "%dx")
        _, opt["transparent"] = imgui.checkbox("Transparent background", bool(opt["transparent"]))
        if not opt["transparent"]:
            changed, col = imgui.color_edit3("Background", [linear_to_srgb(c) for c in opt["background"]],
                                             imgui.ColorEditFlags_.no_inputs)
            if changed:
                opt["background"] = [srgb_to_linear(c) for c in col]
        imgui.text_disabled("Camera and lighting: as in the 3D view")
        if imgui.button(f"{ICON.IMAGE} Render PNG"):
            self.render_image()
        imgui.same_line()
        imgui.begin_disabled(not self.can_export())
        if imgui.button(f"{ICON.FILE_CODE} Export render script"):
            self.export_script()
        imgui.end_disabled()
