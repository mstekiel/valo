"""The shared Camera window: projection, field of view, view presets and fit. They are view changes, made by the
viewer's ``Navigator`` and not recorded for undo (like Blender's view navigation)."""

from __future__ import annotations

from imgui_bundle import imgui

from ...config import rc
from ...scene.camera import PROJECTIONS, VIEWS
from ..style import ICON
from ..widgets import SliderSpec


class CameraWindow:
    """Editor part: the Camera window and the view operations it offers (not recorded for undo)."""

    def set_projection(self, projection: str) -> None:
        """Switch the view between perspective and orthographic, keeping the scene's size on screen."""
        self.viewer.navigator.set_projection(projection)

    def set_fov(self, degrees: float) -> None:
        """Set the perspective field of view in degrees, keeping the target's size on screen."""
        self.viewer.navigator.set_fov(degrees)

    def set_view(self, name: str) -> None:
        """Look from a preset side: one of ``VIEWS`` (+X ... -Z, Iso)."""
        self.viewer.navigator.set_view(name)

    def fit_view(self) -> None:
        """Zoom so the whole scene fits."""
        self.viewer.navigator.fit_view()

    def camera_window(self) -> None:
        """Projection, field of view, view presets, fit and auto-rotate."""
        cam = self.viewer.camera
        for p in PROJECTIONS:
            if imgui.radio_button(p.capitalize(), cam.projection == p):
                self.set_projection(p)
            imgui.same_line()
        imgui.new_line()
        if cam.projection == "perspective":
            spec = SliderSpec(10.0, 90.0, rc.camera.fov, hard_lo=1.0, hard_hi=170.0, fmt="%.0f deg")
            changed, v = self.slider("camera.fov", "Field of view", cam.fovy, spec)
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
        imgui.begin_disabled(self.viewer.selection is None)
        if imgui.button(f"{ICON.CROSSHAIRS} Frame selection"):
            self.viewer.frame_selection()
        imgui.end_disabled()
        imgui.same_line()
        _, self.auto_rotate = imgui.checkbox("Auto-rotate", self.auto_rotate)
