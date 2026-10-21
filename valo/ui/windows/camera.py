"""The shared Camera window: projection, field of view, view presets and fit, and the actions behind them."""

from __future__ import annotations

from imgui_bundle import imgui

from ...render import PROJECTIONS, VIEWS
from ..style import ICON


class CameraWindow:
    """Editor part: the Camera window and its actions (projection, field of view, view presets, fit view)."""

    def set_projection(self, projection: str) -> None:
        """Switch the camera between perspective and orthographic (undoable)."""
        cam, scene = self.viewer.camera, self.viewer.scene
        radius = scene.bounding_radius() if hasattr(scene, "bounding_radius") else None
        self._record(f"camera: {projection}", cam, ("projection", "distance", "ortho_height"),
                     lambda: cam.set_projection(projection, radius))

    def set_fov(self, degrees: float) -> None:
        """Set the perspective field of view in degrees, keeping the target's size on screen (undoable)."""
        cam, scene = self.viewer.camera, self.viewer.scene
        radius = scene.bounding_radius() if hasattr(scene, "bounding_radius") else None
        self._record("camera: field of view", cam, ("fovy", "distance"), lambda: cam.set_fov(degrees, radius),
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
        imgui.begin_disabled(self.viewer.selection is None)
        if imgui.button(f"{ICON.CROSSHAIRS} Frame selection"):
            self.viewer.frame_selection()
        imgui.end_disabled()
        imgui.same_line()
        _, self.auto_rotate = imgui.checkbox("Auto-rotate", self.auto_rotate)
