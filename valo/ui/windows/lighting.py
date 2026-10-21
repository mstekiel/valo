"""The shared Lighting window: headlight or fixed light, its colour and intensity, and the hemisphere ambient."""

from __future__ import annotations

from imgui_bundle import imgui

from ...config import rc
from ...scene.lighting import MODES


class LightingWindow:
    """Editor part: the Lighting window and its undoable actions on ``viewer.lighting``."""

    def set_light(self, key: str, value) -> None:
        """Any ``Lighting`` field except ``mode``: azimuth, elevation, color, intensity, sky, ground, ambient."""
        light = self.viewer.lighting
        value = tuple(float(c) for c in value) if isinstance(value, (tuple, list)) else float(value)
        self._record(f"light: {key}", light, (key,), lambda: setattr(light, key, value), merge_key=("light", key))

    def set_light_mode(self, mode: str) -> None:
        """Switch the light between headlight and fixed without moving it (undoable)."""
        light, cam = self.viewer.lighting, self.viewer.camera
        self._record(f"light: {mode}", light, ("mode", "azimuth", "elevation"), lambda: light.set_mode(mode, cam))

    def set_ambient_flat(self, flat: bool) -> None:
        """Flat ambient on (ground = sky) or off (the ground colour from before comes back); undoable."""
        light = self.viewer.lighting
        if flat == (tuple(light.sky) == tuple(light.ground)):
            return
        if flat:
            self._ground_before_flat = tuple(light.ground)
            self.set_flat_ambient(light.sky)
        else:
            ground = self._ground_before_flat or rc.lighting.ground
            if tuple(ground) == tuple(light.sky):  # e.g. it started flat: any distinct ground will do
                ground = tuple(0.2 * c for c in light.sky)
            self.set_light("ground", ground)

    def set_flat_ambient(self, rgb) -> None:
        """One ambient colour from every side (sky = ground)."""
        light, rgb = self.viewer.lighting, tuple(float(c) for c in rgb)

        def flat():
            light.sky = light.ground = rgb

        self._record("light: ambient colour", light, ("sky", "ground"), flat, merge_key=("light", "flat"))

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
        clicked, want_flat = imgui.checkbox("Flat", flat)
        if clicked:
            self.set_ambient_flat(want_flat)
        changed, v = imgui.slider_float("Ambient", light.ambient, 0.0, 1.5, "%.2f")
        if changed:
            self.set_light("ambient", v)
