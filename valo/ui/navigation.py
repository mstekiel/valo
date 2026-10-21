"""Navigation: how the user moves the 3D view (Blender's view navigation). Mouse orbit / pan / zoom, the navigation keys
of ``rc.keys``, view presets, perspective / orthographic, framing, and smooth transitions between views.

The ``Camera`` is plain view state; the ``Navigator`` changes it. View changes are not edits: like in Blender they
never enter the undo history.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import rc
from ..scene.camera import VIEWS


class Navigator:
    """Moves ``viewer.camera``: drags, wheel and keys from the Viewer, view operations from the Camera window."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._drag = None  # (mode, x0, y0, azimuth0, elevation0, target0) while a mouse button drags the view
        self._anim = None  # running transition: (target0, target1, d0, d1, h0, h1, elapsed, duration)

    @property
    def camera(self):
        """The camera being moved (the viewer's, which apps may replace)."""
        return self.viewer.camera

    @property
    def dragging(self) -> bool:
        """True while a mouse drag orbits or pans the view."""
        return self._drag is not None

    def _scene_radius(self):
        scene = self.viewer.scene
        return scene.bounding_radius() if hasattr(scene, "bounding_radius") else None

    # -- transitions
    def center_on(self, target, radius: float | None = None, duration: float | None = None) -> None:
        """Move the view's target to ``target`` (and, given ``radius``, zoom so that sphere fits), eased over
        ``duration`` seconds (default ``rc.camera.center_duration``; 0 jumps). ``step(dt)`` advances it."""
        cam = self.camera
        distance = cam.fit_distance(radius) if radius else cam.distance
        height = 2 * distance * math.tan(math.radians(cam.fovy / 2)) if radius else cam.ortho_height
        duration = rc.camera.center_duration if duration is None else duration
        self._anim = (np.asarray(cam.target, float), np.asarray(target, float), cam.distance, distance,
                      cam.ortho_height, height, 0.0, max(duration, 0.0))
        self.step(0.0)

    @property
    def animating(self) -> bool:
        """True while a ``center_on`` transition is running."""
        return self._anim is not None

    def step(self, dt: float) -> None:
        """Advance a running transition by ``dt`` seconds (smooth ease in and out)."""
        if self._anim is None:
            return
        cam = self.camera
        t0, t1, d0, d1, h0, h1, elapsed, duration = self._anim
        elapsed += dt
        u = 1.0 if duration == 0 or elapsed >= duration else elapsed / duration
        e = u * u * (3 - 2 * u)  # smoothstep
        cam.target = tuple(float(c) for c in t0 + (t1 - t0) * e)
        cam.distance, cam.ortho_height = d0 + (d1 - d0) * e, h0 + (h1 - h0) * e
        self._anim = None if u >= 1.0 else (t0, t1, d0, d1, h0, h1, elapsed, duration)

    # -- view operations (keys and the Camera window)
    def set_projection(self, projection: str) -> None:
        """Perspective or orthographic, keeping the scene the same size on screen."""
        self.camera.set_projection(projection, self._scene_radius())

    def toggle_projection(self) -> None:
        """Switch between perspective and orthographic."""
        self.set_projection("perspective" if self.camera.projection == "orthographic" else "orthographic")

    def set_fov(self, degrees: float) -> None:
        """The perspective field of view in degrees, keeping the target's size on screen."""
        self.camera.set_fov(degrees, self._scene_radius())

    def set_view(self, name: str) -> None:
        """Look from a preset side: one of ``VIEWS`` (+X ... -Z, Iso)."""
        self.camera.azimuth, self.camera.elevation = VIEWS[name]

    def fit_view(self) -> None:
        """Zoom so the whole scene fits, keeping the target."""
        radius = self._scene_radius()
        if radius:
            self.camera.fit(radius)

    def fit_all(self) -> None:
        """Centre on the scene and zoom so all of it fits (eased)."""
        scene = self.viewer.scene
        center = scene.center() if hasattr(scene, "center") else (0.0, 0.0, 0.0)
        self.center_on(center, radius=self._scene_radius())

    def frame_selection(self) -> None:
        """Centre on the selected instance and zoom so it fits (eased); without a selection, fit everything."""
        sel = self.viewer.selection
        if sel is None or sel.source is None:
            return self.fit_all()
        m = sel.source.matrices[sel.instance_index]
        radius = sel.source.mesh.radius * float(np.abs(np.linalg.norm(m[:3, :3], axis=0)).max())
        self.center_on(m[:3, 3], radius=max(radius, 1e-3))

    def orbit_step(self, sx: int, sy: int) -> None:
        """Orbit by ``rc.camera.orbit_step`` degrees, ``sx`` steps around and ``sy`` steps up."""
        cam = self.camera
        cam.azimuth = (cam.azimuth + sx * rc.camera.orbit_step) % 360
        cam.elevation = float(np.clip(cam.elevation + sy * rc.camera.orbit_step, -89.99, 89.99))

    def pan_step(self, sx: int, sy: int) -> None:
        """Pan by a tenth of the view, ``sx`` steps right and ``sy`` steps up."""
        h = self.viewer.viewport[3]
        step = self.camera.world_per_pixel(h) * h / 10
        self.camera.pan(-sx * step, sy * step)

    # -- input
    def pointer(self, event: dict) -> bool:
        """Mouse navigation for one event already known to be over the 3D view: left-drag orbits, middle- or
        Shift+left-drag pans, the wheel zooms. Returns True when a left press was released without moving (a click,
        which the viewer turns into a selection)."""
        cam, kind = self.camera, event["event_type"]
        if kind == "pointer_down" and event.get("button") in (1, 3):
            pan = event.get("button") == 3 or "Shift" in event.get("modifiers", ())
            self._drag = ("pan" if pan else "orbit", event["x"], event["y"], cam.azimuth, cam.elevation, cam.target)
        elif kind == "pointer_up" and self._drag:
            mode, x0, y0, *_ = self._drag
            self._drag = None
            return mode == "orbit" and abs(event["x"] - x0) + abs(event["y"] - y0) < 4
        elif kind == "pointer_move" and self._drag:
            mode, x0, y0, az0, el0, target0 = self._drag
            dx, dy = event["x"] - x0, event["y"] - y0
            if mode == "orbit":
                speed = rc.camera.orbit_speed
                cam.azimuth = az0 - speed * dx
                cam.elevation = float(np.clip(el0 + speed * dy, -85, 85))
            else:
                r = self.viewer.canvas.get_pixel_ratio()
                upp = cam.world_per_pixel(self.viewer.viewport[3]) * r * rc.camera.pan_speed
                cam.target = target0
                cam.pan(dx * upp, dy * upp)
        elif kind == "wheel":
            cam.zoom(1.0015 ** (event["dy"] * rc.camera.zoom_speed))
        return False

    def key(self, key: str, modifiers=()) -> bool:
        """Keyboard navigation (``rc.keys``): frame selection, fit all, views 1 / 3 / 7 (Ctrl: the opposite side),
        perspective / orthographic, orbit by the arrows (Shift: pan). Returns True when the key was a navigation key."""
        k = rc.keys
        ctrl, shift = "Control" in modifiers, "Shift" in modifiers

        def pressed(binding: str) -> bool:
            return key.lower() == binding.lower()

        if pressed(k.frame_selection):
            self.frame_selection()
        elif pressed(k.fit_all):
            self.fit_all()
        elif pressed(k.view_front):
            self.set_view("+Y" if ctrl else "-Y")
        elif pressed(k.view_right):
            self.set_view("-X" if ctrl else "+X")
        elif pressed(k.view_top):
            self.set_view("-Z" if ctrl else "+Z")
        elif pressed(k.toggle_ortho):
            self.toggle_projection()
        else:
            steps = {k.orbit_left: (-1, 0), k.orbit_right: (1, 0), k.orbit_up: (0, 1), k.orbit_down: (0, -1)}
            for binding, (sx, sy) in steps.items():
                if pressed(binding):
                    (self.pan_step if shift else self.orbit_step)(sx, sy)
                    return True
            return False
        return True
