"""The camera: Blender's viewport view, as state and maths (no GPU, no input handling)."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from ..config import _unknown
from ..math import look_at, orthographic_reversed_z, perspective_reversed_z

__all__ = ["PROJECTIONS", "VIEWS", "Camera"]

PROJECTIONS = ("perspective", "orthographic")
VIEWS = {  # name → (azimuth, elevation): looking from that side toward the target
    "+X": (0.0, 0.0), "-X": (180.0, 0.0), "+Y": (90.0, 0.0), "-Y": (-90.0, 0.0),
    "+Z": (-90.0, 89.99), "-Z": (-90.0, -89.99), "Iso": (-55.0, 25.0),
}


@dataclass
class Camera:
    """The 3D view (Blender's viewport view): it orbits ``target`` at ``distance`` (azimuth / elevation in
    degrees, turntable style, Z up) with a perspective (``fovy`` degrees) or orthographic (``ortho_height`` world
    units) projection. ``settings()`` / ``from_settings`` round-trip it."""
    azimuth: float = -55.0
    elevation: float = 25.0
    distance: float = 14.0
    target: tuple = (0.0, 0.0, 0.0)
    fovy: float = 35.0
    projection: str = "perspective"  # or "orthographic"
    ortho_height: float = 10.0  # world units visible vertically in orthographic mode

    def eye(self) -> np.ndarray:
        """Camera position in world space."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return np.asarray(self.target, float) + self.distance * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    def forward(self) -> np.ndarray:
        """Unit view direction, from the eye toward the target."""
        f = np.asarray(self.target, float) - self.eye()
        return f / np.linalg.norm(f)

    def settings(self) -> dict:
        """The camera as plain data (for scripts and exported settings); ``from_settings`` rebuilds it."""
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
                for f in fields(self) if not f.name.startswith("_")}

    @classmethod
    def from_settings(cls, d: dict) -> Camera:
        """A camera from ``settings()`` data; unknown keys are an error (with a "did you mean")."""
        names = [f.name for f in fields(cls) if not f.name.startswith("_")]
        for key in d:
            if key not in names:
                raise KeyError(_unknown(key, names, "camera settings"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Projection matrix for the current projection mode (reversed-Z)."""
        if self.projection == "orthographic":  # the whole scene, even behind the eye, fits in depth
            return orthographic_reversed_z(self.ortho_height, aspect, -3 * self.distance, 5 * self.distance)
        return perspective_reversed_z(self.fovy, aspect)

    def matrices(self, aspect: float):
        """(view-projection matrix, eye position) for a viewport of the given aspect ratio."""
        eye = self.eye()
        return self.projection_matrix(aspect) @ look_at(eye, self.target), eye

    def _height_at_target(self) -> float:
        return 2 * self.distance * math.tan(math.radians(self.fovy / 2))

    def set_projection(self, projection: str, radius: float | None = None) -> None:
        """Switch projection without an apparent zoom.

        With the scene's bounding ``radius`` the outline of that sphere keeps its size on screen (perspective
        enlarges the near half, so matching only the target's depth looks like zooming out); without it, the
        plane through the target keeps its size.
        """
        if projection not in PROJECTIONS:
            raise ValueError(f"unknown projection {projection!r}")
        if projection == self.projection:
            return
        t = math.tan(math.radians(self.fovy / 2))
        use_sphere = radius is not None and 0 < radius < self.distance
        if projection == "orthographic":
            self.ortho_height = 2 * radius * t / self._outline(radius) if use_sphere else self._height_at_target()
        elif radius and radius > 0:  # the distance at which the sphere's outline fills the same share of the view
            self.distance = radius / math.sin(math.atan(2 * radius / self.ortho_height * t))
        else:
            self.distance = self.ortho_height / (2 * t)
        self.projection = projection

    def _outline(self, radius: float) -> float:
        """tan of the angle from the view axis to the outline of a sphere of ``radius`` around the target."""
        return math.tan(math.asin(radius / self.distance))

    def set_fov(self, degrees: float, radius: float | None = None) -> None:
        """Change the perspective field of view and move the camera so the scene keeps its size (a dolly zoom).

        With the scene's bounding ``radius`` its outline keeps its size; otherwise the plane through the target.
        """
        old = math.tan(math.radians(self.fovy / 2))
        new = math.tan(math.radians(float(degrees) / 2))
        if radius is not None and 0 < radius < self.distance:
            share = self._outline(radius) / old  # outline height as a share of the view
            self.distance = radius / math.sin(math.atan(share * new))
        else:
            self.distance *= old / new
        self.fovy = float(degrees)

    def zoom(self, factor: float) -> None:
        """Zoom out by ``factor`` (> 1) or in (< 1)."""
        if self.projection == "orthographic":
            self.ortho_height = float(np.clip(self.ortho_height * factor, 0.05, 500.0))
        else:
            self.distance = float(np.clip(self.distance * factor, 2.0, 120.0))

    def fit(self, radius: float) -> None:
        """Move back far enough that a sphere of ``radius`` around the target fits vertically (both projections)."""
        self.distance = self.fit_distance(radius)
        self.ortho_height = self._height_at_target()

    def fit_distance(self, radius: float) -> float:
        """The distance at which a sphere of ``radius`` fits the view vertically."""
        return max(0.5, radius / math.sin(math.radians(self.fovy / 2)) * 0.95)

    def right_up(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit right and up vectors of the view (Z-up)."""
        f = self.forward()
        right = np.cross(f, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        return right, np.cross(right, f)

    def world_per_pixel(self, viewport_height: int) -> float:
        """World units per framebuffer pixel on the plane through the target."""
        height = self.ortho_height if self.projection == "orthographic" else self._height_at_target()
        return height / max(viewport_height, 1)

    def pan(self, dx: float, dy: float) -> None:
        """Slide the view by (dx, dy) world units on the target plane; the scene follows (screen y points down)."""
        right, up = self.right_up()
        self.target = tuple(float(c) for c in np.asarray(self.target, float) - right * dx + up * dy)
