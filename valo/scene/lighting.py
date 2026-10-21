"""Scene lighting: one directional light, either a headlight (fixed relative to the camera) or fixed in the world.

Plus a global hemisphere ambient: sky colour from above, ground colour from below, one intensity.

Angles say where the light comes FROM, in degrees:
- **headlight:** in the camera's frame. Azimuth 0 / elevation 0 is straight from the viewer;
  negative azimuth is from the left, positive elevation from above.
- **fixed:** in world coordinates, Z-up. Azimuth is measured from +X toward +Y, elevation above the XY plane.

``set_mode`` converts the angles, so switching never makes the light jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from ..config import _unknown, rc

MODES = ("headlight", "fixed")


def _camera_basis(camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, back) unit vectors of the camera, as in ``geometry.look_at`` (Z-up)."""
    f = camera.forward()
    right = np.cross(f, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    return right, np.cross(right, f), -f


@dataclass
class Lighting:
    """The scene's directional light (headlight or fixed) and its global hemisphere ambient."""
    # Defaults come from rc.lighting at construction time.
    mode: str = field(default_factory=lambda: rc.lighting.mode)
    azimuth: float = field(default_factory=lambda: rc.lighting.azimuth)
    elevation: float = field(default_factory=lambda: rc.lighting.elevation)
    color: tuple = field(default_factory=lambda: rc.lighting.color)  # linear RGB
    intensity: float = field(default_factory=lambda: rc.lighting.intensity)
    sky: tuple = field(default_factory=lambda: rc.lighting.sky)  # ambient from above (linear), also the reflected sky
    ground: tuple = field(default_factory=lambda: rc.lighting.ground)  # ...and from below
    ambient: float = field(default_factory=lambda: rc.lighting.ambient)  # each material scales it by its `ambient`

    def direction(self, camera) -> np.ndarray:
        """Unit world vector pointing toward the light."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        if self.mode == "fixed":
            return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        right, up, back = _camera_basis(camera)
        return math.cos(el) * math.sin(az) * right + math.sin(el) * up + math.cos(el) * math.cos(az) * back

    def set_mode(self, mode: str, camera) -> None:
        """Switch between "headlight" and "fixed", keeping the light where it is right now."""
        if mode not in MODES:
            raise ValueError(f"unknown light mode {mode!r}")
        d = self.direction(camera)
        if mode == "fixed":
            x, y, z = d
        else:
            right, up, back = _camera_basis(camera)
            x, y, z = d @ back, d @ right, d @ up  # (cos·cos az, cos·sin az, sin el) in the camera frame
        self.mode = mode
        self.azimuth = math.degrees(math.atan2(y, x))
        self.elevation = math.degrees(math.asin(float(np.clip(z, -1.0, 1.0))))

    def settings(self) -> dict:
        """The lighting as plain data (for scripts and exported settings); ``from_settings`` rebuilds it."""
        return {f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v) for f in fields(self)}

    @classmethod
    def from_settings(cls, d: dict) -> Lighting:
        """Lighting from ``settings()`` data (missing keys take the rc defaults; unknown keys are an error)."""
        names = [f.name for f in fields(cls)]
        for key in d:
            if key not in names:
                raise KeyError(_unknown(key, names, "lighting settings"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def radiance(self) -> np.ndarray:
        """Light colour times intensity (linear RGB), as sent to the shaders."""
        return np.asarray(self.color, float) * self.intensity
