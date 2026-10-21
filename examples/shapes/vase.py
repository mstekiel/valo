"""A vase: a (ρ, z) profile swept around Z. Open it with: uv run valo-studio examples/shapes/vase.py"""

import numpy as np

from valo.meshes import revolve

MATERIAL = {"shading": "pbr", "base_color": [0.55, 0.2, 0.08], "roughness": 0.6}
RANGES = {"height": (0.3, 3.0), "belly": (0.0, 0.6), "neck": (0.05, 0.5), "segments": (3, 128)}


def mesh(height: float = 1.2, base: float = 0.3, belly: float = 0.25, neck: float = 0.15, segments: int = 48):
    """A vase: a profile of radius against height revolved around Z (lengths in scene units)."""
    z = np.linspace(0, height, 32)
    t = z / height
    rho = base + belly * np.sin(np.pi * t) ** 2 - (base - neck) * t**2  # wide belly, narrow neck
    outside = list(zip(rho, z, strict=True))
    # closed solid: centre of the bottom → rim of the bottom (hard edge) → outside → rim of the top → centre
    return revolve([(0, 0), outside[0], *outside, outside[-1], (0, height)], segments)
