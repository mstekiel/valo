"""A ripple on a square sheet: z = f(x, y). Open it with: uv run valo-studio examples/sculpts/wave.py"""

import numpy as np

from valo.meshes import heightfield

MATERIAL = {"shading": "blinn_phong", "base_color": [0.05, 0.25, 0.5], "shininess": 96.0}
RANGES = {"amplitude": (0.0, 0.5), "frequency": (0.5, 20.0), "resolution": (8, 256)}


def mesh(amplitude: float = 0.12, frequency: float = 9.0, damping: float = 2.0, resolution: int = 96):
    """A damped circular ripple on a 2 × 2 sheet."""
    def z(x, y):
        r = np.hypot(x, y)
        return amplitude * np.cos(frequency * r) * np.exp(-damping * r)

    return heightfield(z, size=2.0, n=resolution)
