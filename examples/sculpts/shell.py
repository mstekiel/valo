"""A seashell: a parametric surface P(u, v). Open it with: uv run valo-studio examples/sculpts/shell.py"""

import numpy as np

from valo.meshes import parametric

MATERIAL = {"shading": "pbr", "base_color": [0.9, 0.72, 0.6], "roughness": 0.25}
RANGES = {"turns": (0.5, 6.0), "growth": (0.05, 0.6), "detail": (8, 256)}


def mesh(turns: float = 3.0, growth: float = 0.25, tube: float = 0.45, detail: int = 128):
    """A spiral seashell tube that widens along ``turns`` turns."""
    def p(u, v):  # u: along the spiral, v: around the tube
        a = 2 * np.pi * turns * u
        k = np.exp(growth * a) / np.exp(growth * 2 * np.pi * turns)  # radius grows along the spiral
        ring = 1 + tube * np.cos(2 * np.pi * v)
        return k * ring * np.cos(a), k * ring * np.sin(a), k * (tube * np.sin(2 * np.pi * v) - 0.4 * a / np.pi)

    return parametric(p, detail, 32)
