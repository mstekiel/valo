"""A coil spring: a tube swept along a helix. Open it with: uv run valo-studio examples/sculpts/helix.py"""

import numpy as np

from valo.meshes import tube

MATERIAL = {"shading": "pbr", "base_color": [0.8, 0.8, 0.82], "metallic": 1.0, "roughness": 0.3}
RANGES = {"coil": (0.1, 2.0), "wire": (0.01, 0.3), "turns": (0.5, 20.0), "pitch": (0.05, 1.0), "segments": (3, 64)}


def mesh(coil: float = 0.5, wire: float = 0.06, turns: float = 5.0, pitch: float = 0.25, segments: int = 16):
    """``turns`` windings of radius ``coil``, rising ``pitch`` per turn, made of wire of radius ``wire``."""
    a = np.linspace(0, 2 * np.pi * turns, max(int(48 * turns), 2))  # 48 points per turn: smooth enough
    path = np.c_[coil * np.cos(a), coil * np.sin(a), pitch * a / (2 * np.pi)]
    return tube(path, radius=wire, segments=segments)
