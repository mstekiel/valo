"""spring: a procedural mesh. Edit, then save (Ctrl+S): the Object tab re-reads the parameters.

mesh(...) returns a valo MeshData. Every parameter needs a default and a float / int / bool type; RANGES sets the
slider ranges. Build with valo.meshes: revolve ((rho, z) profile around Z), parametric (P(u, v)), heightfield,
the primitives, transform and merge.
"""

import numpy as np

from valo import meshes

RANGES = {"bulb_window": (0.0,10), "turns": (1, 50.0), "radius": (0.1, 2.0), "t_segments": (16,512), "r_segments": (3, 32)}

MATERIAL = {
    "shading": "pbr",
    "base_color": [0.8, 0.8, 0.8],
    "metallic": 0.0,
    "roughness": 0.4,
    "ambient": 1.0,
    "emissive": [0.0, 0.0, 0.0],
}
TRANSFORM = {"translate": [0.0, 0.0, 0.0], "rotate_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def mesh(n_bulbs: int=1, bulb_window: float = 8.031, turns: float = 20.0, radius: float = 0.1, t_segments: int = 512, r_segments: int = 16):
    """A closed body of revolution with a narrower waist."""
    t = np.linspace(-1, 1,t_segments)
    phi = np.linspace(0, 2*turns*np.pi, len(t))
    
    w = np.power(np.bartlett(t_segments), bulb_window)
    osc = np.cos(n_bulbs*np.pi * t/2) if n_bulbs%2 else np.sin(n_bulbs*np.pi * t/2)
    rho = w*osc
    
    points = [
     rho*np.cos(phi),
     rho*np.sin(phi),
     5*t
    ]

    # bottom centre -> bottom rim (repeated: a hard edge) -> side -> top rim (repeated) -> top centre
    return meshes.tube(np.transpose(points), radius=radius, segments=r_segments)
